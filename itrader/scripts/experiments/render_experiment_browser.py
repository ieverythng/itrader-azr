"""Build the repository-owned iTrader experiment browser.

The registry stores provenance and comparability metadata only. Metric values are
read from the authoritative experiment summaries every time the browser is built,
which prevents a second hand-maintained result ledger from drifting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path, PureWindowsPath
from typing import Any, Callable

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError

from itrader.scripts.experiments.browser_traces import load_run_traces
from itrader.rl_lib.evaluation.evidence_values import action_fractions
from itrader.rl_lib.evaluation.evaluation_disposition import (
    build_evaluation_disposition,
)


class RegistryError(ValueError):
    """Raised when the experiment registry cannot be resolved safely."""


def _load_json(path: Path) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise RegistryError(f"duplicate JSON key: {key} in {path}")
            value[key] = item
        return value

    def reject_non_finite(token: str) -> None:
        raise RegistryError(f"non-finite JSON number {token} in {path}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_non_finite,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryError(f"could not read JSON source {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RegistryError(f"JSON source must contain an object: {path}")
    _reject_non_finite_values(value, path=path)
    return value


def _reject_non_finite_values(value: Any, *, path: Path) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise RegistryError(f"non-finite JSON number in {path}")
    if isinstance(value, dict):
        for item in value.values():
            _reject_non_finite_values(item, path=path)
    elif isinstance(value, list):
        for item in value:
            _reject_non_finite_values(item, path=path)


def _repo_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _member_from_text(value: str) -> str:
    match = re.search(r"_000(\d+)(?:_|$)", value)
    return f"000{match.group(1)}" if match else "—"


def _checkpoint_receipt(
    source_path: Path,
    checkpoint: dict[str, Any],
    source: dict[str, Any],
    repo_root: Path,
) -> tuple[str, str, str, dict[str, Any]]:
    checkpoint_steps = _required_positive_int(
        checkpoint, "pbt_progress_steps", "g0_closure_v1 checkpoint"
    )
    evidence_dir = source_path.parent / f"checkpoint_{checkpoint_steps}"
    processes = ("process_a", "process_b")
    receipts: list[dict[str, Any]] = []
    episode_hashes: list[str] = []
    state_hashes: list[str] = []
    for process in processes:
        replay_dir = evidence_dir / process / "replay"
        state_dir = evidence_dir / process / "state"
        receipt_paths = sorted(
            path
            for path in replay_dir.glob("*.json")
            if not path.name.endswith(".episodes.json")
        )
        episode_paths = sorted(replay_dir.glob("*.episodes.jsonl"))
        state_paths = sorted(state_dir.glob("*.state_steps.jsonl"))
        if len(receipt_paths) != 1 or len(episode_paths) != 1 or len(state_paths) != 1:
            raise RegistryError(
                "G0 artifact-pair provenance requires one receipt, episode file, "
                f"and state-step file per replay directory at {checkpoint_steps} steps"
            )
        receipts.append(_load_json(receipt_paths[0]))
        episode_hashes.append(_sha256(episode_paths[0]))
        state_hashes.append(_sha256(state_paths[0]))

    for field in (
        "checkpoint",
        "trial",
        "source_panel_sha256",
        "evaluation_panel_sha256",
    ):
        values = [
            _required_text(receipt, field, "G0 replay receipt") for receipt in receipts
        ]
        if values[0] != values[1]:
            raise RegistryError(
                f"G0 artifact-pair receipts disagree on {field} at {checkpoint_steps} steps"
            )

    source_panel = source.get("source_panel")
    fresh_panel = source.get("fresh_panel")
    if not isinstance(source_panel, dict) or not isinstance(fresh_panel, dict):
        raise RegistryError(
            "g0_closure_v1 requires source_panel and fresh_panel objects"
        )
    expected_source_hash = _required_text(source_panel, "sha256", "G0 source panel")
    expected_fresh_hash = _required_text(fresh_panel, "sha256", "G0 fresh panel")
    if receipts[0]["source_panel_sha256"] != expected_source_hash:
        raise RegistryError(
            f"G0 source panel receipt hash mismatch at {checkpoint_steps} steps"
        )
    if receipts[0]["evaluation_panel_sha256"] != expected_fresh_hash:
        raise RegistryError(
            f"G0 evaluation panel receipt hash mismatch at {checkpoint_steps} steps"
        )

    expected_results = {
        "evaluation/net_return_mean": _required_finite_number(
            checkpoint, "fresh_net_return", f"G0 checkpoint {checkpoint_steps}"
        ),
        "evaluation/net_sharpe_per_sqrt_bar_mean": _required_finite_number(
            checkpoint,
            "fresh_sharpe_per_sqrt_bar",
            f"G0 checkpoint {checkpoint_steps}",
        ),
        "evaluation/turnover_mean": _required_finite_number(
            checkpoint, "fresh_turnover", f"G0 checkpoint {checkpoint_steps}"
        ),
    }
    expected_actions = _action_fractions(checkpoint, checkpoint_steps)
    for process, receipt in zip(processes, receipts, strict=True):
        summary = receipt.get("summary")
        if not isinstance(summary, dict):
            raise RegistryError(
                f"G0 replay receipt for {process} requires a summary object"
            )
        for metric, expected in expected_results.items():
            actual = _required_finite_number(summary, metric, f"G0 {process} receipt")
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-15):
                raise RegistryError(
                    f"G0 {process} receipt disagrees on {metric} at "
                    f"{checkpoint_steps} steps"
                )
        for action_index, expected in enumerate(expected_actions):
            field = f"custom_metrics/act_{action_index}_frac_mean"
            actual = _required_finite_number(summary, field, f"G0 {process} receipt")
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-15):
                raise RegistryError(
                    f"G0 {process} receipt disagrees on {field} at "
                    f"{checkpoint_steps} steps"
                )

    expected_episode_hash = _required_sha256(
        checkpoint, "two_process_episode_sha256", "G0 checkpoint"
    )
    expected_state_hash = _required_sha256(
        checkpoint, "two_process_state_steps_sha256", "G0 checkpoint"
    )
    if len(set(episode_hashes)) != 1 or episode_hashes[0] != expected_episode_hash:
        raise RegistryError(
            f"G0 episode artifact parity mismatch at {checkpoint_steps} steps"
        )
    if len(set(state_hashes)) != 1 or state_hashes[0] != expected_state_hash:
        raise RegistryError(
            f"G0 state artifact parity mismatch at {checkpoint_steps} steps"
        )

    attributes = {
        "artifact_pair_parity_passed": True,
        "independent_process_provenance": "not-recorded-in-receipts",
    }
    return (
        receipts[0]["checkpoint"],
        _repo_path(evidence_dir, repo_root),
        receipts[0]["trial"],
        attributes,
    )


def _action_fractions(checkpoint: dict[str, Any], steps: int) -> list[float]:
    try:
        return action_fractions(checkpoint.get("fresh_action_fractions"))
    except ValueError as exc:
        raise RegistryError(
            f"G0 checkpoint {steps} must contain three finite action fractions "
            "in [0, 1] that sum to one"
        ) from exc


def _required_list(
    source: dict[str, Any], field: str, adapter_name: str
) -> list[dict[str, Any]]:
    value = source.get(field)
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, dict) for item in value)
    ):
        raise RegistryError(
            f"{adapter_name} source requires a non-empty {field} object list"
        )
    return value


def _required_finite_number(
    source: dict[str, Any], field: str, context: str
) -> int | float:
    value = source.get(field)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise RegistryError(f"{context} requires finite numeric {field}")
    return value


def _required_positive_int(source: dict[str, Any], field: str, context: str) -> int:
    value = source.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RegistryError(f"{context} requires positive integer {field}")
    return value


def _required_nonnegative_int(source: dict[str, Any], field: str, context: str) -> int:
    value = source.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RegistryError(f"{context} requires non-negative integer {field}")
    return value


def _required_boolean(source: dict[str, Any], field: str, context: str) -> bool:
    value = source.get(field)
    if not isinstance(value, bool):
        raise RegistryError(f"{context} requires boolean {field}")
    return value


def _required_sha256(source: dict[str, Any], field: str, context: str) -> str:
    value = _required_text(source, field, context)
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise RegistryError(f"{context} requires lowercase SHA-256 {field}")
    return value


def _required_text(source: dict[str, Any], field: str, context: str) -> str:
    value = source.get(field)
    if not isinstance(value, str) or not value.strip():
        raise RegistryError(f"{context} requires non-empty {field}")
    return value


def _required_int_list(source: dict[str, Any], field: str, context: str) -> list[int]:
    value = source.get(field)
    if (
        not isinstance(value, list)
        or not value
        or any(
            not isinstance(item, int) or isinstance(item, bool) or item <= 0
            for item in value
        )
    ):
        raise RegistryError(
            f"{context} requires a non-empty positive integer {field} list"
        )
    return value


def _adapt_g0_closure(
    experiment_id: str,
    source: dict[str, Any],
    source_path: Path,
    repo_root: Path,
) -> list[dict[str, Any]]:
    member = str(source.get("member", "—"))
    checkpoints = _required_list(source, "checkpoints", "g0_closure_v1")
    decision = source.get("decision")
    if not isinstance(decision, dict):
        raise RegistryError("g0_closure_v1 requires a decision object")
    decision_fields = {
        "two_process_determinism_passed",
        "fresh_panel_economic_gate_passed",
        "r0_plus_r4_v1_architecture_qualified",
        "r0_plus_r4_v1_economically_promoted",
        "repeat_same_panel_pbt",
        "next_boundary",
    }
    unknown_decisions = set(decision) - decision_fields
    if unknown_decisions:
        raise RegistryError(
            f"g0_closure_v1 decision contains unknown fields: {sorted(unknown_decisions)}"
        )
    decision_attributes = {
        "source_claimed_two_process_determinism": _required_boolean(
            decision, "two_process_determinism_passed", "G0 decision"
        ),
        "fresh_panel_economic_gate_passed": _required_boolean(
            decision, "fresh_panel_economic_gate_passed", "G0 decision"
        ),
        "r0_plus_r4_v1_architecture_qualified": _required_boolean(
            decision, "r0_plus_r4_v1_architecture_qualified", "G0 decision"
        ),
        "r0_plus_r4_v1_economically_promoted": _required_boolean(
            decision, "r0_plus_r4_v1_economically_promoted", "G0 decision"
        ),
        "repeat_same_panel_pbt": _required_boolean(
            decision, "repeat_same_panel_pbt", "G0 decision"
        ),
        "next_boundary": _required_text(decision, "next_boundary", "G0 decision"),
    }
    records: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        steps = _required_positive_int(
            checkpoint, "pbt_progress_steps", "g0_closure_v1 checkpoint"
        )
        if checkpoint.get("two_process_identity") is not True:
            raise RegistryError(
                f"g0_closure_v1 checkpoint {steps} requires two-process identity"
            )
        checkpoint_path, evidence_path, run_id, provenance_attributes = (
            _checkpoint_receipt(source_path, checkpoint, source, repo_root)
        )
        metrics = {
            field: _required_finite_number(checkpoint, field, f"G0 checkpoint {steps}")
            for field in (
                "optimization_cost_bps",
                "source_net_return",
                "source_sharpe_per_sqrt_bar",
                "fresh_net_return",
                "fresh_sharpe_per_sqrt_bar",
                "fresh_turnover",
                "top_two_probability_margin_p01",
                "explained_variance",
                "approx_kl",
            )
        }
        action_fractions = _action_fractions(checkpoint, steps)
        records.append(
            {
                "record_key": f"{experiment_id}:{member}:{steps}",
                "label": f"member {member} · {steps:,} env steps",
                "member": member,
                "run_id": run_id,
                "pbt_progress_steps": steps,
                "checkpoint_path": checkpoint_path,
                "evidence_path": evidence_path,
                "metrics": {
                    "optimization_cost_bps": metrics["optimization_cost_bps"],
                    "source_net_return": metrics["source_net_return"],
                    "source_sharpe_per_sqrt_bar": metrics["source_sharpe_per_sqrt_bar"],
                    "fresh_net_return": metrics["fresh_net_return"],
                    "fresh_sharpe_per_sqrt_bar": metrics["fresh_sharpe_per_sqrt_bar"],
                    "fresh_turnover": metrics["fresh_turnover"],
                    "short_fraction": action_fractions[0],
                    "flat_fraction": action_fractions[1],
                    "long_fraction": action_fractions[2],
                    "probability_margin_p01": metrics["top_two_probability_margin_p01"],
                    "explained_variance": metrics["explained_variance"],
                    "approx_kl": metrics["approx_kl"],
                },
                "attributes": {**provenance_attributes, **decision_attributes},
            }
        )
    return records


def _adapt_ablation_summary(
    experiment_id: str,
    source: dict[str, Any],
    source_path: Path,
    repo_root: Path,
) -> list[dict[str, Any]]:
    primary_metric = source.get("primary_metric")
    if not isinstance(primary_metric, str) or not primary_metric.strip():
        raise RegistryError(
            "ablation_summary_v1 source requires a non-empty primary_metric"
        )
    records = []
    for index, result in enumerate(
        _required_list(source, "results", "ablation_summary_v1")
    ):
        context = f"ablation_summary_v1 result {index}"
        final_metric = _required_finite_number(result, "final_metric", context)
        series = str(result.get("series_id") or result.get("variant_id") or index)
        trial_dir = _required_text(result, "trial_dir", context)
        run_id = PureWindowsPath(trial_dir).name
        member = _member_from_text(run_id)
        records.append(
            {
                "record_key": f"{experiment_id}:{series}",
                "label": series,
                "display_label": (
                    f"{result.get('variant_id', series.split(':')[0])} · member {member}"
                    if ":" in series else series
                ),
                "member": member,
                "run_id": run_id,
                "pbt_progress_steps": None,
                "checkpoint_path": None,
                "evidence_path": None,
                "metrics": {primary_metric: final_metric},
                "attributes": {},
                "trial_dir": trial_dir,
                "stable_report_csv": result.get("stable_report_csv"),
            }
        )
    try:
        traces = load_run_traces(
            source_path.parent,
            {record["label"]: record["metrics"][primary_metric] for record in records},
            primary_metric,
        )
    except ValueError as exc:
        raise RegistryError(f"invalid run traces for {experiment_id}: {exc}") from exc
    for record in records:
        trace = traces.get(record["label"])
        if trace:
            for item in trace["sources"]:
                item["path"] = _repo_path(source_path.parent / item["file"], repo_root)
            record["trace"] = trace
            record["metrics"].update(trace["terminal_metrics"])
            if trace["clock"] == "pbt_progress_steps":
                record["pbt_progress_steps"] = trace["terminal_step"]
    return records


def _adapt_terminal_gate(
    experiment_id: str,
    source: dict[str, Any],
    source_path: Path,
    repo_root: Path,
) -> list[dict[str, Any]]:
    del source_path, repo_root
    arms = source.get("arms")
    if (
        not isinstance(arms, dict)
        or not arms
        or not all(isinstance(arm, dict) for arm in arms.values())
    ):
        raise RegistryError("terminal_gate_v1 source requires a non-empty arms object")
    review_steps = _required_positive_int(
        source, "review_steps", "terminal_gate_v1 source"
    )
    records = []
    for arm_id, arm in arms.items():
        context = f"terminal_gate_v1 arm {arm_id}"
        terminal_net_return = _required_finite_number(
            arm, "terminal_net_return", context
        )
        terminal_sharpe = _required_finite_number(
            arm, "terminal_sharpe_per_sqrt_bar", context
        )
        exploit_count = _required_nonnegative_int(arm, "exploit_count", context)
        lineage_receipts = _required_nonnegative_int(
            arm, "verified_lineage_receipt_count", context
        )
        experiment_name = _required_text(arm, "experiment_name", context)
        decision = _required_text(arm, "decision", context)
        terminal_action = _required_text(arm, "terminal_action", context)
        terminal_steps = _required_int_list(arm, "terminal_steps", context)
        if "evaluation_disposition" in arm:
            raw_disposition = arm["evaluation_disposition"]
            expected = {
                "schema_version",
                "evidence_validity",
                "metric_availability",
                "inference_qualification",
                "promotion_eligibility",
            }
            if (not isinstance(raw_disposition, dict)
                    or set(raw_disposition) != expected
                    or type(raw_disposition.get("schema_version")) is not int
                    or raw_disposition["schema_version"] != 1):
                raise RegistryError(f"{context} has an invalid evaluation_disposition schema")
            try:
                disposition = build_evaluation_disposition(
                    evidence_validity=raw_disposition["evidence_validity"],
                    metric_availability=raw_disposition["metric_availability"],
                    inference_qualification=raw_disposition["inference_qualification"],
                    promotion_eligibility=raw_disposition["promotion_eligibility"],
                )
            except (TypeError, ValueError) as exc:
                raise RegistryError(
                    f"{context} has an invalid evaluation_disposition: {exc}"
                ) from exc
        else:
            disposition = None
        record = {
            "record_key": f"{experiment_id}:{arm_id}:terminal",
            "label": arm_id.replace("_", " "),
            "member": "population",
            "run_id": experiment_name,
            "pbt_progress_steps": None,
            "checkpoint_path": None,
            "evidence_path": None,
            "metrics": {
                "net_return": terminal_net_return,
                "sharpe_per_sqrt_bar": terminal_sharpe,
                "exploit_count": exploit_count,
                "lineage_receipts": lineage_receipts,
            },
            "attributes": {
                "decision": decision,
                "terminal_action": terminal_action,
                "review_threshold_steps": review_steps,
                "terminal_steps": terminal_steps,
            },
        }
        if disposition is not None:
            record["evaluation_disposition"] = disposition
        records.append(record)
    return records


ADAPTERS: dict[
    str,
    Callable[[str, dict[str, Any], Path, Path], list[dict[str, Any]]],
] = {
    "g0_closure_v1": _adapt_g0_closure,
    "ablation_summary_v1": _adapt_ablation_summary,
    "terminal_gate_v1": _adapt_terminal_gate,
}

_DISPOSITION_ATTRIBUTE_UNITS = {
    "evaluation_evidence_validity_status": "text",
    "evaluation_evidence_validity_reason": "text",
    "evaluation_net_return_availability_status": "text",
    "evaluation_net_return_availability_reason": "text",
    "evaluation_sharpe_availability_status": "text",
    "evaluation_sharpe_availability_reason": "text",
    "evaluation_inference_qualification_status": "text",
    "evaluation_inference_qualification_reason": "text",
    "evaluation_promotion_eligibility_status": "text",
    "evaluation_promotion_eligibility_reason": "text",
}


def _disposition_attributes(disposition: dict[str, Any]) -> dict[str, str]:
    values = {
        "evaluation_evidence_validity": disposition["evidence_validity"],
        "evaluation_net_return_availability": disposition["metric_availability"]["net_return"],
        "evaluation_sharpe_availability": disposition["metric_availability"][
            "net_sharpe_per_sqrt_bar"
        ],
        "evaluation_inference_qualification": disposition["inference_qualification"],
        "evaluation_promotion_eligibility": disposition["promotion_eligibility"],
    }
    return {
        field: value
        for name, item in values.items()
        for field, value in (
            (f"{name}_status", item["status"]),
            (f"{name}_reason", item["reason"] or ""),
        )
    }


def _comparability_id(
    factors: dict[str, str], identities: dict[str, str | None]
) -> str:
    serialized = json.dumps(
        {"factors": factors, "identities": identities},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]


def _build_comparisons(
    experiments: list[dict[str, Any]],
    *,
    factor_names: list[str],
    identity_names: list[str],
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for left_index, left in enumerate(experiments):
        for right in experiments[left_index + 1 :]:
            differences = [
                factor
                for factor in factor_names
                if left["factors"][factor] != right["factors"][factor]
            ]
            identity_differences = [
                identity
                for identity in identity_names
                if left["identities"].get(identity) is not None
                and right["identities"].get(identity) is not None
                and left["identities"][identity] != right["identities"][identity]
            ]
            missing_identities = [
                identity
                for identity in identity_names
                if left["identities"].get(identity) is None
                or right["identities"].get(identity) is None
            ]
            comparisons.append(
                {
                    "left": left["id"],
                    "right": right["id"],
                    "directly_comparable": not (
                        differences or identity_differences or missing_identities
                    ),
                    "different_factors": differences,
                    "different_identities": identity_differences,
                    "missing_identities": missing_identities,
                }
            )
    return comparisons


def _load_registry_contract() -> (
    tuple[dict[str, Any], list[str], list[str], dict[str, str], dict[str, str]]
):
    schema_path = (
        Path(__file__).with_name("schemas") / "experiment_registry.schema.json"
    )
    schema = _load_json(schema_path)
    factor_names = schema.get("x-comparison-factors")
    identity_names = schema.get("x-comparison-identities")
    metric_units = schema.get("x-metric-units")
    attribute_units = schema.get("x-attribute-units")
    if (
        not isinstance(factor_names, list)
        or not isinstance(identity_names, list)
        or not isinstance(metric_units, dict)
        or not isinstance(attribute_units, dict)
    ):
        raise RegistryError(
            "registry schema must declare comparison factors, identities, and units"
        )
    factor_properties = (
        schema.get("properties", {})
        .get("experiments", {})
        .get("items", {})
        .get("properties", {})
        .get("factors", {})
        .get("properties", {})
    )
    if set(factor_properties) != set(factor_names):
        raise RegistryError(
            "registry schema factor properties must exactly match comparison factors"
        )
    allowed_units = {
        "boolean",
        "bps",
        "count",
        "count_list",
        "decimal",
        "ratio_percent",
        "text",
    }
    unknown_units = (set(metric_units.values()) | set(attribute_units.values())) - (
        allowed_units
    )
    if unknown_units:
        raise RegistryError(f"registry schema contains unknown units: {unknown_units}")
    return schema, factor_names, identity_names, metric_units, attribute_units


def _validate_registry(registry: dict[str, Any], schema: dict[str, Any]) -> None:
    try:
        Draft202012Validator(schema, format_checker=FormatChecker()).validate(registry)
    except ValidationError as exc:
        location = ".".join(str(part) for part in exc.absolute_path) or "root"
        raise RegistryError(
            f"registry schema violation at {location}: {exc.message}"
        ) from exc


def _source_identities(
    adapter_name: str,
    source: dict[str, Any],
    identity_names: list[str],
) -> dict[str, str | None]:
    identities: dict[str, str | None] = {}
    for identity in identity_names:
        value = source.get(identity)
        normalized = value.strip() if isinstance(value, str) and value.strip() else None
        if (
            normalized is not None
            and identity.endswith("_sha256")
            and re.fullmatch(r"[0-9a-f]{64}", normalized) is None
        ):
            raise RegistryError(
                f"source requires lowercase SHA-256 {identity}, got {normalized!r}"
            )
        identities[identity] = normalized
    if adapter_name == "g0_closure_v1":
        panel = source.get("fresh_panel")
        panel_hash = panel.get("sha256") if isinstance(panel, dict) else None
        if isinstance(panel_hash, str) and panel_hash.strip():
            normalized_panel_hash = panel_hash.strip()
            if re.fullmatch(r"[0-9a-f]{64}", normalized_panel_hash) is None:
                raise RegistryError("G0 fresh panel requires lowercase SHA-256 sha256")
            identities["evaluation_panel_sha256"] = normalized_panel_hash
    return identities


def _validate_records(
    experiment_id: str,
    records: list[dict[str, Any]],
    *,
    metric_units: dict[str, str],
    attribute_units: dict[str, str],
) -> list[dict[str, Any]]:
    if not records:
        raise RegistryError(f"adapter produced no records for {experiment_id}")
    keys = [record.get("record_key") for record in records]
    if len(keys) != len(set(keys)):
        raise RegistryError(f"duplicate record key in {experiment_id}")
    for record in records:
        unknown_metrics = set(record.get("metrics", {})) - set(metric_units)
        unknown_attributes = set(record.get("attributes", {})) - set(attribute_units)
        if unknown_metrics:
            raise RegistryError(
                f"unknown metric unit contract in {experiment_id}: "
                f"{sorted(unknown_metrics)}"
            )
        if unknown_attributes:
            raise RegistryError(
                f"unknown attribute unit contract in {experiment_id}: "
                f"{sorted(unknown_attributes)}"
            )
    return records


def _platform_timing(source: dict[str, Any], context: str) -> dict[str, Any]:
    median = _required_finite_number(source, "median_ms", context)
    p10 = _required_finite_number(source, "p10_ms", context)
    p90 = _required_finite_number(source, "p90_ms", context)
    sample_count = _required_positive_int(source, "sample_count", context)
    if not 0 < p10 <= median <= p90:
        raise RegistryError(f"{context} requires 0 < p10 <= median <= p90")
    return {
        "median_ms": median,
        "p10_ms": p10,
        "p90_ms": p90,
        "sample_count": sample_count,
    }


def _load_platform_qualification(
    entry: dict[str, Any], repo_root: Path
) -> dict[str, Any]:
    """Resolve a digest-bound kernel report without making an economic record."""

    context = f"platform qualification {entry['id']}"
    if entry.get("adapter") != "cuda_kernel_qualification_v1":
        raise RegistryError(f"{context} has an unsupported adapter")
    source_ref = _required_text(entry, "source", context)
    source_path = Path(source_ref)
    if source_path.is_absolute() or PureWindowsPath(source_ref).is_absolute():
        raise RegistryError(f"{context} source must be repository-relative")
    repo_root = repo_root.resolve()
    resolved_path = (repo_root / source_path).resolve()
    try:
        resolved_path.relative_to(repo_root)
    except ValueError as exc:
        raise RegistryError(f"{context} source must stay within the repository") from exc
    if not resolved_path.is_file():
        raise RegistryError(f"{context} source does not exist: {source_ref}")
    expected_digest = _required_sha256(entry, "source_sha256", context)
    actual_digest = _sha256(resolved_path)
    if actual_digest != expected_digest:
        raise RegistryError(
            f"{context} source SHA-256 mismatch: expected {expected_digest}, "
            f"got {actual_digest}"
        )

    source = _load_json(resolved_path)
    # JSON booleans are Python ints, but are not valid report schema versions.
    if type(source.get("schema")) is not int or source["schema"] != 1:
        raise RegistryError(f"{context} requires report schema 1")
    if _required_text(source, "arithmetic_autograd", context) != "pass":
        raise RegistryError(f"{context} does not report arithmetic/autograd pass")
    verdict = _required_text(source, "verdict", context)
    if verdict != "synthetic_kernel_qualification_passed":
        raise RegistryError(f"{context} has an unsupported report verdict: {verdict}")

    packages = source.get("packages")
    if not isinstance(packages, dict):
        raise RegistryError(f"{context} requires reported package metadata")
    package_metadata = {
        name: _required_text(packages, name, context)
        for name in ("torch", "ray", "gymnasium", "numpy")
    }
    runtime = {
        "torch": _required_text(source, "torch", context),
        "cuda": _required_text(source, "cuda", context),
        "cudnn": _required_positive_int(source, "cudnn", context),
        "device": _required_text(source, "device", context),
        "threads": _required_positive_int(source, "threads", context),
        "packages": package_metadata,
    }

    cases = _required_list(source, "cases", context)
    batch_sizes = [
        _required_positive_int(case, "batch", f"{context} case") for case in cases
    ]
    if len(batch_sizes) != 2 or set(batch_sizes) != {1, 128}:
        raise RegistryError(f"{context} requires exactly the B=1 and B=128 cases")

    normalized_cases = []
    for case, batch_size in sorted(zip(cases, batch_sizes), key=lambda pair: pair[1]):
        case_context = f"{context} B={batch_size}"
        operators = case.get("cudnn_operators")
        if (
            not isinstance(operators, list)
            or not all(isinstance(operator, str) for operator in operators)
            or "aten::_cudnn_rnn" not in operators
        ):
            raise RegistryError(f"{case_context} does not report cuDNN RNN execution")
        errors = case.get("max_absolute_errors")
        if not isinstance(errors, dict):
            raise RegistryError(f"{case_context} requires max_absolute_errors")
        normalized_errors = {}
        for name in ("output", "hidden", "cell", "input_gradient", "parameter_gradient"):
            error = _required_finite_number(errors, name, case_context)
            if error < 0:
                raise RegistryError(f"{case_context} requires non-negative {name} error")
            normalized_errors[name] = error

        raw_timings = case.get("timings")
        if not isinstance(raw_timings, dict):
            raise RegistryError(f"{case_context} requires CPU and CUDA timings")
        timings: dict[str, dict[str, Any]] = {}
        for device_name in ("cpu", "cuda"):
            device_timings = raw_timings.get(device_name)
            if not isinstance(device_timings, dict):
                raise RegistryError(f"{case_context} requires {device_name} timings")
            timings[device_name] = {}
            for workload in ("inference", "forward_backward"):
                workload_timing = device_timings.get(workload)
                if not isinstance(workload_timing, dict):
                    raise RegistryError(
                        f"{case_context} requires {device_name} {workload} timing"
                    )
                timings[device_name][workload] = _platform_timing(
                    workload_timing, f"{case_context} {device_name} {workload}"
                )

        normalized_cases.append(
            {
                "batch": batch_size,
                "window_bars": _required_positive_int(case, "T", case_context),
                "feature_count": _required_positive_int(case, "input", case_context),
                "hidden_size": _required_positive_int(case, "hidden", case_context),
                "max_absolute_errors": normalized_errors,
                "cudnn_operators": operators,
                "timings": timings,
            }
        )
    return {
        "id": _required_text(entry, "id", "platform qualification registry"),
        "title": _required_text(entry, "title", context),
        "date": _required_text(entry, "date", context),
        "source": _repo_path(resolved_path, repo_root),
        "source_sha256": actual_digest,
        "model": "Bare LSTM reference workload",
        "runtime": runtime,
        "arithmetic_autograd": "pass",
        "verdict": verdict,
        "cases": normalized_cases,
        "limitations": (
            "Synthetic bare-LSTM kernel qualification only. It does not establish "
            "iTrader compatibility, PPO throughput, economic performance, or "
            "promotion eligibility."
        ),
    }


def build_browser_payload(
    registry_path: Path | str, *, repo_root: Path | str
) -> dict[str, Any]:
    """Resolve a registry into browser-ready records from authoritative sources."""

    registry_path = Path(registry_path)
    repo_root = Path(repo_root)
    registry = _load_json(registry_path)
    (
        schema,
        factor_names,
        identity_names,
        metric_units,
        attribute_units,
    ) = _load_registry_contract()
    _validate_registry(registry, schema)
    experiments = registry.get("experiments")

    seen_ids: set[str] = set()
    resolved: list[dict[str, Any]] = []
    for entry in experiments:
        experiment_id = str(entry.get("id", "")).strip()
        if experiment_id in seen_ids:
            raise RegistryError(f"duplicate experiment id: {experiment_id}")
        seen_ids.add(experiment_id)

        adapter_name = str(entry.get("adapter", ""))
        adapter = ADAPTERS.get(adapter_name)
        if adapter is None:
            raise RegistryError(f"unknown adapter for {experiment_id}: {adapter_name}")
        source_path = repo_root / str(entry.get("source", ""))
        if not source_path.is_file():
            raise RegistryError(
                f"source does not exist for {experiment_id}: {source_path}"
            )

        factors = entry["factors"]
        normalized_factors = {factor: str(factors[factor]) for factor in factor_names}
        source = _load_json(source_path)
        identities = _source_identities(adapter_name, source, identity_names)
        adapted_records = adapter(experiment_id, source, source_path, repo_root)
        if any("evaluation_disposition" in record for record in adapted_records):
            attribute_units.update(_DISPOSITION_ATTRIBUTE_UNITS)
            for record in adapted_records:
                disposition = record.pop("evaluation_disposition", None)
                if disposition is not None:
                    record["attributes"].update(_disposition_attributes(disposition))
        records = _validate_records(
            experiment_id,
            adapted_records,
            metric_units=metric_units,
            attribute_units=attribute_units,
        )
        for record in records:
            record["source"] = _repo_path(source_path, repo_root)
        run_source = entry.get("run_source")
        if run_source:
            run_records, identities = _adapt_population_runs(
                entry, source, identities, repo_root, identity_names,
            )
            records = _validate_records(
                experiment_id, records + run_records,
                metric_units=metric_units, attribute_units=attribute_units,
            )
        resolved.append(
            {
                "id": experiment_id,
                "title": str(entry.get("title", experiment_id)),
                "date": str(entry.get("date", "")),
                "status": str(entry.get("status", "evidence")),
                "summary": str(entry.get("summary", "")),
                "source": _repo_path(source_path, repo_root),
                "run_source": run_source,
                "report": str(entry.get("report", "")),
                "legacy_dashboard": entry.get("legacy_dashboard"),
                "factors": normalized_factors,
                "identities": identities,
                "comparability_id": _comparability_id(normalized_factors, identities),
                "records": records,
            }
        )

    platform_entries = registry.get("platform_qualifications", [])
    platform_ids: set[str] = set()
    platform_qualifications = []
    for entry in platform_entries:
        platform_id = str(entry.get("id", "")).strip()
        if platform_id in platform_ids:
            raise RegistryError(f"duplicate platform qualification id: {platform_id}")
        platform_ids.add(platform_id)
        platform_qualifications.append(
            _load_platform_qualification(entry, repo_root)
        )

    return {
        "schema_version": registry["schema_version"],
        "title": str(registry.get("title", "iTrader Experiment Browser")),
        "generated_from": _repo_path(registry_path, repo_root),
        "metric_units": metric_units,
        "attribute_units": attribute_units,
        "experiments": resolved,
        "platform_qualifications": platform_qualifications,
        "comparisons": _build_comparisons(
            resolved,
            factor_names=factor_names,
            identity_names=identity_names,
        ),
    }


def _adapt_population_runs(entry, source, identities, repo_root, identity_names):
    """Bind an optional individual-run summary to its declared population."""
    if entry["adapter"] != "terminal_gate_v1":
        raise RegistryError("run_source is only supported beside terminal_gate_v1")
    run_path = repo_root / entry["run_source"]
    run_summary = _load_json(run_path)
    arm_names = {arm["experiment_name"] for arm in source["arms"].values()}
    if len(arm_names) != len(source["arms"]):
        raise RegistryError("run_source requires unique arm population identities")
    members = _required_positive_int(source, "expected_members", "terminal gate")
    results = _required_list(run_summary, "results", "run_source")
    result_names = [_required_text(result, "experiment_name", "run_source") for result in results]
    if set(result_names) != arm_names or any(result_names.count(name) != members for name in arm_names):
        raise RegistryError("run_source must match all declared arms and member counts")
    run_records = _adapt_ablation_summary(entry["id"], run_summary, run_path, repo_root)
    # Trial names are Windows-produced identities even when the portable browser
    # is rebuilt elsewhere. Case-only variants cannot create another member.
    if len({record["run_id"].casefold() for record in run_records}) != len(run_records):
        raise RegistryError("run_source must contain unique individual runs")
    run_identities = _source_identities("ablation_summary_v1", run_summary, identity_names)
    combined = {}
    for key in identity_names:
        left, right = identities[key], run_identities[key]
        if left is not None and right is not None and left != right:
            raise RegistryError(f"run_source identity disagreement: {key}")
        combined[key] = left if left is not None and right is not None else None
    for record in run_records:
        record["source"] = _repo_path(run_path, repo_root)
    return run_records, combined


def render_browser(
    payload: dict[str, Any], *, template_path: Path | str, output_path: Path | str
) -> Path:
    """Render a self-contained browser without mutating any evidence source."""

    template_path = Path(template_path)
    output_path = Path(output_path)
    template = template_path.read_text(encoding="utf-8")
    marker = "__EXPERIMENT_PAYLOAD__"
    if template.count(marker) != 1:
        raise RegistryError(f"template must contain exactly one {marker} marker")
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    serialized = serialized.replace("</", "<\\/")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(template.replace(marker, serialized), encoding="utf-8")
    return output_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--template", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    registry = args.registry or repo_root / "docs/research/experiment_registry.json"
    template = (
        args.template
        or Path(__file__).with_name("templates") / "experiment_browser.html"
    )
    output = args.output or repo_root / "docs/research/itrader_experiment_browser.html"
    payload = build_browser_payload(registry, repo_root=repo_root)
    render_browser(payload, template_path=template, output_path=output)
    print(f"Rendered {len(payload['experiments'])} experiments to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

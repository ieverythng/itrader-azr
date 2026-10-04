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
    fractions = checkpoint.get("fresh_action_fractions")
    if (
        not isinstance(fractions, list)
        or len(fractions) != 3
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
            for value in fractions
        )
        or not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-6)
    ):
        raise RegistryError(
            f"G0 checkpoint {steps} must contain three finite action fractions "
            "in [0, 1] that sum to one"
        )
    return [float(value) for value in fractions]


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
    del source_path, repo_root
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
        records.append(
            {
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
        )
    return records


ADAPTERS: dict[
    str,
    Callable[[str, dict[str, Any], Path, Path], list[dict[str, Any]]],
] = {
    "g0_closure_v1": _adapt_g0_closure,
    "ablation_summary_v1": _adapt_ablation_summary,
    "terminal_gate_v1": _adapt_terminal_gate,
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
        records = _validate_records(
            experiment_id,
            adapter(experiment_id, source, source_path, repo_root),
            metric_units=metric_units,
            attribute_units=attribute_units,
        )
        resolved.append(
            {
                "id": experiment_id,
                "title": str(entry.get("title", experiment_id)),
                "date": str(entry.get("date", "")),
                "status": str(entry.get("status", "evidence")),
                "summary": str(entry.get("summary", "")),
                "source": _repo_path(source_path, repo_root),
                "report": str(entry.get("report", "")),
                "legacy_dashboard": entry.get("legacy_dashboard"),
                "factors": normalized_factors,
                "identities": identities,
                "comparability_id": _comparability_id(normalized_factors, identities),
                "records": records,
            }
        )

    return {
        "schema_version": registry["schema_version"],
        "title": str(registry.get("title", "iTrader Experiment Browser")),
        "generated_from": _repo_path(registry_path, repo_root),
        "metric_units": metric_units,
        "attribute_units": attribute_units,
        "experiments": resolved,
        "comparisons": _build_comparisons(
            resolved,
            factor_names=factor_names,
            identity_names=identity_names,
        ),
    }


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

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from itrader.scripts.experiments.render_experiment_browser import (
    RegistryError,
    build_browser_payload,
    render_browser,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _g0_summary() -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment": "R0 + R4-v1 G0 checkpoint closure diagnostic",
        "member": "00002",
        "training_recipe": {
            "runtime": "Ray 2.58",
            "cost_curriculum": "0:0,48000:0.25,96000:0.5,144000:1",
        },
        "source_panel": {"name": "jan-panel", "sha256": "a" * 64},
        "fresh_panel": {"name": "nov-panel", "sha256": "b" * 64},
        "checkpoints": [
            {
                "pbt_progress_steps": 102040,
                "optimization_cost_bps": 0.5,
                "source_net_return": 0.002,
                "source_sharpe_per_sqrt_bar": 0.01,
                "fresh_net_return": -0.001,
                "fresh_sharpe_per_sqrt_bar": -0.004,
                "fresh_turnover": 2.0,
                "fresh_action_fractions": [0.75, 0.0, 0.25],
                "top_two_probability_margin_p01": 0.02,
                "explained_variance": 0.03,
                "approx_kl": 0.001,
                "clip_fraction": None,
                "two_process_identity": True,
            }
        ],
        "decision": {
            "two_process_determinism_passed": True,
            "fresh_panel_economic_gate_passed": False,
            "r0_plus_r4_v1_architecture_qualified": True,
            "r0_plus_r4_v1_economically_promoted": False,
            "repeat_same_panel_pbt": False,
            "next_boundary": "matched R0 versus R1 versus R1 + R4-v1",
        },
    }


def _write_g0_source(path: Path, summary: dict[str, object] | None = None) -> None:
    summary = summary or _g0_summary()
    for checkpoint in summary["checkpoints"]:
        steps = checkpoint["pbt_progress_steps"]
        episode_content = f'{{"steps":{steps},"episode":0}}\n'
        state_content = f'{{"steps":{steps},"state":0}}\n'
        checkpoint["two_process_episode_sha256"] = hashlib.sha256(
            episode_content.encode("utf-8")
        ).hexdigest()
        checkpoint["two_process_state_steps_sha256"] = hashlib.sha256(
            state_content.encode("utf-8")
        ).hexdigest()
        receipt = {
            "trial": "training_fn_ITraderEnv_dead0_00002_2_seed=7_2026-10-03",
            "checkpoint": f"D:\\ray\\checkpoints\\iter_{steps}",
            "source_panel_sha256": summary["source_panel"]["sha256"],
            "evaluation_panel_sha256": summary["fresh_panel"]["sha256"],
            "summary": {
                "evaluation/net_return_mean": checkpoint["fresh_net_return"],
                "evaluation/net_sharpe_per_sqrt_bar_mean": checkpoint[
                    "fresh_sharpe_per_sqrt_bar"
                ],
                "evaluation/turnover_mean": checkpoint["fresh_turnover"],
                "custom_metrics/act_0_frac_mean": checkpoint["fresh_action_fractions"][
                    0
                ],
                "custom_metrics/act_1_frac_mean": checkpoint["fresh_action_fractions"][
                    1
                ],
                "custom_metrics/act_2_frac_mean": checkpoint["fresh_action_fractions"][
                    2
                ],
            },
        }
        for process in ("process_a", "process_b"):
            replay_dir = path.parent / f"checkpoint_{steps}/{process}/replay"
            state_dir = path.parent / f"checkpoint_{steps}/{process}/state"
            _write_json(replay_dir / "member-00002.json", receipt)
            replay_dir.mkdir(parents=True, exist_ok=True)
            (replay_dir / "member-00002.episodes.jsonl").write_bytes(
                episode_content.encode("utf-8")
            )
            state_dir.mkdir(parents=True, exist_ok=True)
            (state_dir / "member-00002.state_steps.jsonl").write_bytes(
                state_content.encode("utf-8")
            )
    _write_json(path, summary)


def _registry(source: str = "evidence/g0.json") -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiments": [
            {
                "id": "r4-v1-g0",
                "title": "G0 fresh-panel closure",
                "date": "2026-10-03",
                "adapter": "g0_closure_v1",
                "source": source,
                "report": "docs/g0.md",
                "factors": {
                    "runtime": "ray-2.58",
                    "dataset": "dataset-v2-rth",
                    "panel": "nov-panel",
                    "cost": "causal-staging",
                    "representation": "r0+r4-v1",
                    "curriculum": "family-00-random-generated",
                    "pbt": "historical-sync-pbt-replay",
                    "seed_scope": "member-00002",
                },
            }
        ],
    }


def test_browser_payload_indexes_g0_checkpoint_and_causal_factor(tmp_path: Path):
    _write_g0_source(tmp_path / "evidence/g0.json")
    _write_json(tmp_path / "registry.json", _registry())

    payload = build_browser_payload(tmp_path / "registry.json", repo_root=tmp_path)

    experiment = payload["experiments"][0]
    checkpoint = experiment["records"][0]
    assert experiment["factors"]["cost"] == "causal-staging"
    assert checkpoint["record_key"] == "r4-v1-g0:00002:102040"
    assert checkpoint["member"] == "00002"
    assert checkpoint["pbt_progress_steps"] == 102040
    assert checkpoint["metrics"]["fresh_net_return"] == pytest.approx(-0.001)
    assert checkpoint["evidence_path"].endswith("checkpoint_102040")
    assert checkpoint["attributes"]["artifact_pair_parity_passed"] is True
    assert (
        checkpoint["attributes"]["independent_process_provenance"]
        == "not-recorded-in-receipts"
    )
    assert checkpoint["attributes"]["fresh_panel_economic_gate_passed"] is False
    assert checkpoint["run_id"].endswith("seed=7_2026-10-03")


def test_registry_rejects_duplicate_experiment_ids(tmp_path: Path):
    _write_g0_source(tmp_path / "evidence/g0.json")
    registry = _registry()
    registry["experiments"].append(dict(registry["experiments"][0]))
    _write_json(tmp_path / "registry.json", registry)

    with pytest.raises(RegistryError, match="duplicate experiment id"):
        build_browser_payload(tmp_path / "registry.json", repo_root=tmp_path)


def test_registry_rejects_a_missing_authoritative_source(tmp_path: Path):
    _write_json(tmp_path / "registry.json", _registry("missing.json"))

    with pytest.raises(RegistryError, match="source does not exist"):
        build_browser_payload(tmp_path / "registry.json", repo_root=tmp_path)


def test_builder_owns_comparability_instead_of_the_template(tmp_path: Path):
    _write_g0_source(tmp_path / "evidence/g0.json")
    registry = _registry()
    second = dict(registry["experiments"][0])
    second["id"] = "r4-v1-g0-exact"
    second["title"] = "G0 exact-cost comparator"
    second["factors"] = dict(second["factors"])
    second["factors"]["cost"] = "exact-1bp"
    registry["experiments"].append(second)
    _write_json(tmp_path / "registry.json", registry)

    payload = build_browser_payload(tmp_path / "registry.json", repo_root=tmp_path)

    assert payload["comparisons"] == [
        {
            "left": "r4-v1-g0",
            "right": "r4-v1-g0-exact",
            "directly_comparable": False,
            "different_factors": ["cost"],
            "different_identities": [],
            "missing_identities": ["dataset_sha256", "metric_contract"],
        }
    ]


def test_comparability_rejects_matching_labels_with_different_panel_hashes(
    tmp_path: Path,
):
    left_summary = _g0_summary()
    right_summary = _g0_summary()
    right_summary["fresh_panel"] = {"name": "nov-panel", "sha256": "c" * 64}
    _write_g0_source(tmp_path / "evidence/left/g0.json", left_summary)
    _write_g0_source(tmp_path / "evidence/right/g0.json", right_summary)
    registry = _registry("evidence/left/g0.json")
    second = json.loads(json.dumps(registry["experiments"][0]))
    second["id"] = "right"
    second["source"] = "evidence/right/g0.json"
    registry["experiments"].append(second)
    _write_json(tmp_path / "registry.json", registry)

    comparison = build_browser_payload(tmp_path / "registry.json", repo_root=tmp_path)[
        "comparisons"
    ][0]

    assert comparison["directly_comparable"] is False
    assert comparison["different_identities"] == ["evaluation_panel_sha256"]


def test_blank_authoritative_identities_remain_missing(tmp_path: Path):
    source = {
        "primary_metric": "evaluation/net_return_mean",
        "dataset_sha256": "",
        "evaluation_panel_sha256": "   ",
        "metric_contract": "\t",
        "results": [
            {
                "series_id": "masked",
                "trial_dir": "D:\\ray\\masked",
                "final_metric": 0.0,
            }
        ],
    }
    _write_json(tmp_path / "left.json", source)
    _write_json(tmp_path / "right.json", source)
    registry = _registry("left.json")
    registry["experiments"][0]["adapter"] = "ablation_summary_v1"
    second = json.loads(json.dumps(registry["experiments"][0]))
    second["id"] = "right"
    second["source"] = "right.json"
    registry["experiments"].append(second)
    _write_json(tmp_path / "registry.json", registry)

    payload = build_browser_payload(tmp_path / "registry.json", repo_root=tmp_path)

    assert payload["comparisons"][0]["directly_comparable"] is False
    assert payload["comparisons"][0]["missing_identities"] == [
        "dataset_sha256",
        "evaluation_panel_sha256",
        "metric_contract",
    ]


def test_malformed_sha256_identity_is_rejected(tmp_path: Path):
    source = {
        "primary_metric": "evaluation/net_return_mean",
        "dataset_sha256": "not-a-sha256",
        "results": [
            {
                "series_id": "masked",
                "trial_dir": "D:\\ray\\masked",
                "final_metric": 0.0,
            }
        ],
    }
    _write_json(tmp_path / "source.json", source)
    registry = _registry("source.json")
    registry["experiments"][0]["adapter"] = "ablation_summary_v1"
    _write_json(tmp_path / "registry.json", registry)

    with pytest.raises(RegistryError, match="lowercase SHA-256 dataset_sha256"):
        build_browser_payload(tmp_path / "registry.json", repo_root=tmp_path)


@pytest.mark.parametrize(
    "patch",
    (
        {"schema_version": 999},
        {"experiments.0.date": "2026-02-29"},
        {"experiments": []},
    ),
)
def test_registry_schema_is_enforced(tmp_path: Path, patch: dict[str, object]):
    _write_g0_source(tmp_path / "evidence/g0.json")
    registry = _registry()
    if "experiments.0.date" in patch:
        registry["experiments"][0]["date"] = patch["experiments.0.date"]
    else:
        registry.update(patch)
    _write_json(tmp_path / "registry.json", registry)

    with pytest.raises(RegistryError, match="registry schema violation"):
        build_browser_payload(tmp_path / "registry.json", repo_root=tmp_path)


def test_terminal_gate_keeps_threshold_distinct_from_observed_steps(tmp_path: Path):
    source = {
        "schema_version": 1,
        "review_steps": 192000,
        "arms": {
            "causal": {
                "experiment_name": "causal-run",
                "decision": "pass",
                "terminal_action": "short_only",
                "terminal_steps": [194368, 195560],
                "terminal_net_return": 0.004,
                "terminal_sharpe_per_sqrt_bar": 0.01,
                "exploit_count": 2,
                "verified_lineage_receipt_count": 2,
            }
        },
    }
    _write_json(tmp_path / "terminal.json", source)
    registry = _registry("terminal.json")
    registry["experiments"][0]["adapter"] = "terminal_gate_v1"
    _write_json(tmp_path / "registry.json", registry)

    record = build_browser_payload(tmp_path / "registry.json", repo_root=tmp_path)[
        "experiments"
    ][0]["records"][0]

    assert record["pbt_progress_steps"] is None
    assert record["attributes"]["review_threshold_steps"] == 192000
    assert record["attributes"]["terminal_steps"] == [194368, 195560]
    assert record["attributes"]["terminal_action"] == "short_only"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("review_steps", 192000.5, "positive integer review_steps"),
        ("exploit_count", 1.5, "non-negative integer exploit_count"),
        (
            "verified_lineage_receipt_count",
            2.25,
            "non-negative integer verified_lineage_receipt_count",
        ),
        ("terminal_steps", [-1], "positive integer terminal_steps"),
    ),
)
def test_terminal_gate_rejects_invalid_count_semantics(
    tmp_path: Path, field: str, value: object, message: str
):
    source = {
        "schema_version": 1,
        "review_steps": 192000,
        "arms": {
            "causal": {
                "experiment_name": "causal-run",
                "decision": "pass",
                "terminal_action": "short_only",
                "terminal_steps": [194368],
                "terminal_net_return": 0.004,
                "terminal_sharpe_per_sqrt_bar": 0.01,
                "exploit_count": 2,
                "verified_lineage_receipt_count": 2,
            }
        },
    }
    if field == "review_steps":
        source[field] = value
    else:
        source["arms"]["causal"][field] = value
    _write_json(tmp_path / "terminal.json", source)
    registry = _registry("terminal.json")
    registry["experiments"][0]["adapter"] = "terminal_gate_v1"
    _write_json(tmp_path / "registry.json", registry)

    with pytest.raises(RegistryError, match=message):
        build_browser_payload(tmp_path / "registry.json", repo_root=tmp_path)


def test_ablation_run_and_member_are_indexed_from_trial_directory(tmp_path: Path):
    source = {
        "primary_metric": "evaluation/net_return_mean",
        "results": [
            {
                "variant_id": "masked",
                "series_id": "masked",
                "trial_dir": (
                    "D:\\ray\\training_fn_ITraderEnv_dead0_00003_3_"
                    "2026-09-25_14-12-15"
                ),
                "final_metric": 0.25,
            }
        ],
    }
    _write_json(tmp_path / "ablation.json", source)
    registry = _registry("ablation.json")
    registry["experiments"][0]["adapter"] = "ablation_summary_v1"
    _write_json(tmp_path / "registry.json", registry)

    record = build_browser_payload(tmp_path / "registry.json", repo_root=tmp_path)[
        "experiments"
    ][0]["records"][0]

    assert record["member"] == "00003"
    assert record["run_id"].startswith("training_fn_ITraderEnv_dead0_00003")
    assert record["metrics"] == {"evaluation/net_return_mean": 0.25}


@pytest.mark.parametrize(
    ("adapter", "source", "message"),
    (
        ("g0_closure_v1", {}, "non-empty checkpoints"),
        ("ablation_summary_v1", {}, "primary_metric"),
        ("terminal_gate_v1", {}, "non-empty arms"),
    ),
)
def test_adapters_reject_missing_authoritative_result_shapes(
    tmp_path: Path,
    adapter: str,
    source: dict[str, object],
    message: str,
):
    _write_json(tmp_path / "source.json", source)
    registry = _registry("source.json")
    registry["experiments"][0]["adapter"] = adapter
    _write_json(tmp_path / "registry.json", registry)

    with pytest.raises(RegistryError, match=message):
        build_browser_payload(tmp_path / "registry.json", repo_root=tmp_path)


def test_duplicate_checkpoint_record_identity_is_rejected(tmp_path: Path):
    summary = _g0_summary()
    summary["checkpoints"].append(dict(summary["checkpoints"][0]))
    _write_g0_source(tmp_path / "evidence/g0.json", summary)
    _write_json(tmp_path / "registry.json", _registry())

    with pytest.raises(RegistryError, match="duplicate record key"):
        build_browser_payload(tmp_path / "registry.json", repo_root=tmp_path)


def test_g0_cannot_opt_out_of_two_process_provenance(tmp_path: Path):
    summary = _g0_summary()
    summary["checkpoints"][0]["two_process_identity"] = False
    _write_json(tmp_path / "evidence/g0.json", summary)
    _write_json(tmp_path / "registry.json", _registry())

    with pytest.raises(RegistryError, match="requires two-process identity"):
        build_browser_payload(tmp_path / "registry.json", repo_root=tmp_path)


@pytest.mark.parametrize(
    ("adapter", "source", "message"),
    (
        (
            "ablation_summary_v1",
            {
                "primary_metric": "evaluation/net_return_mean",
                "results": [{"series_id": "missing-metric"}],
            },
            "final_metric",
        ),
        (
            "terminal_gate_v1",
            {"review_steps": 192000, "arms": {"arm-a": {}}},
            "terminal_net_return",
        ),
    ),
)
def test_adapters_reject_incomplete_authoritative_records(
    tmp_path: Path,
    adapter: str,
    source: dict[str, object],
    message: str,
):
    _write_json(tmp_path / "source.json", source)
    registry = _registry("source.json")
    registry["experiments"][0]["adapter"] = adapter
    _write_json(tmp_path / "registry.json", registry)

    with pytest.raises(RegistryError, match=message):
        build_browser_payload(tmp_path / "registry.json", repo_root=tmp_path)


def test_duplicate_json_object_keys_are_rejected(tmp_path: Path):
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        '{"schema_version":1,"experiments":[{"id":"first","id":"second"}]}',
        encoding="utf-8",
    )

    with pytest.raises(RegistryError, match="duplicate JSON key: id"):
        build_browser_payload(registry_path, repo_root=tmp_path)


def test_registry_rejects_undeclared_comparison_factors(tmp_path: Path):
    _write_g0_source(tmp_path / "evidence/g0.json")
    registry = _registry()
    registry["experiments"][0]["factors"]["optimizer"] = "adam"
    _write_json(tmp_path / "registry.json", registry)

    with pytest.raises(RegistryError, match="registry schema violation"):
        build_browser_payload(tmp_path / "registry.json", repo_root=tmp_path)


@pytest.mark.parametrize("value", ("", "   ", "\t"))
def test_registry_rejects_blank_comparison_factors(tmp_path: Path, value: str):
    _write_g0_source(tmp_path / "evidence/g0.json")
    registry = _registry()
    registry["experiments"][0]["factors"]["runtime"] = value
    _write_json(tmp_path / "registry.json", registry)

    with pytest.raises(RegistryError, match="registry schema violation"):
        build_browser_payload(tmp_path / "registry.json", repo_root=tmp_path)


@pytest.mark.parametrize("field", ("id", "title"))
@pytest.mark.parametrize("value", ("", "   ", "\t"))
def test_registry_rejects_blank_experiment_metadata(
    tmp_path: Path, field: str, value: str
):
    _write_g0_source(tmp_path / "evidence/g0.json")
    registry = _registry()
    registry["experiments"][0][field] = value
    _write_json(tmp_path / "registry.json", registry)

    with pytest.raises(RegistryError, match="registry schema violation"):
        build_browser_payload(tmp_path / "registry.json", repo_root=tmp_path)


@pytest.mark.parametrize(
    "fractions",
    (
        [-1.0, 2.0, 0.0],
        [None, True, 0.0],
        [0.25, 0.25, 0.25],
    ),
)
def test_g0_rejects_invalid_action_fractions(tmp_path: Path, fractions: list[object]):
    summary = _g0_summary()
    summary["checkpoints"][0]["fresh_action_fractions"] = fractions
    _write_g0_source(tmp_path / "evidence/g0.json", summary)
    _write_json(tmp_path / "registry.json", _registry())

    with pytest.raises(RegistryError, match="action fractions"):
        build_browser_payload(tmp_path / "registry.json", repo_root=tmp_path)


def test_g0_rejects_incomplete_checkpoint_without_raw_key_error(tmp_path: Path):
    summary = _g0_summary()
    summary["checkpoints"] = [{}]
    _write_json(tmp_path / "evidence/g0.json", summary)
    _write_json(tmp_path / "registry.json", _registry())

    with pytest.raises(RegistryError, match="positive integer pbt_progress_steps"):
        build_browser_payload(tmp_path / "registry.json", repo_root=tmp_path)


def test_g0_rejects_mismatched_artifact_pair_hash(tmp_path: Path):
    _write_g0_source(tmp_path / "evidence/g0.json")
    state_path = next(
        (tmp_path / "evidence/checkpoint_102040/process_b/state").glob(
            "*.state_steps.jsonl"
        )
    )
    state_path.write_text('{"tampered":true}\n', encoding="utf-8")
    _write_json(tmp_path / "registry.json", _registry())

    with pytest.raises(RegistryError, match="state artifact parity mismatch"):
        build_browser_payload(tmp_path / "registry.json", repo_root=tmp_path)


def test_non_finite_json_numbers_are_rejected(tmp_path: Path):
    summary = _g0_summary()
    summary["checkpoints"][0]["fresh_net_return"] = float("nan")
    _write_g0_source(tmp_path / "evidence/g0.json", summary)
    _write_json(tmp_path / "registry.json", _registry())

    with pytest.raises(RegistryError, match="non-finite JSON number NaN"):
        build_browser_payload(tmp_path / "registry.json", repo_root=tmp_path)


def test_json_exponent_overflow_is_rejected_globally(tmp_path: Path):
    source_path = tmp_path / "evidence/g0.json"
    _write_g0_source(source_path)
    source_path.write_text(
        source_path.read_text(encoding="utf-8").replace(
            '"clip_fraction": null', '"clip_fraction": 1e999'
        ),
        encoding="utf-8",
    )
    _write_json(tmp_path / "registry.json", _registry())

    with pytest.raises(RegistryError, match="non-finite JSON number"):
        build_browser_payload(tmp_path / "registry.json", repo_root=tmp_path)


def test_g0_rejects_unknown_decision_fields(tmp_path: Path):
    summary = _g0_summary()
    summary["decision"]["silent_override"] = True
    _write_g0_source(tmp_path / "evidence/g0.json", summary)
    _write_json(tmp_path / "registry.json", _registry())

    with pytest.raises(RegistryError, match="unknown fields.*silent_override"):
        build_browser_payload(tmp_path / "registry.json", repo_root=tmp_path)


def test_ablation_rejects_metric_without_explicit_unit_contract(tmp_path: Path):
    _write_json(
        tmp_path / "ablation.json",
        {
            "primary_metric": "win_rate",
            "results": [
                {
                    "series_id": "unknown-unit",
                    "trial_dir": "D:\\ray\\unknown-unit",
                    "final_metric": 0.25,
                }
            ],
        },
    )
    registry = _registry("ablation.json")
    registry["experiments"][0]["adapter"] = "ablation_summary_v1"
    _write_json(tmp_path / "registry.json", registry)

    with pytest.raises(RegistryError, match="unknown metric unit contract"):
        build_browser_payload(tmp_path / "registry.json", repo_root=tmp_path)


def test_rendered_browser_contains_search_tabs_and_embedded_payload(tmp_path: Path):
    _write_g0_source(tmp_path / "evidence/g0.json")
    _write_json(tmp_path / "registry.json", _registry())
    template = tmp_path / "template.html"
    template.write_text(
        '<main data-role="experiment-tabs"></main>'
        '<script id="experiment-payload" type="application/json">'
        "__EXPERIMENT_PAYLOAD__</script>",
        encoding="utf-8",
    )
    output = tmp_path / "browser.html"

    render_browser(
        build_browser_payload(tmp_path / "registry.json", repo_root=tmp_path),
        template_path=template,
        output_path=output,
    )

    html = output.read_text(encoding="utf-8")
    assert 'data-role="experiment-tabs"' in html
    assert 'id="experiment-payload"' in html
    assert "G0 fresh-panel closure" in html
    assert "__EXPERIMENT_PAYLOAD__" not in html


def test_committed_browser_matches_registry_and_authoritative_sources(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    registry = repo_root / "docs/research/experiment_registry.json"
    template = (
        repo_root / "itrader/scripts/experiments/templates/experiment_browser.html"
    )
    expected = repo_root / "docs/research/itrader_experiment_browser.html"
    rendered = tmp_path / "itrader_experiment_browser.html"

    payload = build_browser_payload(registry, repo_root=repo_root)
    render_browser(payload, template_path=template, output_path=rendered)

    assert [experiment["id"] for experiment in payload["experiments"]] == [
        "r4-v1-min-smoke",
        "r4-v1-blocking-4x3",
        "r4-v1-actor-robustness",
        "r4-v1-g0-fresh-panel",
    ]
    assert rendered.read_text(encoding="utf-8") == expected.read_text(encoding="utf-8")

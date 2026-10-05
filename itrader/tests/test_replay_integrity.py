import json
import sys
from types import SimpleNamespace

import pytest

from itrader.rl_lib.api_stack import LEGACY_API_STACK_ID, NATIVE_API_STACK_ID
from itrader.rl_lib.evaluation.provenance import checkpoint_sha256
from itrader.rl_lib.train.pbt_scheduler import SeededPopulationBasedTraining


def test_equal_selection_scores_do_not_attest_different_replay_payloads():
    from itrader.rl_lib.evaluation.replay_evidence import replay_parity

    original = {"metric": "evaluation/economic_search_score", "score": 0.0}
    replay = [{"net_return": -0.025, "action_reward_trace_sha256": "b" * 64}]
    result = replay_parity(original, observed_score=0.0, episodes=replay)
    assert result["selection_score_parity"] is True
    assert result["checkpoint_parity"] is None
    original["evaluation_payload_sha256"] = "a" * 64
    result = replay_parity(original, observed_score=0.0, episodes=replay)
    assert result["checkpoint_parity"] is False


def test_process_receipts_are_non_overwriting_and_bind_artifact_bytes(tmp_path):
    from itrader.rl_lib.evaluation.replay_evidence import ReplayInvocation, verify_receipt

    receipt_path = tmp_path / "trial.json"
    invocation = ReplayInvocation(receipt_path, operation="episode_replay")
    payload = tmp_path / "trial.episodes.jsonl"
    payload.write_text('{}\n')
    invocation.complete({"checkpoint_parity": None}, contract={"data": "a" * 64}, artifacts=[payload])
    receipt = json.loads(receipt_path.read_text())
    assert verify_receipt(receipt_path)["status"] == "complete"
    assert receipt["producer"]["process_instance_id"]
    with pytest.raises(FileExistsError):
        ReplayInvocation(receipt_path, operation="episode_replay")
    payload.write_text('{"changed":true}\n')
    with pytest.raises(ValueError, match="artifact"):
        verify_receipt(receipt_path)


def test_independent_receipt_gate_executes_two_processes_and_rejects_copy(tmp_path):
    from itrader.rl_lib.evaluation.replay_evidence import compare_process_replays
    from itrader.rl_lib.evaluation.run_replay_replication import launch_replay

    code = """
import sys
from pathlib import Path
from itrader.rl_lib.evaluation.replay_evidence import ReplayInvocation
root = Path(sys.argv[1]); root.mkdir()
from itrader.rl_lib.evaluation.provenance import runtime_identity
from itrader.rl_lib.evaluation.replay_evidence import episode_payload_sha256
import json
invocation = ReplayInvocation(root / 'receipt.json', operation='episode_replay')
rows = [{'net_return':0.01, 'action_reward_trace_sha256':'b'*64}]
artifact = root / 'trial.episodes.jsonl'; artifact.write_text(json.dumps(rows[0])+'\\n')
contract = {key:'a'*64 for key in ('checkpoint_sha256','dataset_sha256','source_panel_sha256',
    'evaluation_panel_sha256','effective_env_config_sha256','checkpoint_config_sha256')}
contract.update(version='deterministic_replay_v1', progress_steps=12, restore_generation=0,
    observation_schema={'schema_id':'test'}, runtime=runtime_identity(),
    replay_source={'source_tree_sha256':'c'*64}, explore=False, evaluation_seed=37,
    execution={'device':'cpu','torch_deterministic_algorithms':False,'torch_num_threads':1})
invocation.complete({'episode_count':1,'evaluation_payload_sha256':episode_payload_sha256(rows)},
    contract=contract, artifacts=[artifact])
"""
    launches, paths = [], []
    for index in range(2):
        root = tmp_path / str(index)
        launches.append(launch_replay([sys.executable, "-c", code, str(root)],
                                      log_path=tmp_path / f"{index}.log", timeout_seconds=30))
        paths.append(root / "receipt.json")
    assert compare_process_replays(paths, launches)["verified_process_replication"] is True
    assert compare_process_replays([paths[0], paths[0]], [launches[0], launches[0]])["verified_process_replication"] is False
    bad_launches = [dict(row) for row in launches]
    bad_launches[1]["observed_processes"] = []
    with pytest.raises(ValueError, match="launch"):
        compare_process_replays(paths, bad_launches)
    from itrader.rl_lib.evaluation.provenance import canonical_json_sha256
    receipt = json.loads(paths[0].read_text())
    receipt["evaluation_contract"] = {}
    receipt["evaluation_contract_sha256"] = canonical_json_sha256({})
    paths[0].write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="contract"):
        compare_process_replays(paths, launches)


def test_pbt_recovery_rejects_changed_indexed_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    module = checkpoint / "learner_group/learner/rl_module/default_policy"
    module.mkdir(parents=True)
    (checkpoint / "algorithm_state.pkl").write_bytes(b"algorithm")
    (checkpoint / "rllib_checkpoint.json").write_text("{}")
    weights = module / "module_state.pkl"
    weights.write_bytes(b"scored weights")
    digest = checkpoint_sha256(checkpoint, api_stack_id=NATIVE_API_STACK_ID)
    stable = tmp_path / "stable/trial"
    stable.mkdir(parents=True)
    (stable / "checkpoint_index.jsonl").write_text(json.dumps({
        "pbt_progress_steps": 100, "stable_checkpoint": str(checkpoint),
        "api_stack_id": NATIVE_API_STACK_ID,
        "checkpoint_sha256": digest,
    }) + "\n")
    weights.write_bytes(b"different weights")
    trial = SimpleNamespace(config={"stable_metrics_root": str(stable.parent),
                                   "provenance_enforced": True,
                                   "api_stack_id": NATIVE_API_STACK_ID},
                            local_path=str(tmp_path / "trial"), trial_id="trial")
    state = SimpleNamespace(last_checkpoint=None, last_train_time=100)
    scheduler = SeededPopulationBasedTraining(
        random_seed=1, metric="score", mode="max", time_attr="steps",
        perturbation_interval=100, hyperparam_mutations={"lr": [0.001, 0.002]},
    )
    with pytest.raises(ValueError, match="digest"):
        scheduler._recover_stable_checkpoint(trial, state, exact_progress=True)
    assert state.last_checkpoint is None
    weights.write_bytes(b"scored weights")
    from itrader.rl_lib.evaluation.run_checkpoint_replay import load_checkpoint_selection

    mismatched_stack_state = SimpleNamespace(
        last_checkpoint=None, last_train_time=100
    )
    trial.config["api_stack_id"] = LEGACY_API_STACK_ID
    with pytest.raises(ValueError, match="API stack"):
        scheduler._recover_stable_checkpoint(
            trial, mismatched_stack_state, exact_progress=True
        )
    assert mismatched_stack_state.last_checkpoint is None
    trial.config["api_stack_id"] = NATIVE_API_STACK_ID
    valid_state = SimpleNamespace(last_checkpoint=None, last_train_time=100)
    assert scheduler._recover_stable_checkpoint(
        trial, valid_state, exact_progress=True
    )
    assert valid_state.last_checkpoint is not None

    for malformed in (100.9, True, "100", -1, None):
        (stable / "checkpoint_index.jsonl").write_text(json.dumps({
            "pbt_progress_steps": malformed, "stable_checkpoint": str(checkpoint),
            "api_stack_id": NATIVE_API_STACK_ID,
            "checkpoint_sha256": digest, "selection_metric": "score", "selection_score": 1,
        }) + "\n")
        with pytest.raises(ValueError, match="integer"):
            scheduler._recover_stable_checkpoint(trial, state, exact_progress=True)
        with pytest.raises(ValueError, match="integer"):
            load_checkpoint_selection(stable, "best", progress_steps=100)
        assert state.last_checkpoint is None
    (stable / "checkpoint_index.jsonl").write_text(json.dumps({
        "pbt_progress_steps": 100, "stable_checkpoint": str(checkpoint),
        "api_stack_id": NATIVE_API_STACK_ID,
        "checkpoint_sha256": digest,
    }) + "\n")
    for malformed in (100.9, True, "100", -1, None):
        state.last_train_time = malformed
        with pytest.raises(ValueError, match="integer"):
            scheduler._recover_stable_checkpoint(trial, state, exact_progress=True)
        assert state.last_checkpoint is None


def test_native_checkpoint_digest_requires_native_module_authority(tmp_path):
    checkpoint = tmp_path / "native-checkpoint"
    module = checkpoint / "learner_group/learner/rl_module/default_policy"
    module.mkdir(parents=True)
    (checkpoint / "algorithm_state.pkl").write_bytes(b"algorithm")
    (checkpoint / "rllib_checkpoint.json").write_text("{}", encoding="utf-8")
    (module / "module_state.pkl").write_bytes(b"native weights")

    digest = checkpoint_sha256(checkpoint, api_stack_id=NATIVE_API_STACK_ID)
    assert len(digest) == 64

    (module / "module_state.pkl").write_bytes(b"")
    legacy_policy = checkpoint / "policies/default_policy"
    legacy_policy.mkdir(parents=True)
    (legacy_policy / "policy_state.pkl").write_bytes(b"legacy weights must not qualify")
    with pytest.raises(ValueError, match="module_state.pkl"):
        checkpoint_sha256(checkpoint, api_stack_id=NATIVE_API_STACK_ID)

    with pytest.raises(ValueError, match="Unknown checkpoint API stack"):
        checkpoint_sha256(checkpoint, api_stack_id="unqualified_stack")


def test_native_replay_selection_binds_checkpoint_to_resolved_stack(tmp_path, monkeypatch):
    from itrader.rl_lib.evaluation import run_checkpoint_replay
    from itrader.rl_lib.evaluation.provenance import canonical_json_sha256

    checkpoint = tmp_path / "native-checkpoint"
    module = checkpoint / "learner_group/learner/rl_module/default_policy"
    module.mkdir(parents=True)
    (checkpoint / "algorithm_state.pkl").write_bytes(b"algorithm")
    (checkpoint / "rllib_checkpoint.json").write_text("{}", encoding="utf-8")
    (module / "module_state.pkl").write_bytes(b"native weights")
    digest = checkpoint_sha256(checkpoint, api_stack_id=NATIVE_API_STACK_ID)

    stable = tmp_path / "stable"
    stable.mkdir()
    resolved_path = stable / "resolved.json"
    config = {"api_stack_id": NATIVE_API_STACK_ID, "test_config": True}
    resolved = {
        "api_stack_id": NATIVE_API_STACK_ID,
        "config": config,
        "dataset_sha256": "dataset-digest",
        "evaluation_panel_sha256": "panel-digest",
        "observation_schema": {"schema_id": "native-test"},
    }
    resolved_path.write_text(json.dumps(resolved), encoding="utf-8")
    selection = {
        "api_stack_id": NATIVE_API_STACK_ID,
        "stable_checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "config_snapshot": str(resolved_path),
        "config_hash": canonical_json_sha256(config),
        "dataset_sha256": "dataset-digest",
        "evaluation_panel_sha256": "panel-digest",
        "observation_schema": {"schema_id": "native-test"},
    }
    (stable / "best_checkpoint.json").write_text(
        json.dumps(selection), encoding="utf-8"
    )
    validation = SimpleNamespace(source_panel_sha256="panel-digest")
    monkeypatch.setattr(
        run_checkpoint_replay,
        "load_validation_context",
        lambda *_args, **_kwargs: validation,
    )

    selected, loaded_checkpoint, _resolved, _validation, loaded_digest = (
        run_checkpoint_replay.load_replay_inputs(stable, "best")
    )
    assert selected["api_stack_id"] == NATIVE_API_STACK_ID
    assert loaded_checkpoint == checkpoint.resolve()
    assert loaded_digest == digest

    selection["api_stack_id"] = LEGACY_API_STACK_ID
    (stable / "best_checkpoint.json").write_text(
        json.dumps(selection), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="api_stack_id"):
        run_checkpoint_replay.load_replay_inputs(stable, "best")


def test_native_replay_rejects_valid_legacy_checkpoint_before_loading_context(
    tmp_path, monkeypatch
):
    from itrader.rl_lib.evaluation import run_checkpoint_replay
    from itrader.rl_lib.evaluation.provenance import canonical_json_sha256

    checkpoint = tmp_path / "legacy-checkpoint"
    policy = checkpoint / "policies/default_policy"
    policy.mkdir(parents=True)
    (checkpoint / "algorithm_state.pkl").write_bytes(b"algorithm")
    (checkpoint / "rllib_checkpoint.json").write_text("{}", encoding="utf-8")
    (policy / "policy_state.pkl").write_bytes(b"legacy policy weights")
    digest = checkpoint_sha256(checkpoint, api_stack_id=LEGACY_API_STACK_ID)

    stable = tmp_path / "stable-legacy"
    stable.mkdir()
    resolved_path = stable / "resolved.json"
    config = {"api_stack_id": LEGACY_API_STACK_ID, "test_config": True}
    resolved = {
        "api_stack_id": LEGACY_API_STACK_ID,
        "config": config,
    }
    resolved_path.write_text(json.dumps(resolved), encoding="utf-8")
    selection = {
        "api_stack_id": LEGACY_API_STACK_ID,
        "stable_checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "config_snapshot": str(resolved_path),
        "config_hash": canonical_json_sha256(config),
    }
    (stable / "best_checkpoint.json").write_text(
        json.dumps(selection), encoding="utf-8"
    )

    def unexpected_validation_load(*_args, **_kwargs):
        raise AssertionError("legacy checkpoint reached native validation loading")

    monkeypatch.setattr(
        run_checkpoint_replay,
        "load_validation_context",
        unexpected_validation_load,
    )
    monkeypatch.setattr(
        run_checkpoint_replay,
        "verify_checkpoint_digest",
        unexpected_validation_load,
    )
    with pytest.raises(ValueError, match="historical compatibility workflow"):
        run_checkpoint_replay.load_replay_inputs(stable, "best")


def test_native_process_replay_rejects_dataset_changed_after_checkpoint(
    tmp_path, monkeypatch
):
    from itrader.rl_lib.evaluation.provenance import (
        canonical_json_sha256,
        checkpoint_sha256,
        dataset_sha256,
    )
    from itrader.tools.replay_ray_checkpoint import _validate_native_checkpoint
    from itrader.rl_lib.tasks.spec import TaskSpec

    dataset = tmp_path / "validation.parquet"
    dataset.write_bytes(b"original dataset")
    tasks = [
        TaskSpec.from_dict(
            {
                "family_id": "00",
                "price_id": "SIM",
                "horizon": 24,
                "reward_id": "net_return",
                "reward_params": {"cost_bps": 1.0, "reward_scale": 1000.0},
                "seed": 10,
                "obs_dim": 332,
                "action_dim": 3,
                "start_idx": 100,
                "bar_resolution": "1m",
            }
        ).to_dict()
    ]
    config = {
        "api_stack_id": NATIVE_API_STACK_ID,
        "evaluation_config": {
            "env_config": {
                "df_path": str(dataset),
                "data_split": "validation",
                "window_size": 60,
                "task_sampler": {"fixed_tasks": tasks},
            }
        },
    }
    resolved = {
        "api_stack_id": NATIVE_API_STACK_ID,
        "config": config,
        "config_hash": canonical_json_sha256(config),
        "dataset_sha256": dataset_sha256(dataset),
        "evaluation_panel_sha256": canonical_json_sha256(tasks),
        "observation_schema": {"schema_id": "test"},
    }
    snapshot = tmp_path / "resolved.json"
    snapshot.write_text(json.dumps(resolved), encoding="utf-8")

    checkpoint = tmp_path / "checkpoint"
    module = checkpoint / "learner_group/learner/rl_module/default_policy"
    module.mkdir(parents=True)
    (checkpoint / "algorithm_state.pkl").write_bytes(b"algorithm")
    (checkpoint / "rllib_checkpoint.json").write_text("{}", encoding="utf-8")
    (module / "module_state.pkl").write_bytes(b"weights")
    record = {
        "training_iteration": 1,
        "api_stack_id": NATIVE_API_STACK_ID,
        "stable_checkpoint": str(checkpoint),
        "config_hash": resolved["config_hash"],
        "dataset_sha256": resolved["dataset_sha256"],
        "evaluation_panel_sha256": resolved["evaluation_panel_sha256"],
        "observation_schema": resolved["observation_schema"],
        "config_snapshot": str(snapshot),
        "checkpoint_sha256": checkpoint_sha256(
            checkpoint,
            api_stack_id=NATIVE_API_STACK_ID,
        ),
    }

    assert _validate_native_checkpoint(tmp_path, record, checkpoint) == resolved
    stable_trial_dir = tmp_path / "stable-trial"
    stable_trial_dir.mkdir()
    (stable_trial_dir / "checkpoint_index.jsonl").write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )
    report_columns = (
        "evaluation/optimization_reward_mean",
        "evaluation/gross_return_mean",
        "evaluation/net_return_mean",
        "evaluation/net_sharpe_per_sqrt_bar_mean",
    )
    (stable_trial_dir / "report_rows.csv").write_text(
        ",".join(("training_iteration", *report_columns)) + "\n"
        + ",".join(("1", *(["0"] * len(report_columns))))
        + "\n",
        encoding="utf-8",
    )
    dataset.write_bytes(b"changed dataset")

    from itrader.tools import replay_ray_checkpoint

    def ray_must_not_start(**_kwargs):
        raise AssertionError("dataset tampering must reject before Ray startup")

    monkeypatch.setattr(replay_ray_checkpoint.ray, "init", ray_must_not_start)
    with pytest.raises(ValueError, match="Dataset hash differs from checkpoint provenance"):
        replay_ray_checkpoint.replay(stable_trial_dir, None, 1e-12)

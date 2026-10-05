import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from itrader.rl_lib.api_stack import LEGACY_API_STACK_ID
pytest.importorskip("ray")


def test_sync_pbt_resumes_a_paused_tail_after_other_trials_terminate():
    from ray.tune.experiment.trial import Trial

    from itrader.rl_lib.train.pbt_scheduler import SeededPopulationBasedTraining

    scheduler = SeededPopulationBasedTraining(
        random_seed=271828,
        synch=True,
        time_attr="pbt_progress_steps",
        metric="evaluation/net_return_mean",
        mode="max",
        perturbation_interval=24_000,
        hyperparam_mutations={"lr": [0.001, 0.002]},
    )
    class FakeTrial:
        def __init__(self, status):
            self.status = status

    paused = FakeTrial(Trial.PAUSED)
    terminated = FakeTrial(Trial.TERMINATED)
    scheduler._next_perturbation_sync = 360_000
    scheduler._trial_state[paused] = SimpleNamespace(last_train_time=366_192)
    scheduler._trial_state[terminated] = SimpleNamespace(last_train_time=388_416)
    controller = SimpleNamespace(get_trials=lambda: [terminated, paused])

    assert scheduler.choose_trial_to_run(controller) is paused


@pytest.mark.parametrize("peer_status", ["RUNNING", "PENDING"])
def test_sync_pbt_does_not_resume_a_boundary_trial_while_peers_are_below_sync(
    peer_status,
    monkeypatch,
):
    from ray.tune.experiment.trial import Trial
    from ray.tune.schedulers import PopulationBasedTraining

    from itrader.rl_lib.train.pbt_scheduler import SeededPopulationBasedTraining

    scheduler = SeededPopulationBasedTraining(
        random_seed=271828,
        synch=True,
        time_attr="pbt_progress_steps",
        metric="evaluation/net_return_mean",
        mode="max",
        perturbation_interval=24_000,
        hyperparam_mutations={"lr": [0.001, 0.002]},
    )

    class FakeTrial:
        def __init__(self, status):
            self.status = status

    paused_at_boundary = FakeTrial(Trial.PAUSED)
    peer_below_boundary = FakeTrial(getattr(Trial, peer_status))
    scheduler._next_perturbation_sync = 48_000
    scheduler._trial_state[paused_at_boundary] = SimpleNamespace(
        last_train_time=48_768
    )
    scheduler._trial_state[peer_below_boundary] = SimpleNamespace(
        last_train_time=29_232
    )
    controller = SimpleNamespace(
        get_trials=lambda: [paused_at_boundary, peer_below_boundary]
    )
    monkeypatch.setattr(
        PopulationBasedTraining,
        "choose_trial_to_run",
        lambda self, tune_controller: None,
    )

    assert scheduler.choose_trial_to_run(controller) is None


def test_sync_pbt_unstages_an_orphaned_candidate_with_direct_controller():
    from ray.tune.experiment.trial import Trial

    from itrader.rl_lib.train.pbt_scheduler import SeededPopulationBasedTraining

    scheduler = SeededPopulationBasedTraining(
        random_seed=271828,
        synch=True,
        time_attr="pbt_progress_steps",
        metric="evaluation/net_return_mean",
        mode="max",
        perturbation_interval=24_000,
        hyperparam_mutations={"lr": [0.001, 0.002]},
    )

    class FakeTrial:
        status = Trial.PAUSED
        placement_group_factory = "three-cpus"

    candidate = FakeTrial()
    scheduler._next_perturbation_sync = 120_000
    scheduler._trial_state[candidate] = SimpleNamespace(last_train_time=99_248)
    actor_cache = SimpleNamespace(decrease_max_calls=[])
    actor_cache.decrease_max = actor_cache.decrease_max_calls.append
    controller = SimpleNamespace(
        get_trials=lambda: [candidate],
        _staged_trials={candidate},
        _trial_to_actor={},
        _actor_cache=actor_cache,
    )

    assert scheduler.choose_trial_to_run(controller) is candidate
    assert candidate not in controller._staged_trials
    assert actor_cache.decrease_max_calls == ["three-cpus"]


def test_sync_pbt_exploit_invalidates_a_prestaged_paused_target(monkeypatch):
    from ray.tune.experiment.trial import Trial
    from ray.tune.schedulers import PopulationBasedTraining

    from itrader.rl_lib.train.pbt_scheduler import SeededPopulationBasedTraining

    scheduler = SeededPopulationBasedTraining(
        random_seed=271828,
        synch=True,
        time_attr="pbt_progress_steps",
        metric="evaluation/net_return_mean",
        mode="max",
        perturbation_interval=24_000,
        hyperparam_mutations={"lr": [0.001, 0.002]},
    )
    class FakeTrial:
        def __init__(self, status, placement_group_factory=None):
            self.status = status
            self.placement_group_factory = placement_group_factory

    target = FakeTrial(Trial.PAUSED, "three-cpus")
    donor = FakeTrial(Trial.PAUSED)
    actor_cache = SimpleNamespace(decrease_max_calls=[])
    actor_cache.decrease_max = actor_cache.decrease_max_calls.append
    controller = SimpleNamespace(
        _staged_trials={target},
        _trial_to_actor={},
        _actor_cache=actor_cache,
    )
    monkeypatch.setattr(
        PopulationBasedTraining,
        "_exploit",
        lambda self, tune_controller, trial, trial_to_clone: None,
    )

    scheduler._exploit(controller, target, donor)

    assert target not in controller._staged_trials
    assert actor_cache.decrease_max_calls == ["three-cpus"]
    assert target in scheduler._awaiting_post_restore_result


def test_sync_pbt_exploit_fails_closed_for_an_already_created_paused_actor(
    monkeypatch,
):
    from ray.tune.experiment.trial import Trial
    from ray.tune.schedulers import PopulationBasedTraining

    from itrader.rl_lib.train.pbt_scheduler import SeededPopulationBasedTraining

    scheduler = SeededPopulationBasedTraining(
        random_seed=271828,
        synch=True,
        time_attr="pbt_progress_steps",
        metric="evaluation/net_return_mean",
        mode="max",
        perturbation_interval=24_000,
        hyperparam_mutations={"lr": [0.001, 0.002]},
    )
    class FakeTrial:
        def __init__(self, status):
            self.status = status

    target = FakeTrial(Trial.PAUSED)
    donor = FakeTrial(Trial.PAUSED)
    controller = SimpleNamespace(
        _staged_trials={target},
        _trial_to_actor={target: object()},
        _actor_cache=None,
    )
    monkeypatch.setattr(
        PopulationBasedTraining,
        "_exploit",
        lambda self, tune_controller, trial, trial_to_clone: None,
    )

    with pytest.raises(RuntimeError, match="already-created actor"):
        scheduler._exploit(controller, target, donor)


def test_sync_pbt_reattaches_a_paused_donors_own_stable_checkpoint(tmp_path: Path):
    from ray.train import Checkpoint
    from ray.tune.experiment.trial import Trial

    from itrader.rl_lib.train.pbt_scheduler import SeededPopulationBasedTraining

    stable_checkpoint_dir = tmp_path / "stable" / "donor" / "iter_000020"
    stable_checkpoint_dir.mkdir(parents=True)
    wrong_checkpoint_dir = tmp_path / "tune" / "other" / "checkpoint_000004"
    wrong_checkpoint_dir.mkdir(parents=True)

    scheduler = SeededPopulationBasedTraining(
        random_seed=271828,
        synch=True,
        time_attr="pbt_progress_steps",
        metric="evaluation/net_return_mean",
        mode="max",
        perturbation_interval=24_000,
        hyperparam_mutations={"lr": [0.001, 0.002]},
    )

    class FakeTrial:
        status = Trial.PAUSED
        placement_group_factory = "three-cpus"

    candidate = FakeTrial()
    candidate.run_metadata = SimpleNamespace(
        checkpoint_manager=SimpleNamespace(
            _latest_checkpoint_result=SimpleNamespace(
                checkpoint=Checkpoint.from_directory(str(wrong_checkpoint_dir)),
                metrics={"pbt_progress_steps": 120_000},
            )
        )
    )
    stable_checkpoint = Checkpoint.from_directory(str(stable_checkpoint_dir))
    scheduler._next_perturbation_sync = 120_000
    scheduler._trial_state[candidate] = SimpleNamespace(
        last_train_time=120_000,
        last_checkpoint=stable_checkpoint,
        last_result={"pbt_progress_steps": 120_000},
    )
    controller = SimpleNamespace(
        get_trials=lambda: [candidate],
        _staged_trials=set(),
        _trial_to_actor={},
        _actor_cache=None,
    )

    assert scheduler.choose_trial_to_run(controller) is candidate
    restored = candidate.run_metadata.checkpoint_manager._latest_checkpoint_result
    assert Path(restored.checkpoint.path) == stable_checkpoint_dir
    assert restored.metrics == {"pbt_progress_steps": 120_000}
    assert candidate in scheduler._awaiting_post_restore_result


def test_sync_pbt_capacity_resume_uses_exact_latest_checkpoint(tmp_path: Path):
    from itrader.rl_lib.evaluation.provenance import checkpoint_sha256
    from ray.train import Checkpoint
    from ray.tune.experiment.trial import Trial

    from itrader.rl_lib.train.pbt_scheduler import SeededPopulationBasedTraining

    boundary_dir = tmp_path / "boundary"
    boundary_dir.mkdir()
    exact_dir = tmp_path / "exact"
    exact_dir.mkdir()
    for checkpoint_dir, label in ((boundary_dir, b"old"), (exact_dir, b"new")):
        (checkpoint_dir / "algorithm_state.pkl").write_bytes(label)
        (checkpoint_dir / "rllib_checkpoint.json").write_text("{}", encoding="utf-8")
        policy_dir = checkpoint_dir / "policies" / "default_policy"
        policy_dir.mkdir(parents=True)
        (policy_dir / "policy_state.pkl").write_bytes(label)

    stable_root = tmp_path / "stable"
    local_path = tmp_path / "trial_00000"
    stable_dir = stable_root / local_path.name
    stable_dir.mkdir(parents=True)
    (stable_dir / "checkpoint_index.jsonl").write_text(
        json.dumps(
            {
                "pbt_progress_steps": 73_448,
                "stable_checkpoint": str(exact_dir),
                "api_stack_id": LEGACY_API_STACK_ID,
                "checkpoint_sha256": checkpoint_sha256(
                    exact_dir, api_stack_id=LEGACY_API_STACK_ID
                ),
                "reported_to_tune": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeTrial:
        status = Trial.PAUSED
        placement_group_factory = "three-cpus"

    candidate = FakeTrial()
    candidate.config = {
        "stable_metrics_root": str(stable_root),
        "provenance_enforced": True,
        "api_stack_id": LEGACY_API_STACK_ID,
    }
    candidate.local_path = str(local_path)
    candidate.local_experiment_path = str(tmp_path / "experiment")
    candidate.run_metadata = SimpleNamespace(
        checkpoint_manager=SimpleNamespace(_latest_checkpoint_result=None)
    )
    scheduler = SeededPopulationBasedTraining(
        random_seed=271828,
        synch=True,
        time_attr="pbt_progress_steps",
        metric="evaluation/net_return_mean",
        mode="max",
        perturbation_interval=24_000,
        hyperparam_mutations={"lr": [0.001, 0.002]},
    )
    scheduler._trial_state[candidate] = SimpleNamespace(
        last_train_time=73_448,
        last_checkpoint=Checkpoint.from_directory(str(boundary_dir)),
        last_result={"pbt_progress_steps": 73_448},
        orig_tag="candidate-tag",
    )
    controller = SimpleNamespace(
        get_trials=lambda: [candidate],
        _staged_trials=set(),
        _trial_to_actor={},
        _actor_cache=None,
    )

    assert scheduler.choose_trial_to_run(controller) is candidate
    restored = candidate.run_metadata.checkpoint_manager._latest_checkpoint_result
    assert Path(restored.checkpoint.path) == exact_dir


def test_sync_pbt_pending_exploit_reattaches_declared_donor_checkpoint(tmp_path: Path):
    from ray.train import Checkpoint
    from ray.tune.experiment.trial import Trial

    from itrader.rl_lib.evaluation.provenance import checkpoint_sha256
    from itrader.rl_lib.train.pbt_scheduler import SeededPopulationBasedTraining

    donor_dir = tmp_path / "donor_48960"
    donor_dir.mkdir()
    (donor_dir / "algorithm_state.pkl").write_bytes(b"donor")
    (donor_dir / "rllib_checkpoint.json").write_text("{}", encoding="utf-8")
    policy_dir = donor_dir / "policies" / "default_policy"
    policy_dir.mkdir(parents=True)
    (policy_dir / "policy_state.pkl").write_bytes(b"donor-policy")
    donor_digest = checkpoint_sha256(
        donor_dir, api_stack_id=LEGACY_API_STACK_ID
    )

    target_local_path = tmp_path / "trial_target"
    target_stable_dir = tmp_path / "stable" / target_local_path.name
    target_stable_dir.mkdir(parents=True)
    (target_stable_dir / "checkpoint_index.jsonl").write_text(
        json.dumps(
            {
                "pbt_progress_steps": 48_768,
                "stable_checkpoint": str(tmp_path / "target_48768"),
                "reported_to_tune": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeTrial:
        status = Trial.PAUSED

    target = FakeTrial()
    target.trial_id = "target-id"
    target.local_path = str(target_local_path)
    target.config = {
        "stable_metrics_root": str(tmp_path / "stable"),
        "provenance_enforced": True,
        "api_stack_id": LEGACY_API_STACK_ID,
        "pbt_restore_generation": 1,
        "pbt_restore_provenance": {
            "exploit_id": "exploit-1",
            "target_trial_id": "target-id",
            "donor_trial_id": "donor-id",
            "donor_progress_steps": 48_960,
            "donor_checkpoint_path": str(donor_dir),
            "donor_checkpoint_sha256": donor_digest,
            "api_stack_id": LEGACY_API_STACK_ID,
        },
    }
    target.run_metadata = SimpleNamespace(
        checkpoint_manager=SimpleNamespace(_latest_checkpoint_result=None)
    )
    scheduler = SeededPopulationBasedTraining(
        random_seed=271828,
        synch=True,
        time_attr="pbt_progress_steps",
        metric="evaluation/net_return_mean",
        mode="max",
        perturbation_interval=24_000,
        hyperparam_mutations={"lr": [0.001, 0.002]},
    )
    scheduler._trial_state[target] = SimpleNamespace(
        last_train_time=48_960,
        last_checkpoint=Checkpoint.from_directory(str(donor_dir)),
        last_result={
            "pbt_progress_steps": 48_960,
            "pbt_restore_generation": 0,
            "pbt_parent_exploit_id": None,
        },
        orig_tag="target-tag",
    )

    scheduler._reattach_stable_checkpoint(target)

    restored = target.run_metadata.checkpoint_manager._latest_checkpoint_result
    assert Path(restored.checkpoint.path) == donor_dir
    assert target in scheduler._awaiting_post_restore_result


def test_pbt_exploit_event_is_retained_outside_tune_logs(tmp_path: Path):
    from itrader.rl_lib.train.pbt_scheduler import SeededPopulationBasedTraining

    experiment_dir = tmp_path / "ray" / "experiment"
    experiment_dir.mkdir(parents=True)
    stable_root = tmp_path / "stable"
    target = SimpleNamespace(
        trial_id="target-id",
        local_experiment_path=str(experiment_dir),
        local_dir=str(experiment_dir),
        last_result={"training_iteration": 20, "pbt_progress_steps": 98_656},
        config={"stable_metrics_root": str(stable_root), "lr": 0.003},
    )
    donor = SimpleNamespace(
        trial_id="donor-id",
        local_experiment_path=str(experiment_dir),
        local_dir=str(experiment_dir),
        last_result={"training_iteration": 20, "pbt_progress_steps": 98_552},
        config={
            "stable_metrics_root": str(stable_root),
            "api_stack_id": LEGACY_API_STACK_ID,
            "lr": 0.002,
        },
    )
    scheduler = SeededPopulationBasedTraining(
        random_seed=271828,
        time_attr="pbt_progress_steps",
        metric="evaluation/net_return_mean",
        mode="max",
        perturbation_interval=48_000,
        hyperparam_mutations={"lr": [0.001, 0.002]},
    )

    scheduler._log_config_on_step(
        SimpleNamespace(orig_tag="target-tag"),
        SimpleNamespace(orig_tag="donor-tag"),
        target,
        donor,
        {"stable_metrics_root": str(stable_root), "lr": 0.001},
    )

    event_path = stable_root / "_pbt_events" / "experiment" / "events.jsonl"
    event = json.loads(event_path.read_text(encoding="utf-8"))
    assert event["target"] == {
        "trial_id": "target-id",
        "tag": "target-tag",
        "training_iteration": 20,
        "pbt_progress_steps": 98_656,
    }
    assert event["donor"] == {
        "trial_id": "donor-id",
        "tag": "donor-tag",
        "training_iteration": 20,
        "pbt_progress_steps": 98_552,
    }
    assert event["old_config"]["lr"] == pytest.approx(0.002)
    assert event["target_previous_config"]["lr"] == pytest.approx(0.003)
    assert event["donor_config"]["lr"] == pytest.approx(0.002)
    assert event["new_config"]["lr"] == pytest.approx(0.001)


def test_pbt_event_uses_explicit_experiment_name_with_ray_258_layout(
    tmp_path: Path,
):
    from itrader.rl_lib.train.pbt_scheduler import SeededPopulationBasedTraining

    stable_root = tmp_path / "stable"
    target = SimpleNamespace(
        local_experiment_path=str(tmp_path / "experiment" / "driver_artifacts"),
        config={
            "stable_metrics_root": str(stable_root),
            "pbt_experiment_name": "experiment",
        },
    )

    SeededPopulationBasedTraining._append_pbt_event(
        target,
        {"event": "probe"},
    )

    event_path = stable_root / "_pbt_events" / "experiment" / "events.jsonl"
    assert json.loads(event_path.read_text(encoding="utf-8"))["event"] == "probe"


def test_sync_pbt_exploit_receipt_uses_current_scheduler_state_and_exact_donor(
    tmp_path: Path,
):
    from ray.train import Checkpoint

    from itrader.rl_lib.evaluation.provenance import checkpoint_sha256
    from itrader.rl_lib.train.pbt_scheduler import SeededPopulationBasedTraining

    experiment_dir = tmp_path / "ray" / "experiment"
    experiment_dir.mkdir(parents=True)
    stable_root = tmp_path / "stable"
    donor_checkpoint = stable_root / "donor" / "checkpoints" / "iter_000010"
    policy_dir = donor_checkpoint / "policies" / "default_policy"
    policy_dir.mkdir(parents=True)
    (donor_checkpoint / "algorithm_state.pkl").write_bytes(b"weights")
    (donor_checkpoint / "rllib_checkpoint.json").write_text("{}\n", encoding="utf-8")
    (policy_dir / "policy_state.pkl").write_bytes(b"policy")

    target = SimpleNamespace(
        trial_id="target-id",
        local_experiment_path=str(experiment_dir),
        local_dir=str(experiment_dir),
        last_result={"training_iteration": 9, "pbt_progress_steps": 44_048},
        config={
            "stable_metrics_root": str(stable_root),
            "api_stack_id": LEGACY_API_STACK_ID,
            "lr": 0.003,
            "pbt_restore_generation": 2,
        },
    )
    donor = SimpleNamespace(
        trial_id="donor-id",
        local_experiment_path=str(experiment_dir),
        local_dir=str(experiment_dir),
        last_result={"training_iteration": 9, "pbt_progress_steps": 44_048},
        config={
            "stable_metrics_root": str(stable_root),
            "api_stack_id": LEGACY_API_STACK_ID,
            "lr": 0.002,
        },
    )
    target_state = SimpleNamespace(
        orig_tag="target-tag",
        last_train_time=48_856,
        last_result={"training_iteration": 10, "pbt_progress_steps": 48_856},
    )
    donor_state = SimpleNamespace(
        orig_tag="donor-tag",
        last_train_time=48_760,
        last_result={"training_iteration": 10, "pbt_progress_steps": 48_760},
        last_checkpoint=Checkpoint.from_directory(str(donor_checkpoint)),
    )
    new_config = {
        "stable_metrics_root": str(stable_root),
        "api_stack_id": LEGACY_API_STACK_ID,
        "lr": 0.001,
    }
    scheduler = SeededPopulationBasedTraining(
        random_seed=271828,
        synch=True,
        time_attr="pbt_progress_steps",
        metric="evaluation/net_return_mean",
        mode="max",
        perturbation_interval=24_000,
        hyperparam_mutations={"lr": [0.001, 0.002]},
    )

    scheduler._log_config_on_step(
        target_state,
        donor_state,
        target,
        donor,
        new_config,
    )

    provenance = new_config["pbt_restore_provenance"]
    assert new_config["pbt_restore_generation"] == 3
    assert provenance["target_progress_steps"] == 48_856
    assert provenance["donor_progress_steps"] == 48_760
    assert provenance["donor_checkpoint_path"] == str(donor_checkpoint)
    assert provenance["donor_checkpoint_sha256"] == checkpoint_sha256(
        donor_checkpoint, api_stack_id=LEGACY_API_STACK_ID
    )
    assert provenance["api_stack_id"] == LEGACY_API_STACK_ID
    assert len(provenance["exploit_id"]) == 64
    event_path = stable_root / "_pbt_events" / "experiment" / "events.jsonl"
    event = json.loads(event_path.read_text(encoding="utf-8"))
    assert event["target"]["pbt_progress_steps"] == 48_856
    assert event["donor"]["pbt_progress_steps"] == 48_760
    assert event["exploit_id"] == provenance["exploit_id"]


def test_pbt_does_not_clone_an_ineligible_donor():
    from itrader.rl_lib.train.pbt_scheduler import SeededPopulationBasedTraining

    scheduler = SeededPopulationBasedTraining(
        random_seed=271828,
        require_positive_donor=True,
        time_attr="pbt_progress_steps",
        metric="evaluation/economic_selection_score",
        mode="max",
        perturbation_interval=48_000,
        hyperparam_mutations={"lr": [0.001, 0.002]},
    )
    target = object()
    donor = object()
    scheduler._trial_state[target] = SimpleNamespace(last_checkpoint="stale")
    scheduler._trial_state[donor] = SimpleNamespace(last_score=-0.999)

    scheduler._checkpoint_or_exploit(
        target,
        SimpleNamespace(),
        upper_quantile=[donor],
        lower_quantile=[target],
    )

    assert scheduler._trial_state[target].last_checkpoint is None


def test_sync_pbt_recovers_a_paused_donor_checkpoint_from_the_stable_index(
    tmp_path: Path,
):
    from ray.tune.experiment.trial import Trial

    from itrader.rl_lib.train.pbt_scheduler import SeededPopulationBasedTraining

    stable_root = tmp_path / "stable"
    local_path = tmp_path / "ray" / "experiment" / "trial_00000"
    checkpoint_dir = stable_root / local_path.name / "checkpoints" / "iter_000020"
    checkpoint_dir.mkdir(parents=True)
    index_path = stable_root / local_path.name / "checkpoint_index.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(
        json.dumps(
            {
                "training_iteration": 20,
                "pbt_progress_steps": 99_792,
                "stable_checkpoint": str(checkpoint_dir),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeTrial:
        pass

    donor = FakeTrial()
    donor.status = Trial.PAUSED
    donor.temporary_state = SimpleNamespace(saving_to=None)
    donor.checkpoint = None
    donor.config = {"stable_metrics_root": str(stable_root)}
    donor.local_path = str(local_path)
    donor.local_experiment_path = str(local_path.parent)
    donor.local_dir = str(local_path.parent)
    donor.last_result = {"training_iteration": 19, "pbt_progress_steps": 94_992}
    donor.trial_id = "donor"
    scheduler = SeededPopulationBasedTraining(
        random_seed=271828,
        synch=True,
        time_attr="pbt_progress_steps",
        metric="evaluation/net_return_mean",
        mode="max",
        perturbation_interval=24_000,
        hyperparam_mutations={"lr": [0.001, 0.002]},
    )
    scheduler._trial_state[donor] = SimpleNamespace(
        last_checkpoint=None,
        last_train_time=99_792,
        last_result={"training_iteration": 20, "pbt_progress_steps": 99_792},
        orig_tag="donor-tag",
    )

    scheduler._checkpoint_or_exploit(
        donor,
        SimpleNamespace(),
        upper_quantile=[donor],
        lower_quantile=[],
    )

    recovered = scheduler._trial_state[donor].last_checkpoint
    assert Path(recovered.path) == checkpoint_dir
    events = [
        json.loads(line)
        for line in (
            stable_root / "_pbt_events" / "experiment" / "events.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert events[-1]["event"] == "stable_checkpoint_selected"
    assert events[-1]["trial"] == {
        "trial_id": "donor",
        "tag": "donor-tag",
        "training_iteration": 20,
        "pbt_progress_steps": 99_792,
    }


def test_positive_donor_skip_records_current_scheduler_state(tmp_path: Path):
    from itrader.rl_lib.train.pbt_scheduler import SeededPopulationBasedTraining

    stable_root = tmp_path / "stable"
    experiment_dir = tmp_path / "ray" / "experiment"
    class FakeTrial:
        pass

    target = FakeTrial()
    target.trial_id = "target-id"
    target.local_experiment_path = str(experiment_dir)
    target.last_result = {"training_iteration": 9, "pbt_progress_steps": 44_000}
    target.config = {"stable_metrics_root": str(stable_root)}
    donor = object()
    scheduler = SeededPopulationBasedTraining(
        random_seed=271828,
        require_positive_donor=True,
        time_attr="pbt_progress_steps",
        metric="evaluation/economic_selection_score",
        mode="max",
        perturbation_interval=24_000,
        hyperparam_mutations={"lr": [0.001, 0.002]},
    )
    scheduler._trial_state[target] = SimpleNamespace(
        orig_tag="target-tag",
        last_train_time=48_800,
        last_result={"training_iteration": 10, "pbt_progress_steps": 48_800},
        last_checkpoint="stale",
    )
    scheduler._trial_state[donor] = SimpleNamespace(last_score=-0.999)

    scheduler._checkpoint_or_exploit(
        target,
        SimpleNamespace(),
        upper_quantile=[donor],
        lower_quantile=[target],
    )

    event_path = stable_root / "_pbt_events" / "experiment" / "events.jsonl"
    event = json.loads(event_path.read_text(encoding="utf-8"))
    assert event["event"] == "exploit_skipped"
    assert event["trial"] == {
        "trial_id": "target-id",
        "tag": "target-tag",
        "training_iteration": 10,
        "pbt_progress_steps": 48_800,
    }


def test_sync_pbt_prefers_an_exact_stable_checkpoint_over_actor_save(
    tmp_path: Path,
):
    from ray.tune.experiment.trial import Trial

    from itrader.rl_lib.train.pbt_scheduler import SeededPopulationBasedTraining

    stable_root = tmp_path / "stable"
    experiment_dir = tmp_path / "ray" / "experiment"
    local_path = experiment_dir / "donor_trial"
    checkpoint_dir = stable_root / local_path.name / "checkpoints" / "iter_000020"
    checkpoint_dir.mkdir(parents=True)
    index_path = stable_root / local_path.name / "checkpoint_index.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(
        json.dumps(
            {
                "training_iteration": 20,
                "pbt_progress_steps": 99_792,
                "stable_checkpoint": str(checkpoint_dir),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeTrial:
        status = Trial.RUNNING

    donor = FakeTrial()
    donor.trial_id = "donor"
    donor.config = {
        "stable_metrics_root": str(stable_root),
        "pbt_target_steps": 168_000,
    }
    donor.local_path = str(local_path)
    donor.local_experiment_path = str(experiment_dir)
    donor.last_result = {
        "training_iteration": 20,
        "pbt_progress_steps": 99_792,
    }

    scheduler = SeededPopulationBasedTraining(
        random_seed=271828,
        synch=True,
        time_attr="pbt_progress_steps",
        metric="evaluation/net_return_mean",
        mode="max",
        perturbation_interval=24_000,
        hyperparam_mutations={"lr": [0.001, 0.002]},
    )
    scheduler._trial_state[donor] = SimpleNamespace(
        last_checkpoint=None,
        last_train_time=99_792,
        last_result=donor.last_result,
        orig_tag="donor-tag",
    )

    class FailingSaveController:
        @staticmethod
        def _schedule_trial_save(*args, **kwargs):
            raise AssertionError("synchronous PBT must use the exact stable checkpoint")

    scheduler._checkpoint_or_exploit(
        donor,
        FailingSaveController(),
        upper_quantile=[donor],
        lower_quantile=[],
    )

    recovered = scheduler._trial_state[donor].last_checkpoint
    assert Path(recovered.path) == checkpoint_dir


def test_sync_pbt_fails_closed_when_exact_stable_checkpoint_is_missing(
    tmp_path: Path,
):
    from ray.tune.experiment.trial import Trial

    from itrader.rl_lib.train.pbt_scheduler import SeededPopulationBasedTraining

    stable_root = tmp_path / "stable"
    experiment_dir = tmp_path / "ray" / "experiment"
    local_path = experiment_dir / "donor_trial"
    checkpoint_dir = stable_root / local_path.name / "checkpoints" / "iter_000019"
    checkpoint_dir.mkdir(parents=True)
    index_path = stable_root / local_path.name / "checkpoint_index.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(
        json.dumps(
            {
                "training_iteration": 19,
                "pbt_progress_steps": 95_000,
                "stable_checkpoint": str(checkpoint_dir),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeTrial:
        status = Trial.RUNNING

    donor = FakeTrial()
    donor.trial_id = "donor"
    donor.config = {
        "stable_metrics_root": str(stable_root),
        "pbt_target_steps": 168_000,
    }
    donor.local_path = str(local_path)
    donor.local_experiment_path = str(experiment_dir)
    donor.last_result = {
        "training_iteration": 20,
        "pbt_progress_steps": 99_792,
    }

    scheduler = SeededPopulationBasedTraining(
        random_seed=271828,
        synch=True,
        time_attr="pbt_progress_steps",
        metric="evaluation/net_return_mean",
        mode="max",
        perturbation_interval=24_000,
        hyperparam_mutations={"lr": [0.001, 0.002]},
    )
    scheduler._trial_state[donor] = SimpleNamespace(
        last_checkpoint=None,
        last_train_time=99_792,
        last_result=donor.last_result,
        orig_tag="donor-tag",
    )

    with pytest.raises(RuntimeError, match="exact stable checkpoint"):
        scheduler._checkpoint_or_exploit(
            donor,
            SimpleNamespace(),
            upper_quantile=[donor],
            lower_quantile=[],
        )


def test_sync_pbt_retries_exploit_when_tune_save_resolves_without_checkpoint(
    tmp_path: Path,
    monkeypatch,
):
    from ray.tune.schedulers import PopulationBasedTraining

    from itrader.rl_lib.train.pbt_scheduler import SeededPopulationBasedTraining

    stable_root = tmp_path / "stable"
    experiment_dir = tmp_path / "ray" / "experiment"
    donor_path = experiment_dir / "donor_trial"
    target_path = experiment_dir / "target_trial"
    checkpoint_dir = stable_root / donor_path.name / "checkpoints" / "iter_000020"
    checkpoint_dir.mkdir(parents=True)
    index_path = stable_root / donor_path.name / "checkpoint_index.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(
        json.dumps(
            {
                "training_iteration": 20,
                "pbt_progress_steps": 99_792,
                "stable_checkpoint": str(checkpoint_dir),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeTrial:
        pass

    donor = FakeTrial()
    donor.trial_id = "donor"
    donor.config = {"stable_metrics_root": str(stable_root)}
    donor.local_path = str(donor_path)
    donor.local_experiment_path = str(experiment_dir)
    donor.last_result = {"training_iteration": 20, "pbt_progress_steps": 99_792}
    target = FakeTrial()
    target.trial_id = "target"
    target.config = {"stable_metrics_root": str(stable_root)}
    target.local_path = str(target_path)
    target.local_experiment_path = str(experiment_dir)
    target.last_result = {"training_iteration": 20, "pbt_progress_steps": 99_248}

    scheduler = SeededPopulationBasedTraining(
        random_seed=271828,
        synch=True,
        time_attr="pbt_progress_steps",
        metric="evaluation/net_return_mean",
        mode="max",
        perturbation_interval=24_000,
        hyperparam_mutations={"lr": [0.001, 0.002]},
    )
    unresolved_save = SimpleNamespace(resolve=lambda: None)
    scheduler._trial_state[donor] = SimpleNamespace(
        last_checkpoint=unresolved_save,
        last_train_time=99_792,
        last_result=donor.last_result,
        orig_tag="donor-tag",
    )
    scheduler._trial_state[target] = SimpleNamespace(
        last_checkpoint=None,
        last_train_time=99_248,
        last_result=target.last_result,
        orig_tag="target-tag",
    )
    calls = []

    def fake_checkpoint_or_exploit(
        instance,
        trial,
        tune_controller,
        upper_quantile,
        lower_quantile,
    ):
        calls.append(instance._trial_state[donor].last_checkpoint)
        if not hasattr(instance._trial_state[donor].last_checkpoint, "resolve"):
            instance._num_perturbations += 1

    monkeypatch.setattr(
        PopulationBasedTraining,
        "_checkpoint_or_exploit",
        fake_checkpoint_or_exploit,
    )

    scheduler._checkpoint_or_exploit(
        target,
        SimpleNamespace(),
        upper_quantile=[donor],
        lower_quantile=[target],
    )

    assert len(calls) == 2
    assert hasattr(calls[0], "resolve")
    assert Path(calls[1].path) == checkpoint_dir
    assert scheduler._num_perturbations == 1


def test_sync_pbt_reuses_attached_checkpoint_before_first_post_restore_result(
    tmp_path: Path,
):
    from ray.train import Checkpoint
    from ray.tune.experiment.trial import Trial

    from itrader.rl_lib.train.pbt_scheduler import SeededPopulationBasedTraining

    checkpoint_dir = tmp_path / "donor_checkpoint"
    checkpoint_dir.mkdir()
    attached_checkpoint = Checkpoint.from_directory(str(checkpoint_dir))

    class FakeTrial:
        pass

    trial = FakeTrial()
    trial.status = Trial.RUNNING
    trial.config = {"pbt_target_steps": 288_000}
    trial.checkpoint = attached_checkpoint
    trial.last_result = {
        "training_iteration": 24,
        "pbt_progress_steps": 121_648,
    }
    scheduler = SeededPopulationBasedTraining(
        random_seed=271828,
        synch=True,
        time_attr="pbt_progress_steps",
        metric="evaluation/net_return_mean",
        mode="max",
        perturbation_interval=24_000,
        hyperparam_mutations={"lr": [0.001, 0.002]},
    )
    scheduler._trial_state[trial] = SimpleNamespace(
        last_checkpoint=None,
        last_train_time=121_648,
        last_result=trial.last_result,
        orig_tag="restored-tag",
    )
    scheduler._awaiting_post_restore_result = {trial}

    class FailingSaveController:
        @staticmethod
        def _schedule_trial_save(*args, **kwargs):
            raise AssertionError("save requested before first post-restore result")

    scheduler._checkpoint_or_exploit(
        trial,
        FailingSaveController(),
        upper_quantile=[trial],
        lower_quantile=[],
    )

    assert scheduler._trial_state[trial].last_checkpoint is attached_checkpoint


def test_sync_pbt_prefers_exact_current_checkpoint_over_stale_attached_checkpoint(
    tmp_path: Path,
):
    import json

    from ray.train import Checkpoint
    from ray.tune.experiment.trial import Trial

    from itrader.rl_lib.evaluation.provenance import checkpoint_sha256
    from itrader.rl_lib.train.pbt_scheduler import SeededPopulationBasedTraining

    stale_dir = tmp_path / "stale_checkpoint"
    stale_dir.mkdir()
    (stale_dir / "state.txt").write_text("stale", encoding="utf-8")
    exact_dir = tmp_path / "exact_checkpoint"
    exact_dir.mkdir()
    (exact_dir / "algorithm_state.pkl").write_bytes(b"algorithm")
    (exact_dir / "rllib_checkpoint.json").write_text("{}", encoding="utf-8")
    policy_dir = exact_dir / "policies" / "default_policy"
    policy_dir.mkdir(parents=True)
    (policy_dir / "policy_state.pkl").write_bytes(b"policy")
    stable_root = tmp_path / "stable_metrics"
    local_path = tmp_path / "trial_00000"
    stable_trial_dir = stable_root / local_path.name
    stable_trial_dir.mkdir(parents=True)
    (stable_trial_dir / "checkpoint_index.jsonl").write_text(
        json.dumps(
            {
                "pbt_progress_steps": 73_448,
                "stable_checkpoint": str(exact_dir),
                "api_stack_id": LEGACY_API_STACK_ID,
                "checkpoint_sha256": checkpoint_sha256(
                    exact_dir, api_stack_id=LEGACY_API_STACK_ID
                ),
                "reported_to_tune": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeTrial:
        pass

    trial = FakeTrial()
    trial.status = Trial.RUNNING
    trial.config = {
        "pbt_target_steps": 144_000,
        "stable_metrics_root": str(stable_root),
        "provenance_enforced": True,
        "api_stack_id": LEGACY_API_STACK_ID,
    }
    trial.local_path = str(local_path)
    trial.local_experiment_path = str(tmp_path / "experiment")
    trial.checkpoint = Checkpoint.from_directory(str(stale_dir))
    trial.last_result = {
        "training_iteration": 15,
        "pbt_progress_steps": 73_448,
    }
    scheduler = SeededPopulationBasedTraining(
        random_seed=271828,
        synch=True,
        time_attr="pbt_progress_steps",
        metric="evaluation/net_return_mean",
        mode="max",
        perturbation_interval=24_000,
        hyperparam_mutations={"lr": [0.001, 0.002]},
    )
    scheduler._trial_state[trial] = SimpleNamespace(
        last_checkpoint=None,
        last_train_time=73_448,
        last_result=trial.last_result,
        orig_tag="restored-tag",
    )
    scheduler._awaiting_post_restore_result = {trial}

    scheduler._checkpoint_or_exploit(
        trial,
        SimpleNamespace(),
        upper_quantile=[trial],
        lower_quantile=[],
    )
    scheduler._trial_state[trial].last_checkpoint = Checkpoint.from_directory(
        str(stale_dir)
    )
    trial.local_path = ""
    scheduler._checkpoint_or_exploit(
        trial,
        SimpleNamespace(),
        upper_quantile=[trial],
        lower_quantile=[],
    )

    selected = scheduler._trial_state[trial].last_checkpoint
    assert Path(selected.path) == exact_dir
    event_path = stable_root / "_pbt_events" / "experiment" / "events.jsonl"
    assert len(event_path.read_text(encoding="utf-8").splitlines()) == 1


def test_sync_pbt_tracks_only_trials_without_a_post_restore_result(monkeypatch):
    from ray.tune.schedulers import PopulationBasedTraining

    from itrader.rl_lib.train.pbt_scheduler import SeededPopulationBasedTraining

    scheduler = SeededPopulationBasedTraining(
        random_seed=271828,
        synch=True,
        time_attr="pbt_progress_steps",
        metric="evaluation/net_return_mean",
        mode="max",
        perturbation_interval=24_000,
        hyperparam_mutations={"lr": [0.001, 0.002]},
    )
    target = object()
    donor = object()
    monkeypatch.setattr(
        PopulationBasedTraining,
        "_exploit",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        PopulationBasedTraining,
        "on_trial_result",
        lambda *args, **kwargs: "CONTINUE",
    )

    scheduler._exploit(SimpleNamespace(), target, donor)

    assert target in scheduler._awaiting_post_restore_result

    scheduler.on_trial_result(
        SimpleNamespace(),
        target,
        {"pbt_progress_steps": 125_000, "evaluation/net_return_mean": 0.001},
    )

    assert target not in scheduler._awaiting_post_restore_result


def test_sync_pbt_does_not_exploit_at_the_terminal_exposure(monkeypatch):
    from ray.tune.schedulers import PopulationBasedTraining

    from itrader.rl_lib.train.pbt_scheduler import SeededPopulationBasedTraining

    scheduler = SeededPopulationBasedTraining(
        random_seed=271828,
        synch=True,
        time_attr="pbt_progress_steps",
        metric="evaluation/net_return_mean",
        mode="max",
        perturbation_interval=24_000,
        hyperparam_mutations={"lr": [0.001, 0.002]},
    )
    class FakeTrial:
        config = {"pbt_target_steps": 120_000}

    trial = FakeTrial()
    scheduler._trial_state[trial] = SimpleNamespace(last_train_time=120_000)
    calls = []
    monkeypatch.setattr(
        PopulationBasedTraining,
        "_checkpoint_or_exploit",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    scheduler._checkpoint_or_exploit(
        trial,
        SimpleNamespace(),
        upper_quantile=[trial],
        lower_quantile=[],
    )

    assert calls == []

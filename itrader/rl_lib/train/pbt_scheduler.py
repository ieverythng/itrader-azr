"""Ray PBT adapter with trial-population-owned random state."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
from typing import Iterator

import numpy as np
from ray.train import Checkpoint
from ray.train._internal.session import _TrainingResult
from ray.tune.experiment.trial import Trial
from ray.tune.schedulers import PopulationBasedTraining

from itrader.rl_lib.evaluation.provenance import (
    canonical_json_sha256,
    checkpoint_sha256,
    require_indexed_integer,
)


class SeededPopulationBasedTraining(PopulationBasedTraining):
    """Make PBT choices reproducible without leaking global RNG state.

    Ray Tune's population scheduler draws donor choices and mutations from the
    process-global Python and NumPy generators. The
    ``BasicVariantGenerator`` seed therefore does not reproduce PBT decisions
    by itself. This adapter gives scheduler callbacks an isolated random stream
    keyed by the population seed.
    """

    def __init__(self, *, random_seed: int, require_positive_donor: bool = False, **kwargs):
        super().__init__(**kwargs)
        seed = int(random_seed)
        self._require_positive_donor = bool(require_positive_donor)
        self._python_random_state = random.Random(seed).getstate()
        self._numpy_random_state = np.random.RandomState(seed).get_state()
        self._random_state_depth = 0
        self._awaiting_post_restore_result: set[Trial] = set()
        self._recorded_selection_events: set[tuple[str, int, str, str]] = set()
        self._stable_trial_dirs: dict[str, Path] = {}

    @contextmanager
    def _owned_random_state(self) -> Iterator[None]:
        if self._random_state_depth:
            self._random_state_depth += 1
            try:
                yield
            finally:
                self._random_state_depth -= 1
            return

        process_python_state = random.getstate()
        process_numpy_state = np.random.get_state()
        random.setstate(self._python_random_state)
        np.random.set_state(self._numpy_random_state)
        self._random_state_depth = 1
        try:
            yield
        finally:
            self._python_random_state = random.getstate()
            self._numpy_random_state = np.random.get_state()
            self._random_state_depth = 0
            random.setstate(process_python_state)
            np.random.set_state(process_numpy_state)

    def on_trial_add(self, tune_controller, trial):
        with self._owned_random_state():
            return super().on_trial_add(tune_controller, trial)

    def on_trial_result(self, tune_controller, trial, result):
        # A reported result proves the restored actor has completed at least
        # one train call, so Ray's Trainable.save() can safely consult its
        # actor-local ``_last_result`` again.
        self._awaiting_post_restore_result.discard(trial)
        with self._owned_random_state():
            return super().on_trial_result(tune_controller, trial, result)

    def choose_trial_to_run(self, tune_controller):
        candidate = super().choose_trial_to_run(tune_controller)
        if candidate is not None:
            self._reattach_stable_checkpoint(candidate)
            self._unstage_orphaned_candidate(tune_controller, candidate)
            return candidate
        if not self._synch:
            return None

        # Tune can strand the final synchronous member after its peers have
        # terminated: the member is PAUSED beyond the stale sync boundary, so
        # the base scheduler returns no candidate despite an idle cluster.
        # Let the least-advanced paused member report again and close its own
        # target. This fallback is reached only when the normal scheduler has
        # no runnable candidate.
        trials = list(tune_controller.get_trials())
        live_states = [
            self._trial_state.get(trial)
            for trial in trials
            if trial.status not in (Trial.TERMINATED, Trial.ERROR)
        ]
        if any(
            state is not None
            and int(getattr(state, "last_train_time", 0) or 0)
            < int(self._next_perturbation_sync)
            for state in live_states
        ):
            return None
        paused = [
            trial
            for trial in trials
            if trial.status in (Trial.PENDING, Trial.PAUSED)
        ]
        if not paused:
            return None
        candidate = min(
            paused,
            key=lambda trial: self._trial_state[trial].last_train_time,
        )
        self._reattach_stable_checkpoint(candidate)
        self._unstage_orphaned_candidate(tune_controller, candidate)
        return candidate

    def _reattach_stable_checkpoint(self, candidate) -> None:
        """Keep a paused upper donor attached to its own durable checkpoint.

        Tune may leave a resumed synchronous donor pointing at another
        trial's transient Tune checkpoint after a fallback exploit.  Lower
        trials have ``last_checkpoint`` cleared before exploitation, so this
        repair resolves the application-owned checkpoint at the candidate's
        exact progress clock before consulting Ray's cached boundary checkpoint.
        """
        state = self._trial_state.get(candidate)
        if state is None:
            return
        progress_steps = require_indexed_integer(getattr(state, "last_train_time", 0), "scheduler progress_steps")
        pending_exploit = self._attach_pending_exploit_checkpoint(candidate, state)
        if progress_steps > 0 and not pending_exploit:
            recovered = self._recover_stable_checkpoint(
                candidate,
                state,
                replace_pending=True,
                exact_progress=True,
                event_name="capacity_resume_exact_checkpoint_selected",
            )
            if not recovered and bool(
                (getattr(candidate, "config", {}) or {}).get(
                    "provenance_enforced",
                    False,
                )
            ):
                raise RuntimeError(
                    "PBT capacity resume has no exact stable checkpoint at "
                    f"pbt_progress_steps={progress_steps}"
                )
        checkpoint = getattr(state, "last_checkpoint", None)
        if not isinstance(checkpoint, Checkpoint):
            return
        run_metadata = getattr(candidate, "run_metadata", None)
        manager = getattr(run_metadata, "checkpoint_manager", None)
        if manager is None:
            return
        manager._latest_checkpoint_result = _TrainingResult(
            checkpoint=checkpoint,
            metrics=getattr(state, "last_result", {}) or {},
        )
        # Starting this paused trial creates a fresh actor restored from the
        # attached checkpoint. Until that actor reports once, Tune's
        # Trainable.save() has no actor-local ``_last_result`` and will assert.
        self._awaiting_post_restore_result.add(candidate)

    @staticmethod
    def _attach_pending_exploit_checkpoint(trial, state) -> bool:
        """Attach the declared donor while an exploit awaits its first report."""
        config = getattr(trial, "config", {}) or {}
        provenance = config.get("pbt_restore_provenance")
        if not isinstance(provenance, dict):
            return False
        exploit_id = str(provenance.get("exploit_id", ""))
        generation = require_indexed_integer(config.get("pbt_restore_generation", 0), "restore generation")
        if not exploit_id or generation < 1:
            return False
        result = getattr(state, "last_result", {}) or {}
        result_generation = require_indexed_integer(result.get("pbt_restore_generation", 0), "result restore generation")
        result_exploit_id = str(result.get("pbt_parent_exploit_id", "") or "")
        if result_generation >= generation and result_exploit_id == exploit_id:
            return False

        trial_id = str(getattr(trial, "trial_id", ""))
        expected_target = str(provenance.get("target_trial_id", ""))
        donor_steps = require_indexed_integer(provenance.get("donor_progress_steps"), "donor progress_steps")
        current_steps = require_indexed_integer(getattr(state, "last_train_time", None), "scheduler progress_steps")
        donor_path = Path(str(provenance.get("donor_checkpoint_path", "")))
        expected_digest = str(provenance.get("donor_checkpoint_sha256", ""))
        api_stack_id = str(config.get("api_stack_id", ""))
        if not api_stack_id or provenance.get("api_stack_id") != api_stack_id:
            raise RuntimeError("Pending PBT exploit API stack does not match trial config")
        if expected_target != trial_id:
            raise RuntimeError("Pending PBT exploit target identity does not match trial")
        if donor_steps != current_steps:
            raise RuntimeError(
                "Pending PBT exploit donor progress does not match scheduler clock: "
                f"{donor_steps} != {current_steps}"
            )
        try:
            observed_digest = checkpoint_sha256(
                donor_path, api_stack_id=api_stack_id
            )
        except (OSError, ValueError) as exc:
            raise RuntimeError("Pending PBT exploit donor checkpoint is invalid") from exc
        if not expected_digest or observed_digest != expected_digest:
            raise RuntimeError("Pending PBT exploit donor checkpoint digest does not match")
        state.last_checkpoint = Checkpoint.from_directory(str(donor_path.resolve()))
        return True

    @staticmethod
    def _unstage_orphaned_candidate(tune_controller, candidate) -> None:
        """Clear Tune's stale staged marker after a synchronous exploit.

        A genuinely pending actor is registered in ``_trial_to_actor`` as soon
        as Tune schedules it.  A staged candidate with no corresponding actor
        is therefore an orphan: leaving it staged makes Tune repeatedly try an
        empty actor cache and report an idle cluster forever.
        """
        controller = tune_controller
        staged = getattr(controller, "_staged_trials", set())
        trial_to_actor = getattr(controller, "_trial_to_actor", {})
        if candidate not in staged or candidate in trial_to_actor:
            return
        staged.remove(candidate)
        actor_cache = getattr(controller, "_actor_cache", None)
        if actor_cache is not None:
            actor_cache.decrease_max(candidate.placement_group_factory)

    def _get_new_config(self, trial, trial_to_clone):
        with self._owned_random_state():
            return super()._get_new_config(trial, trial_to_clone)

    def _record_skipped_exploit(
        self,
        trial,
        state,
        *,
        reason: str,
        candidate_donors=(),
    ) -> None:
        event = {
            "event": "exploit_skipped",
            "reason": reason,
            "trial": self._trial_event_identity(
                trial,
                getattr(state, "orig_tag", ""),
                result=getattr(state, "last_result", None),
            ),
        }
        donors = list(candidate_donors)
        if donors:
            event["candidate_donors"] = [
                self._trial_event_identity(
                    donor,
                    self._trial_state[donor].orig_tag,
                    result=getattr(self._trial_state[donor], "last_result", None),
                )
                for donor in donors
            ]
        self._append_pbt_event(trial, event)

    def _checkpoint_or_exploit(
        self,
        trial,
        tune_controller,
        upper_quantile,
        lower_quantile,
    ):
        state = self._trial_state[trial]
        config = getattr(trial, "config", {}) or {}
        target_steps = int(config.get("pbt_target_steps", 0) or 0)
        if target_steps and int(state.last_train_time) >= target_steps:
            return
        if trial in upper_quantile and trial in self._awaiting_post_restore_result:
            if self._attach_pending_exploit_checkpoint(trial, state):
                self._num_checkpoints += 1
                return
            recovered = self._recover_stable_checkpoint(
                trial,
                state,
                replace_pending=True,
                exact_progress=True,
                event_name="post_restore_exact_checkpoint_selected",
            )
            if recovered:
                self._num_checkpoints += 1
                return
            if bool(config.get("provenance_enforced", False)):
                raise RuntimeError(
                    "Synchronous PBT upper trial is awaiting its first post-restore "
                    "result and has no exact stable checkpoint at "
                    f"pbt_progress_steps={int(state.last_train_time)}"
                )
            attached_checkpoint = getattr(trial, "checkpoint", None)
            if not isinstance(attached_checkpoint, Checkpoint):
                raise RuntimeError(
                    "PBT trial is awaiting its first post-restore result but has "
                    "no attached checkpoint to reuse"
                )
            state.last_checkpoint = attached_checkpoint
            self._num_checkpoints += 1
            self._append_pbt_event(
                trial,
                {
                    "event": "post_restore_checkpoint_reused",
                    "trial": self._trial_event_identity(
                        trial,
                        state.orig_tag,
                        result=getattr(state, "last_result", None),
                    ),
                    "checkpoint": str(attached_checkpoint.path),
                },
            )
            return
        if self._synch and trial in upper_quantile:
            recovered = self._recover_stable_checkpoint(
                trial,
                state,
                replace_pending=True,
                exact_progress=True,
                event_name="stable_checkpoint_selected",
            )
            if not recovered:
                raise RuntimeError(
                    "Synchronous PBT upper trial has no exact stable checkpoint "
                    f"at pbt_progress_steps={int(state.last_train_time)}"
                )
            # Function trainables have already materialized this immutable
            # checkpoint before ``tune.report`` invokes the scheduler.
            # The durable sidecar is the synchronous donor authority, avoiding
            # a redundant actor save before the restored function reports.
            self._num_checkpoints += 1
            return
        if self._require_positive_donor and trial in lower_quantile:
            eligible_donors = [
                donor
                for donor in upper_quantile
                if self._trial_state[donor].last_score > 0.0
            ]
            if not eligible_donors:
                state.last_checkpoint = None
                self._record_skipped_exploit(
                    trial,
                    state,
                    reason="no_positive_donor",
                )
                return
            upper_quantile = eligible_donors
        perturbations_before = self._num_perturbations
        result = super()._checkpoint_or_exploit(
            trial,
            tune_controller,
            upper_quantile,
            lower_quantile,
        )
        if trial in upper_quantile and not state.last_checkpoint:
            self._recover_stable_checkpoint(trial, state)
        if trial in lower_quantile and self._num_perturbations == perturbations_before:
            recovered_donors = [
                donor
                for donor in upper_quantile
                if self._recover_stable_checkpoint(
                    donor,
                    self._trial_state[donor],
                    replace_pending=True,
                )
            ]
            if recovered_donors:
                super()._checkpoint_or_exploit(
                    trial,
                    tune_controller,
                    recovered_donors,
                    lower_quantile,
                )
        if trial in lower_quantile and self._num_perturbations == perturbations_before:
            self._record_skipped_exploit(
                trial,
                state,
                reason="donor_checkpoint_unavailable",
                candidate_donors=upper_quantile,
            )
        return result

    def _exploit(self, tune_controller, trial, trial_to_clone):
        was_paused = getattr(trial, "status", None) == Trial.PAUSED
        super()._exploit(tune_controller, trial, trial_to_clone)
        if was_paused:
            controller = tune_controller
            trial_to_actor = getattr(controller, "_trial_to_actor", {})
            if trial in trial_to_actor:
                raise RuntimeError(
                    "Synchronous PBT exploit target has an already-created actor; "
                    "refusing an unverified stale restore"
                )
            self._unstage_orphaned_candidate(tune_controller, trial)
        self._awaiting_post_restore_result.add(trial)

    def _recover_stable_checkpoint(
        self,
        trial,
        state,
        *,
        replace_pending: bool = False,
        exact_progress: bool = False,
        event_name: str = "stable_checkpoint_fallback",
    ) -> bool:
        if getattr(state, "last_checkpoint", None) and not replace_pending:
            return False
        config = getattr(trial, "config", {}) or {}
        stable_root = str(config.get("stable_metrics_root", "")).strip()
        local_path = str(getattr(trial, "local_path", "") or "").strip()
        if not stable_root:
            return False
        trial_id = str(getattr(trial, "trial_id", ""))
        trial_key = trial_id or str(id(trial))
        stable_trial_dir = self._stable_trial_dirs.get(trial_key)
        if stable_trial_dir is None and local_path:
            stable_trial_dir = Path(stable_root) / Path(local_path).name
        if stable_trial_dir is None and trial_id:
            matches = sorted(Path(stable_root).glob(f"*_{trial_id}_*"))
            if len(matches) == 1:
                stable_trial_dir = matches[0]
        if stable_trial_dir is None:
            return False
        index_path = stable_trial_dir / "checkpoint_index.jsonl"
        if not index_path.is_file():
            return False
        self._stable_trial_dirs[trial_key] = stable_trial_dir

        current_steps = require_indexed_integer(getattr(state, "last_train_time", None), "scheduler progress_steps")
        candidates = []
        for line in index_path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            progress_steps = require_indexed_integer(record.get("pbt_progress_steps"), "pbt_progress_steps")
            checkpoint_path = Path(str(record.get("stable_checkpoint", "")))
            progress_matches = (
                progress_steps == current_steps
                if exact_progress
                else progress_steps <= current_steps
            )
            if progress_matches and checkpoint_path.is_dir():
                candidates.append(
                    (
                        progress_steps,
                        checkpoint_path,
                        record.get("checkpoint_sha256"),
                        record.get("api_stack_id"),
                    )
                )
        if not candidates:
            return False

        progress_steps, checkpoint_path, indexed_digest, indexed_api_stack = max(
            candidates, key=lambda item: item[0]
        )
        if sum(item[0] == progress_steps for item in candidates) != 1:
            raise ValueError("Ambiguous indexed checkpoint digest at selected progress")
        current_checkpoint = getattr(state, "last_checkpoint", None)
        checkpoint_path = checkpoint_path.resolve()
        current_matches = isinstance(current_checkpoint, Checkpoint) and (
            Path(current_checkpoint.path).resolve() == checkpoint_path
        )
        if indexed_digest is not None or config.get("provenance_enforced", False):
            from itrader.rl_lib.evaluation.provenance import verify_checkpoint_digest

            api_stack_id = str(config.get("api_stack_id", ""))
            if not api_stack_id or indexed_api_stack != api_stack_id:
                raise ValueError("Indexed checkpoint API stack does not match trial config")
            checkpoint_digest = verify_checkpoint_digest(
                checkpoint_path,
                indexed_digest,
                api_stack_id=api_stack_id,
            )
        else:
            # Legacy compatibility is explicitly unverified, not a new attestation.
            checkpoint_digest = None
        if not current_matches:
            state.last_checkpoint = Checkpoint.from_directory(str(checkpoint_path))
        selection_key = (
            str(getattr(trial, "trial_id", "")),
            progress_steps,
            str(checkpoint_path),
            str(checkpoint_digest or ""),
        )
        if selection_key not in self._recorded_selection_events:
            self._recorded_selection_events.add(selection_key)
            self._append_pbt_event(
                trial,
                {
                    "event": event_name,
                    "api_stack_id": str(config.get("api_stack_id", "")),
                    "trial": self._trial_event_identity(
                        trial,
                        getattr(state, "orig_tag", ""),
                        result=getattr(state, "last_result", None),
                    ),
                    "checkpoint_progress_steps": progress_steps,
                    "stable_checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": checkpoint_digest,
                    "checkpoint_index": str(index_path),
                },
            )
        return True

    def _log_config_on_step(
        self,
        trial_state,
        new_state,
        trial,
        trial_to_clone,
        new_config,
    ):
        target = self._trial_event_identity(
            trial,
            trial_state.orig_tag,
            result=getattr(trial_state, "last_result", None),
        )
        donor = self._trial_event_identity(
            trial_to_clone,
            new_state.orig_tag,
            result=getattr(new_state, "last_result", None),
        )
        if (getattr(trial_to_clone, "config", {}) or {}).get("provenance_enforced", False):
            if not self._recover_stable_checkpoint(
                trial_to_clone, new_state, replace_pending=True, exact_progress=True,
            ):
                raise ValueError("PBT donor has no uniquely indexed scored checkpoint digest")
        donor_checkpoint = getattr(new_state, "last_checkpoint", None)
        exploit_id = None
        restore_provenance = None
        if isinstance(donor_checkpoint, Checkpoint):
            checkpoint_path = Path(donor_checkpoint.path).resolve(strict=True)
            donor_config = getattr(trial_to_clone, "config", {}) or {}
            target_config = getattr(trial, "config", {}) or {}
            api_stack_id = str(donor_config.get("api_stack_id", ""))
            if not api_stack_id or target_config.get("api_stack_id") != api_stack_id:
                raise RuntimeError("PBT cannot exploit across different API stacks")
            checkpoint_digest = checkpoint_sha256(
                checkpoint_path, api_stack_id=api_stack_id
            )
            target_generation = int(
                (getattr(trial, "config", {}) or {}).get(
                    "pbt_restore_generation",
                    0,
                )
            ) + 1
            exploit_payload = {
                "schema": "pbt-exploit-v1",
                "experiment_name": self._experiment_name(trial),
                "target_trial_id": target["trial_id"],
                "target_progress_steps": target["pbt_progress_steps"],
                "donor_trial_id": donor["trial_id"],
                "donor_progress_steps": donor["pbt_progress_steps"],
                "donor_checkpoint_sha256": checkpoint_digest,
                "api_stack_id": api_stack_id,
                "new_config_sha256": canonical_json_sha256(new_config),
            }
            exploit_id = canonical_json_sha256(exploit_payload)
            restore_provenance = {
                "schema_version": 1,
                "exploit_id": exploit_id,
                "target_trial_id": target["trial_id"],
                "target_progress_steps": target["pbt_progress_steps"],
                "donor_trial_id": donor["trial_id"],
                "donor_progress_steps": donor["pbt_progress_steps"],
                "donor_checkpoint_path": str(checkpoint_path),
                "donor_checkpoint_sha256": checkpoint_digest,
                "api_stack_id": api_stack_id,
            }
            new_config["pbt_restore_generation"] = target_generation
            new_config["pbt_restore_provenance"] = restore_provenance
        elif self._synch:
            raise RuntimeError(
                "Synchronous PBT exploit has no content-addressable donor checkpoint"
            )

        super()._log_config_on_step(
            trial_state,
            new_state,
            trial,
            trial_to_clone,
            new_config,
        )
        event = {
            "event": "exploit",
            "api_stack_id": str(
                (getattr(trial, "config", {}) or {}).get("api_stack_id", "")
            ),
            "exploit_id": exploit_id,
            "target_restore_generation": new_config.get("pbt_restore_generation"),
            "target": target,
            "donor": donor,
            "donor_checkpoint": restore_provenance,
            "target_previous_config": trial.config,
            "donor_config": trial_to_clone.config,
            # Retain the Ray-compatible historical name for downstream readers.
            "old_config": trial_to_clone.config,
            "new_config": new_config,
        }
        self._append_pbt_event(trial, event)

    @staticmethod
    def _append_pbt_event(trial, event: dict) -> None:
        config = getattr(trial, "config", {}) or {}
        stable_root = str(config.get("stable_metrics_root", "")).strip()
        if not stable_root:
            return
        experiment_name = SeededPopulationBasedTraining._experiment_name(trial)
        if not experiment_name:
            return
        event_path = Path(stable_root) / "_pbt_events" / experiment_name / "events.jsonl"
        event_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            **event,
        }
        with event_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _experiment_name(trial) -> str:
        config = getattr(trial, "config", {}) or {}
        explicit_name = str(config.get("pbt_experiment_name", "")).strip()
        if explicit_name:
            return explicit_name
        local_experiment_path = str(
            getattr(trial, "local_experiment_path", "") or ""
        ).strip()
        return Path(local_experiment_path).name if local_experiment_path else ""

    @staticmethod
    def _trial_event_identity(trial, tag: str, *, result: dict | None = None) -> dict:
        result = result or getattr(trial, "last_result", {}) or {}
        return {
            "trial_id": str(getattr(trial, "trial_id", "")),
            "tag": tag,
            "training_iteration": int(result.get("training_iteration", 0)),
            "pbt_progress_steps": int(result.get("pbt_progress_steps", 0)),
        }

from __future__ import annotations

import sys
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

from ray.rllib.algorithms.callbacks import DefaultCallbacks


class AutoPlotterCallback(DefaultCallbacks):
    """
    Safe, headless auto-plotter that never interrupts training.
    - Detects the trial dir from `result["logdir"]` or Algorithm.logdir.
    - Calls tools/plot_ray_results.py inside the trial dir.
    - Swallows all exceptions (prints warnings instead).
    """

    def __init__(
        self,
        plot_interval: Optional[int] = None,
        plot_script_path: Optional[str] = None,
        auto_plot: bool = True,
        plot_at_end_of_loop: bool = False,
    ):
        super().__init__()
        self.plot_interval = plot_interval
        self.auto_plot = auto_plot
        self.plot_at_end_of_loop = plot_at_end_of_loop
        self.last_plot_iter = 0

        # Locate plotting script (…/tools/plot_ray_results.py)
        if plot_script_path is None:
            plot_script_path = Path(__file__).parent.parents[1] / "tools" / "plot_ray_results.py"
        self.plot_script_path = Path(plot_script_path)

        if not self.plot_script_path.exists():
            print(f"[AutoPlotter] Warning: plot script not found at {self.plot_script_path}")
            self.auto_plot = False

    # ------------------------------------------------------------------ #
    def on_train_result(self, *, algorithm, result: Dict[str, Any], **kwargs):
        if not self.auto_plot:
            return
        try:
            current_iter = int(result.get("training_iteration", 0))
            if self.plot_at_end_of_loop:
                if self.plot_interval is None:
                    do_plot = True
                else:
                    do_plot = (current_iter - self.last_plot_iter) >= self.plot_interval
            else:
                do_plot = (
                    self.plot_interval is not None and current_iter - self.last_plot_iter >= self.plot_interval
                )
            if do_plot:
                self._generate_plots(algorithm, result)
                self.last_plot_iter = current_iter
        except (TypeError, ValueError, OSError, subprocess.SubprocessError) as e:
            print(f"[AutoPlotter] Suppressed plotting error in on_train_result: {e}")

    # ------------------------------------------------------------------ #
    @staticmethod
    def _looks_like_trial_dir(p: Path) -> bool:
        return (
            any((p / name).exists() for name in ("progress.csv", "result.json", "params.json", "metrics_full.jsonl"))
            or any((p / "rllib" / name).exists() for name in ("progress.csv", "result.json", "metrics_full.jsonl"))
        )

    @staticmethod
    def _trial_mtime(p: Path) -> float:
        for name in ("progress.csv", "result.json", "params.json", "metrics_full.jsonl"):
            for fp in (p / name, p / "rllib" / name):
                if fp.exists():
                    return fp.stat().st_mtime
        return p.stat().st_mtime

    # ------------------------------------------------------------------ #
    def _trial_dir(self, algorithm, result: Dict[str, Any]) -> Optional[Path]:
        """Resolve the Ray Tune trial directory robustly. Prioritizes Phase_A paths."""
        # 0) Prefer Ray AIR session trial dir (most reliable when using session.report)
        try:
            from ray.air import session
            td = session.get_trial_dir()
            if td:
                p = Path(td)
                if p.exists() and self._looks_like_trial_dir(p):
                    return p
        except Exception:
            pass

        # 1) result has logdir (preferred)
        try:
            if "logdir" in result and result["logdir"]:
                p = Path(result["logdir"])
                if p.exists() and self._looks_like_trial_dir(p):
                    return p
        except Exception:
            pass

        # 2) Algorithm.logdir
        try:
            if hasattr(algorithm, "logdir") and algorithm.logdir:
                p = Path(algorithm.logdir)
                if p.exists() and self._looks_like_trial_dir(p):
                    return p
        except Exception:
            pass

        # 3) Fallback: scan top-level experiment dirs and pick the most recent trial.
        try:
            base = Path.home() / "ray_results"
            if base.exists():
                # Check only top-level experiment dirs to avoid deep rglob scans.
                top_level = [d for d in base.iterdir() if d.is_dir()]
                all_candidates = []
                for exp_dir in top_level:
                    for trial_dir in exp_dir.iterdir():
                        if trial_dir.is_dir() and self._looks_like_trial_dir(trial_dir):
                            all_candidates.append(trial_dir)
                if all_candidates:
                    return max(all_candidates, key=self._trial_mtime)
        except OSError:
            pass

        return None

    # ------------------------------------------------------------------ #
    @staticmethod
    def _stable_trial_dir(trial_dir: Path) -> Path:
        return Path.home() / "ray_results" / "_stable_metrics" / trial_dir.name

    # ------------------------------------------------------------------ #
    def _generate_plots(self, algorithm, result: Dict[str, Any]):
        try:
            trial_dir = self._trial_dir(algorithm, result)
            if not trial_dir:
                print("[AutoPlotter] Warning: could not resolve trial directory; skipping plots.")
                return

            stable_dir = self._stable_trial_dir(trial_dir)
            stable_metrics = stable_dir / "metrics_full.jsonl"
            out_dir = stable_dir / "plots"
            if stable_metrics.exists() and stable_metrics.stat().st_size > 0:
                cmd = [
                    sys.executable,
                    str(self.plot_script_path),
                    "--metrics-jsonl",
                    str(stable_metrics),
                    "--out",
                    str(out_dir),
                ]
            else:
                cmd = [
                    sys.executable,
                    str(self.plot_script_path),
                    str(trial_dir),
                    "--out",
                    str(out_dir),
                ]
            print(f"[AutoPlotter] Generating plots @ iter {result.get('training_iteration', 0)}")
            print(f"[AutoPlotter] Trial directory: {trial_dir}")
            print(f"[AutoPlotter] Plot output dir: {out_dir}")

            # Run inside the trial dir so relative output is stable.
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=trial_dir, timeout=120)

            if proc.returncode == 0:
                print("[AutoPlotter] Plots generated.")
                if proc.stdout:
                    print(proc.stdout)
            else:
                print(f"[AutoPlotter] Plot generation failed (code={proc.returncode})")
                if proc.stderr:
                    print(proc.stderr)
        except (OSError, subprocess.SubprocessError, subprocess.TimeoutExpired) as e:
            print(f"[AutoPlotter] Suppressed plotting error: {e}")

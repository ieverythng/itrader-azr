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
        plot_at_end_of_loop: bool = True,
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
            do_plot = self.plot_at_end_of_loop or (
                self.plot_interval is not None and current_iter - self.last_plot_iter >= self.plot_interval
            )
            if do_plot:
                self._generate_plots(algorithm, result)
                self.last_plot_iter = current_iter
        except Exception as e:
            print(f"[AutoPlotter] Suppressed plotting error in on_train_result: {e}")

    # ------------------------------------------------------------------ #
    def _trial_dir(self, algorithm, result: Dict[str, Any]) -> Optional[Path]:
        """Resolve the Ray Tune trial directory robustly. Prioritizes Phase_A paths."""
        # 0) Prefer Ray AIR session trial dir (most reliable when using session.report)
        try:
            from ray.air import session
            td = session.get_trial_dir()
            if td:
                p = Path(td)
                if (p / "progress.csv").exists():
                    return p
        except Exception:
            pass

        # 1) result has logdir (preferred)
        try:
            if "logdir" in result and result["logdir"]:
                p = Path(result["logdir"])
                if (p / "progress.csv").exists():
                    return p
        except Exception:
            pass

        # 2) Algorithm.logdir
        try:
            if hasattr(algorithm, "logdir") and algorithm.logdir:
                p = Path(algorithm.logdir)
                if (p / "progress.csv").exists():
                    return p
        except Exception:
            pass

        # 3) Fallback: Prioritize Phase_A paths, then most recent
        try:
            base = Path.home() / "ray_results"
            if base.exists():
                # First try Phase_A/* directories
                phase_a_base = base / "Phase_A"
                if phase_a_base.exists():
                    candidates = [d for d in phase_a_base.iterdir() if (d.is_dir() and (d / "progress.csv").exists())]
                    if candidates:
                        # Return most recent Phase_A trial
                        return max(candidates, key=lambda d: (d / "progress.csv").stat().st_mtime)
                
                # Fallback: search all directories (but prefer Phase_A structure)
                all_candidates = [d for d in base.rglob("*") if (d.is_dir() and (d / "progress.csv").exists())]
                if all_candidates:
                    # Prefer directories under Phase_A
                    phase_a_candidates = [d for d in all_candidates if "Phase_A" in str(d)]
                    if phase_a_candidates:
                        return max(phase_a_candidates, key=lambda d: (d / "progress.csv").stat().st_mtime)
                    # Otherwise most recent overall
                    return max(all_candidates, key=lambda d: (d / "progress.csv").stat().st_mtime)
        except Exception:
            pass

        return None

    # ------------------------------------------------------------------ #
    def _generate_plots(self, algorithm, result: Dict[str, Any]):
        try:
            trial_dir = self._trial_dir(algorithm, result)
            if not trial_dir:
                print("[AutoPlotter] Warning: could not resolve trial directory; skipping plots.")
                return

            cmd = [sys.executable, str(self.plot_script_path), str(trial_dir)]
            print(f"\n🎨 Generating plots @ iter {result.get('training_iteration', 0)}")
            print(f"📁 Trial directory: {trial_dir}")

            # Run inside the trial dir so relative output is stable.
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=trial_dir)

            if proc.returncode == 0:
                print("✅ Plots generated.")
                if proc.stdout:
                    print(proc.stdout)
            else:
                print(f"❌ Plot generation failed (code={proc.returncode})")
                if proc.stderr:
                    print(proc.stderr)
        except Exception as e:
            print(f"[AutoPlotter] Suppressed plotting error: {e}")
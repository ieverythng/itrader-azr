from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from ray import tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks


class AutoPlotterCallback(DefaultCallbacks):
    """
    Safe, headless auto-plotter that never interrupts training.
    - Detects the trial directory from Ray 2.58's Tune context.
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

        if plot_script_path is None:
            plot_script_path = Path(__file__).parent.parents[1] / "tools" / "plot_ray_results.py"
        self.plot_script_path = Path(plot_script_path)

        if not self.plot_script_path.exists():
            print(f"[AutoPlotter] Warning: plot script not found at {self.plot_script_path}")
            self.auto_plot = False

    def _should_plot(self, current_iter: int) -> bool:
        if self.plot_at_end_of_loop:
            if self.plot_interval is None:
                return True
            return (current_iter - self.last_plot_iter) >= self.plot_interval
        return self.plot_interval is not None and (current_iter - self.last_plot_iter) >= self.plot_interval

    def on_train_result(self, *, algorithm, result: Dict[str, Any], **kwargs):
        if not self.auto_plot:
            return
        try:
            current_iter = int(result.get("training_iteration", 0))
            if self._should_plot(current_iter):
                self._generate_plots(algorithm, result)
                self.last_plot_iter = current_iter
        except (TypeError, ValueError, OSError, subprocess.SubprocessError) as e:
            print(f"[AutoPlotter] Suppressed plotting error in on_train_result: {e}")

    def _trial_dir(self) -> Optional[Path]:
        """Resolve the current function-trainable directory from Tune context."""
        try:
            raw_trial_dir = tune.get_context().get_trial_dir()
        except RuntimeError:
            return None
        if not raw_trial_dir:
            return None
        trial_dir = Path(raw_trial_dir)
        return trial_dir if trial_dir.is_dir() else None

    @staticmethod
    def _stable_trial_dir(trial_dir: Path) -> Path:
        return Path.home() / "ray_results" / "_stable_metrics" / trial_dir.name

    def _plot_command(self, trial_dir: Path, stable_dir: Path) -> tuple[list[str], Path]:
        stable_metrics = stable_dir / "metrics_full.jsonl"
        out_dir = stable_dir / "plots"
        if stable_metrics.exists() and stable_metrics.stat().st_size > 0:
            return (
                [
                    sys.executable,
                    str(self.plot_script_path),
                    "--metrics-jsonl",
                    str(stable_metrics),
                    "--out",
                    str(out_dir),
                ],
                out_dir,
            )
        return (
            [
                sys.executable,
                str(self.plot_script_path),
                str(trial_dir),
                "--out",
                str(out_dir),
            ],
            out_dir,
        )

    @staticmethod
    def _run_plot_command(cmd: list[str], trial_dir: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(cmd, capture_output=True, text=True, cwd=trial_dir, timeout=120)

    def _generate_plots(self, algorithm, result: Dict[str, Any]):
        try:
            trial_dir = self._trial_dir()
            if not trial_dir:
                print("[AutoPlotter] Warning: could not resolve trial directory; skipping plots.")
                return

            stable_dir = self._stable_trial_dir(trial_dir)
            cmd, out_dir = self._plot_command(trial_dir, stable_dir)
            print(f"[AutoPlotter] Generating plots @ iter {result.get('training_iteration', 0)}")
            print(f"[AutoPlotter] Trial directory: {trial_dir}")
            print(f"[AutoPlotter] Plot output dir: {out_dir}")

            # Run inside the trial dir so relative output is stable.
            proc = self._run_plot_command(cmd, trial_dir)

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

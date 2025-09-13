from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess
from ray.rllib.algorithms.callbacks import DefaultCallbacks


class AutoPlotterCallback(DefaultCallbacks):
    """
    Generates plots during training by calling tools/plot_ray_results.py
    after each training iteration (or at a configured interval).
    """

    def __init__(
        self,
        plot_interval: Optional[int] = None,
        plot_script_path: Optional[str] = None,
        auto_plot: bool = True,
        plot_at_end_of_loop: bool = True,
        plots_dir: Optional[str] = None,
    ):
        super().__init__()
        self.plot_interval = plot_interval
        self.auto_plot = auto_plot
        self.plot_at_end_of_loop = plot_at_end_of_loop
        self.last_plot_iter = 0
        self.plots_dir = Path(plots_dir) if plots_dir else None

        if plot_script_path is None:
            plot_script_path = Path(__file__).parents[2] / "tools" / "plot_ray_results.py"
        self.plot_script_path = Path(plot_script_path)

        if not self.plot_script_path.exists():
            print(f"[AutoPlotter] Warning: plot script not found at {self.plot_script_path}")
            self.auto_plot = False

    def on_train_result(self, *, algorithm, result: Dict[str, Any], **kwargs):
        if not self.auto_plot:
            return
        current_iter = int(result.get("training_iteration", 0))

        doit = self.plot_at_end_of_loop or (
            self.plot_interval is not None and current_iter - self.last_plot_iter >= self.plot_interval
        )
        if not doit:
            return

        trial_dir = self._find_trial_directory(algorithm, result)
        if trial_dir is None:
            print("[AutoPlotter] Warning: could not locate trial directory (progress.csv).")
            return

        cmd = [sys.executable, str(self.plot_script_path), str(trial_dir)]
        if self.plots_dir:
            cmd += ["--out", str(self.plots_dir)]

        print(f"\n🎨 Generating plots for iteration {current_iter}")
        print(f"📁 Trial directory: {trial_dir}")

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            print("✅ Plots generated.")
            if proc.stdout:
                print(proc.stdout.strip())
        else:
            print(f"❌ Plot generation failed ({proc.returncode})")
            if proc.stderr:
                print(proc.stderr.strip())

        self.last_plot_iter = current_iter

    def _find_trial_directory(self, algorithm, result: Dict[str, Any]) -> Optional[Path]:
        # 1) direct from result (Ray AIR)
        for k in ("log_dir", "logdir", "trial_logdir"):
            if k in result:
                p = Path(result[k])
                if (p / "progress.csv").exists():
                    return p

        # 2) algorithm.logdir (RLlib)
        p = Path(getattr(algorithm, "logdir", ""))
        if p and (p / "progress.csv").exists():
            return p

        # 3) search recent Phase_A in ray_results
        root = Path.home() / "ray_results"
        if root.exists():
            cands = sorted(root.rglob("progress.csv"), key=lambda pp: pp.stat().st_mtime, reverse=True)
            for cand in cands:
                # prefer our Phase_A/PPO run if present
                if "Phase_A" in str(cand) or "PPO_ITraderEnv" in str(cand):
                    return cand.parent
            if cands:
                return cands[0].parent
        return None
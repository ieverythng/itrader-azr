from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from ray.rllib.algorithms.callbacks import DefaultCallbacks


class AutoPlotterCallback(DefaultCallbacks):
    """
    Callback that automatically generates plots during training.
    Integrates with the plot_ray_results.py script to create visualizations
    at the end of every training loop or at regular intervals during training.
    """
    
    def __init__(self, 
                 plot_interval: Optional[int] = None,
                 plot_script_path: Optional[str] = None,
                 auto_plot: bool = True,
                 plot_at_end_of_loop: bool = True):
        """
        Args:
            plot_interval: Generate plots every N training iterations (ignored if plot_at_end_of_loop=True)
            plot_script_path: Path to plot_ray_results.py script (auto-detected if None)
            auto_plot: Whether to automatically generate plots
            plot_at_end_of_loop: Whether to generate plots at the end of every training loop
        """
        super().__init__()
        self.plot_interval = plot_interval
        self.auto_plot = auto_plot
        self.plot_at_end_of_loop = plot_at_end_of_loop
        self.last_plot_iter = 0
        
        # Auto-detect plot script path
        if plot_script_path is None:
            # Find plot_ray_results.py relative to this file
            current_dir = Path(__file__).parent
            plot_script_path = current_dir.parents[1] / "tools" / "plot_ray_results.py"
        
        self.plot_script_path = Path(plot_script_path)
        
        if not self.plot_script_path.exists():
            print(f"Warning: Plot script not found at {self.plot_script_path}")
            self.auto_plot = False
    
    def on_train_result(self, *, algorithm, result: Dict[str, Any], **kwargs):
        """Called after each training iteration."""
        if not self.auto_plot:
            return
            
        current_iter = result.get("training_iteration", 0)
        
        # Generate plots at the end of every training loop if enabled
        if self.plot_at_end_of_loop:
            self._generate_plots(algorithm, result)
            self.last_plot_iter = current_iter
        # Otherwise, use the interval-based approach
        elif self.plot_interval is not None and current_iter - self.last_plot_iter >= self.plot_interval:
            self._generate_plots(algorithm, result)
            self.last_plot_iter = current_iter
    
    def _generate_plots(self, algorithm, result: Dict[str, Any]):
        """Generate plots using the plot_ray_results.py script."""
        try:
            # Find the trial directory
            trial_dir = self._find_trial_directory(algorithm)
            if not trial_dir:
                print("Warning: Could not find trial directory for plotting")
                return
            
            # Run the plotting script
            cmd = [
                sys.executable,
                str(self.plot_script_path),
                str(trial_dir)
            ]
            
            print(f"\n🎨 Generating plots for iteration {result.get('training_iteration', 0)}...")
            print(f"📁 Trial directory: {trial_dir}")
            
            # Run the plotting script
            result_proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=Path.cwd()
            )
            
            if result_proc.returncode == 0:
                print("✅ Plots generated successfully!")
                if result_proc.stdout:
                    print(result_proc.stdout)
            else:
                print(f"❌ Plot generation failed with return code {result_proc.returncode}")
                if result_proc.stderr:
                    print(f"Error: {result_proc.stderr}")
                    
        except Exception as e:
            print(f"❌ Error generating plots: {e}")
    
    def _find_trial_directory(self, algorithm) -> Optional[Path]:
        """Find the Ray Tune trial directory for the current training run."""
        try:
            # Try to get the trial directory from the algorithm's config
            if hasattr(algorithm, 'config') and 'local_dir' in algorithm.config:
                local_dir = Path(algorithm.config['local_dir'])
                
                # Look for the most recent trial directory
                if local_dir.exists():
                    # Find directories matching the pattern
                    trial_dirs = [d for d in local_dir.iterdir() 
                                if d.is_dir() and 'PPO_ITraderEnv' in d.name]
                    
                    if trial_dirs:
                        # Return the most recently modified one
                        return max(trial_dirs, key=lambda x: x.stat().st_mtime)
            
            # Fallback: look in default Ray results directory
            default_dir = Path.home() / "ray_results"
            if default_dir.exists():
                trial_dirs = [d for d in default_dir.iterdir() 
                            if d.is_dir() and 'PPO_ITraderEnv' in d.name]
                
                if trial_dirs:
                    return max(trial_dirs, key=lambda x: x.stat().st_mtime)
                    
        except Exception as e:
            print(f"Warning: Could not find trial directory: {e}")
        
        return None


class TrainingStartPlotter:
    """
    Utility class to generate initial plots when training starts.
    Can be used independently of the callback system.
    """
    
    @staticmethod
    def generate_initial_plots(trial_dir: str, plot_script_path: Optional[str] = None):
        """Generate initial plots for a trial directory."""
        if plot_script_path is None:
            plot_script_path = Path(__file__).parent.parents[1] / "tools" / "plot_ray_results.py"
        
        plot_script_path = Path(plot_script_path)
        
        if not plot_script_path.exists():
            print(f"Warning: Plot script not found at {plot_script_path}")
            return False
        
        try:
            cmd = [sys.executable, str(plot_script_path), trial_dir]
            
            print(f"\n🎨 Generating initial plots...")
            print(f"📁 Trial directory: {trial_dir}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
            
            if result.returncode == 0:
                print("✅ Initial plots generated successfully!")
                if result.stdout:
                    print(result.stdout)
                return True
            else:
                print(f"❌ Initial plot generation failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error generating initial plots: {e}")
            return False

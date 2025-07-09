# -----------------------------------------------------------
#  Phase-A training script – solver-only PPO
# -----------------------------------------------------------
import argparse, pathlib, numpy as np, torch
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog

# --- local imports
from itrader.envs.itrader_env import ITraderEnv           # your custom Gymnasium env
from itrader.models.solver_lstm import SolverLSTM
from itrader.models.solver_ray import SolverRay

# ----------------------------------------------------------- #
def build_config(model_name: str, window_size: int = 32):
    """
    Return an RLlib PPOConfig pre-populated for Phase A
    """
    cfg = (
        PPOConfig()
        .environment(
            env=ITraderEnv,
            env_config={
                # << path to your parquet >> 
                "df_path": str(pathlib.Path(__file__).parent.parent / "data" / "ohlcv.parquet"),
                "window_size": window_size,
            },
        )
        .framework("torch")
        .resources(num_gpus=int(torch.cuda.is_available()))
        .rollouts(num_rollout_workers=0)                     # local sampling
        .training(
            gamma=0.99,
            lr=5e-4,
            train_batch_size=4000,
            sgd_minibatch_size=128,
            model={
                "custom_model": model_name,
                "window_size": window_size,                 # passed into model_config
            },
        )
    )
    return cfg


# ----------------------------------------------------------- #
def main(args):
    # --- register models once per process
    ModelCatalog.register_custom_model("solver_lstm", SolverLSTM)
    ModelCatalog.register_custom_model("solver_ray",  SolverRay)

    # --- choose model
    model_name = "solver_lstm" if args.lstm else "solver_ray"
    config     = build_config(model_name, window_size=args.window)

    algo = config.build()

    for i in range(args.iters):
        result = algo.train()
        if i % args.log == 0:
            print(
                f"[{i:04d}]  "
                f"reward_mean={result['episode_reward_mean']:.3f}  "
                f"policy_loss={result['learner_results']['default_policy']['policy_loss']:.4f}"
            )

    algo.stop()


# ----------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=100, help="train iterations")
    parser.add_argument("--window", type=int, default=32, help="price window size")
    parser.add_argument("--log", type=int, default=10, help="stdout frequency")
    parser.add_argument("--lstm", action="store_true", help="use LSTM (else MLP)")
    main(parser.parse_args())

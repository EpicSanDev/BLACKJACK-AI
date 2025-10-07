from __future__ import annotations

import argparse
from pathlib import Path

from blackjack.rl_policy import ACTION_LABELS, save_policy, train_dqn, train_q_learning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an advanced Blackjack advisor policy (DQN or tabular).")
    parser.add_argument("--algo", choices=["dqn", "tabular"], default="dqn", help="Training algorithm (dqn or tabular Q-learning)")
    parser.add_argument("--episodes", type=int, default=600000, help="Training episodes to run")
    parser.add_argument("--alpha", type=float, default=0.05, help="Tabular Q-learning rate")
    parser.add_argument("--gamma", type=float, default=0.995, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial exploration epsilon")
    parser.add_argument("--epsilon-min", dest="epsilon_min", type=float, default=0.05, help="Minimum exploration epsilon")
    parser.add_argument("--epsilon-decay", dest="epsilon_decay", type=float, default=0.999, help="Multiplicative epsilon decay")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for DQN optimizer")
    parser.add_argument("--batch-size", type=int, default=512, help="DQN mini-batch size")
    parser.add_argument("--replay-size", type=int, default=200_000, help="Replay buffer capacity")
    parser.add_argument("--warmup", type=int, default=5_000, help="Steps before starting gradient updates")
    parser.add_argument(
        "--target-sync",
        type=int,
        default=1_000,
        help="How often to sync DQN target network (in gradient steps)",
    )
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[256, 256, 128],
        help="Hidden layer sizes for the DQN model",
    )
    parser.add_argument(
        "--parallel-envs",
        type=int,
        default=1,
        help="Number of blackjack environments to run in parallel for DQN training",
    )
    parser.add_argument("--dealer-hits-soft-17", action="store_true", help="Train assuming the dealer hits on soft 17")
    parser.add_argument("--disable-surrender", action="store_true", help="Disable surrender during training")
    parser.add_argument("--disable-double", action="store_true", help="Disable doubling down during training")
    parser.add_argument("--device", type=str, default=None, help="Torch device to run DQN on (default: auto)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model/advanced_policy.json"),
        help="Where to store the trained policy (JSON)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    surrender_allowed = not args.disable_surrender
    double_allowed = not args.disable_double

    if args.algo == "tabular":
        q_table, visits = train_q_learning(
            episodes=args.episodes,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon,
            epsilon_min=args.epsilon_min,
            epsilon_decay=args.epsilon_decay,
            dealer_hits_soft_17=args.dealer_hits_soft_17,
        )
    else:
        q_table, visits = train_dqn(
            episodes=args.episodes,
            gamma=args.gamma,
            lr=args.lr,
            epsilon_start=args.epsilon,
            epsilon_end=args.epsilon_min,
            epsilon_decay=args.epsilon_decay,
            batch_size=args.batch_size,
            replay_size=args.replay_size,
            warmup=args.warmup,
            target_sync_interval=args.target_sync,
            hidden_sizes=tuple(args.hidden_sizes),
            parallel_envs=args.parallel_envs,
            dealer_hits_soft_17=args.dealer_hits_soft_17,
            surrender_allowed=surrender_allowed,
            double_allowed=double_allowed,
            seed=args.seed,
            device=args.device,
        )

    meta = {
        "algorithm": args.algo,
        "episodes": args.episodes,
        "gamma": args.gamma,
        "epsilon_start": args.epsilon,
        "epsilon_min": args.epsilon_min,
        "epsilon_decay": args.epsilon_decay,
        "dealer_hits_soft_17": args.dealer_hits_soft_17,
        "actions": ACTION_LABELS,
        "surrender_allowed": surrender_allowed,
        "double_allowed": double_allowed,
    }

    if args.algo == "tabular":
        meta["alpha"] = args.alpha
    else:
        meta.update(
            {
                "lr": args.lr,
                "batch_size": args.batch_size,
                "replay_size": args.replay_size,
                "warmup": args.warmup,
                "target_sync_interval": args.target_sync,
                "hidden_sizes": args.hidden_sizes,
                "parallel_envs": args.parallel_envs,
                "device": args.device or "auto",
                "seed": args.seed,
            }
        )

    save_policy(
        str(output_path),
        q_table,
        visits,
        meta=meta,
        rules={
            "surrender_allowed": surrender_allowed,
            "double_allowed": double_allowed,
            "dealer_hits_on_soft_17": args.dealer_hits_soft_17,
        },
    )

    print(f"Policy saved to {output_path}")


if __name__ == "__main__":
    main()

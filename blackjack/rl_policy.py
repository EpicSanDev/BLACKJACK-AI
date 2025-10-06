"""Utilities for training a learned Blackjack advisor policy."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .strategy import CARD_VALUES, DEFAULT_GAME_RULES, evaluate_hand, normalise_hand


# Action ordering is important because it is used both during training and
# inference when the policy is loaded back.
ACTION_LABELS = ["Stand", "Hit", "Double", "Surrender"]
ACTION_INDEX = {name: idx for idx, name in enumerate(ACTION_LABELS)}
ACTION_STAND = ACTION_INDEX["Stand"]
ACTION_HIT = ACTION_INDEX["Hit"]
ACTION_DOUBLE = ACTION_INDEX["Double"]
ACTION_SURRENDER = ACTION_INDEX["Surrender"]


def _build_card_pool() -> List[Tuple[str, int]]:
    """Return a 52-card pool represented as (rank, value) tuples."""

    pool: List[Tuple[str, int]] = []
    for rank, value in CARD_VALUES.items():
        pool.extend([(rank, int(value))] * 4)
    return pool


CARD_POOL: Tuple[Tuple[str, int], ...] = tuple(_build_card_pool())


def _draw_card() -> Dict[str, int | str]:
    rank, value = random.choice(CARD_POOL)
    return {"rank": rank, "value": value}


def _dealer_value(card: Dict[str, int | str]) -> int:
    value = int(card["value"])
    return 11 if value == 11 else value


def available_actions(
    state: BlackjackState,
    *,
    surrender_allowed: bool = DEFAULT_GAME_RULES["surrender_allowed"],
    double_allowed: bool = True,
) -> List[int]:
    actions = [ACTION_STAND, ACTION_HIT]
    if state.first_move and double_allowed:
        actions.append(ACTION_DOUBLE)
    if state.first_move and surrender_allowed:
        actions.append(ACTION_SURRENDER)
    return actions


def _actions_to_mask(actions: Sequence[int]) -> Tuple[bool, ...]:
    mask = [False] * len(ACTION_LABELS)
    for idx in actions:
        mask[idx] = True
    return tuple(mask)


@dataclass(frozen=True)
class BlackjackState:
    player_total: int
    usable_ace: bool
    dealer_up_card: int
    first_move: bool
    is_pair: bool

    def key(self) -> str:
        return (
            f"{self.player_total}|{int(self.usable_ace)}|"
            f"{self.dealer_up_card}|{int(self.first_move)}|{int(self.is_pair)}"
        )


def state_from_key(key: str) -> BlackjackState:
    total, usable, dealer, first_move, is_pair = key.split("|")
    return BlackjackState(
        int(total),
        bool(int(usable)),
        int(dealer),
        bool(int(first_move)),
        bool(int(is_pair)),
    )


def state_feature_vector(state: BlackjackState) -> Tuple[float, ...]:
    total = float(state.player_total)
    total_scaled = total / 21.0
    total_centered = (total - 15.0) / 10.0
    soft = 1.0 if state.usable_ace else 0.0
    dealer_scaled = (float(state.dealer_up_card) - 2.0) / 9.0
    first_move = 1.0 if state.first_move else 0.0
    pair = 1.0 if state.is_pair else 0.0
    blackjack = 1.0 if state.first_move and state.player_total == 21 else 0.0
    return (
        total_scaled,
        total_centered,
        soft,
        dealer_scaled,
        first_move,
        pair,
        blackjack,
    )


FEATURE_DIM = len(state_feature_vector(BlackjackState(12, False, 6, True, False)))
ZERO_FEATURES: Tuple[float, ...] = tuple(0.0 for _ in range(FEATURE_DIM))
ZERO_MASK: Tuple[bool, ...] = tuple(False for _ in ACTION_LABELS)


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: deque[Tuple[Tuple[float, ...], int, float, Tuple[float, ...], bool, Tuple[bool, ...]]] = deque(
            maxlen=capacity
        )

    def push(
        self,
        state: Tuple[float, ...],
        action: int,
        reward: float,
        next_state: Tuple[float, ...],
        done: bool,
        next_mask: Tuple[bool, ...],
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done, next_mask))

    def sample(self, batch_size: int) -> Dict[str, List[object]]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, next_masks = zip(*batch)
        return {
            "states": list(states),
            "actions": list(actions),
            "rewards": list(rewards),
            "next_states": list(next_states),
            "dones": list(dones),
            "next_masks": list(next_masks),
        }

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.buffer)


def _enumerate_reachable_states(
    surrender_allowed: bool,
    double_allowed: bool,
) -> List[BlackjackState]:
    states: List[BlackjackState] = []
    for total in range(4, 22):
        for usable in (False, True):
            for dealer in range(2, 12):
                for first_move in (True, False):
                    pair_options = (False, True) if first_move else (False,)
                    for is_pair in pair_options:
                        state = BlackjackState(total, usable, dealer, first_move, is_pair)
                        valid = available_actions(
                            state,
                            surrender_allowed=surrender_allowed,
                            double_allowed=double_allowed,
                        )
                        if valid:
                            states.append(state)
    return states


class BlackjackEnv:
    """Simple Blackjack simulator tuned for policy learning."""

    def __init__(
        self,
        dealer_hits_soft_17: bool = DEFAULT_GAME_RULES["dealer_hits_on_soft_17"],
        surrender_allowed: bool = DEFAULT_GAME_RULES["surrender_allowed"],
        double_allowed: bool = True,
    ) -> None:
        self.dealer_hits_soft_17 = bool(dealer_hits_soft_17)
        self.surrender_allowed = bool(surrender_allowed)
        self.double_allowed = bool(double_allowed)
        self.player_cards: List[Dict[str, int | str]] = []
        self.dealer_cards: List[Dict[str, int | str]] = []
        self._first_move = True
        self.done = False

    def reset(self) -> BlackjackState:
        self.done = False
        self._first_move = True
        self.player_cards = [_draw_card(), _draw_card()]
        self.dealer_cards = [_draw_card(), _draw_card()]
        return self._state()

    def step(self, action: int) -> Tuple[Optional[BlackjackState], float, bool, Dict[str, object]]:
        if self.done:
            raise RuntimeError("Environment already finished. Call reset().")

        info: Dict[str, object] = {}
        reward = 0.0

        if action == ACTION_SURRENDER:
            if self._first_move and self.surrender_allowed:
                self.done = True
                return None, -0.5, True, info
            info["invalid_action"] = "surrender_not_allowed"
            action = ACTION_HIT

        if action == ACTION_DOUBLE and (not self._first_move or not self.double_allowed):
            info["invalid_action"] = "double_not_allowed"
            action = ACTION_HIT

        if action == ACTION_STAND:
            reward = self._resolve_round(double=False)
            self.done = True
            return None, reward, True, info

        if action == ACTION_DOUBLE:
            self.player_cards.append(_draw_card())
            reward = self._resolve_round(double=True)
            self.done = True
            return None, reward, True, info

        if action == ACTION_HIT:
            self.player_cards.append(_draw_card())
            total, _ = evaluate_hand(self.player_cards)
            self._first_move = False
            if total > 21:
                self.done = True
                return None, -1.0, True, info
            return self._state(), 0.0, False, info

        raise ValueError(f"Unknown action id {action}")

    def _state(self) -> BlackjackState:
        total, usable_ace = evaluate_hand(self.player_cards)
        dealer_up = _dealer_value(self.dealer_cards[0])
        is_pair = False
        if self._first_move and len(self.player_cards) == 2:
            first, second = normalise_hand(self.player_cards)
            is_pair = first["rank"] == second["rank"]
        return BlackjackState(total, usable_ace, dealer_up, self._first_move, is_pair)

    def _resolve_round(self, double: bool) -> float:
        player_total, _ = evaluate_hand(self.player_cards)
        if player_total > 21:
            return -2.0 if double else -1.0

        dealer_total, dealer_soft = evaluate_hand(self.dealer_cards)
        while dealer_total < 17 or (dealer_total == 17 and dealer_soft and self.dealer_hits_soft_17):
            self.dealer_cards.append(_draw_card())
            dealer_total, dealer_soft = evaluate_hand(self.dealer_cards)

        if dealer_total > 21:
            base = 1.0
        elif player_total > dealer_total:
            base = 1.0
        elif player_total < dealer_total:
            base = -1.0
        else:
            base = 0.0

        return base * 2.0 if double else base


def train_q_learning(
    episodes: int,
    alpha: float = 0.05,
    gamma: float = 0.99,
    epsilon: float = 1.0,
    epsilon_min: float = 0.1,
    epsilon_decay: float = 0.9995,
    dealer_hits_soft_17: bool = DEFAULT_GAME_RULES["dealer_hits_on_soft_17"],
) -> Tuple[Dict[str, List[float]], Dict[str, int]]:
    """Train a Q-learning policy for Blackjack.

    Returns:
        Tuple containing a Q-table keyed by state string and a visit counter per state.
    """

    env = BlackjackEnv(dealer_hits_soft_17=dealer_hits_soft_17)
    q_table: Dict[str, List[float]] = defaultdict(lambda: [0.0] * len(ACTION_LABELS))
    visits: Dict[str, int] = defaultdict(int)

    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False

        while not done:
            state_key = state.key()
            visits[state_key] += 1
            valid_actions = available_actions(
                state,
                surrender_allowed=env.surrender_allowed,
                double_allowed=env.double_allowed,
            )

            if random.random() < epsilon:
                action = random.choice(valid_actions)
            else:
                q_values = q_table[state_key]
                max_q = max(q_values[i] for i in valid_actions)
                best_actions = [i for i in valid_actions if math.isclose(q_values[i], max_q, rel_tol=1e-6)]
                action = random.choice(best_actions)

            next_state, reward, done, _ = env.step(action)

            q_values = q_table[state_key]
            if done:
                target = reward
            else:
                next_key = next_state.key()
                next_valid_actions = available_actions(
                    next_state,
                    surrender_allowed=env.surrender_allowed,
                    double_allowed=env.double_allowed,
                )
                next_best = max(q_table[next_key][i] for i in next_valid_actions)
                target = reward + gamma * next_best
            q_values[action] += alpha * (target - q_values[action])

            if next_state is not None:
                state = next_state

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return q_table, visits


def train_dqn(
    episodes: int,
    *,
    gamma: float = 0.995,
    lr: float = 1e-3,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.999,
    batch_size: int = 512,
    replay_size: int = 200_000,
    warmup: int = 5_000,
    target_sync_interval: int = 1_000,
    hidden_sizes: Sequence[int] = (256, 256, 128),
    dealer_hits_soft_17: bool = DEFAULT_GAME_RULES["dealer_hits_on_soft_17"],
    surrender_allowed: bool = DEFAULT_GAME_RULES["surrender_allowed"],
    double_allowed: bool = True,
    seed: Optional[int] = None,
    device: Optional[str] = None,
) -> Tuple[Dict[str, List[float]], Dict[str, int]]:
    """Train a Deep Q-Network policy for Blackjack."""

    try:
        import torch
        from torch import nn
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError("PyTorch is required for DQN training.") from exc

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    class PolicyNetwork(nn.Module):
        def __init__(self, input_dim: int, hidden_layers: Sequence[int]) -> None:
            super().__init__()
            layers: List[nn.Module] = []
            last_dim = input_dim
            for width in hidden_layers:
                layers.append(nn.Linear(last_dim, width))
                layers.append(nn.ReLU())
                last_dim = width
            layers.append(nn.Linear(last_dim, len(ACTION_LABELS)))
            self.model = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple delegation
            return self.model(x)

    env = BlackjackEnv(
        dealer_hits_soft_17=dealer_hits_soft_17,
        surrender_allowed=surrender_allowed,
        double_allowed=double_allowed,
    )
    policy_net = PolicyNetwork(FEATURE_DIM, hidden_sizes).to(torch_device)
    target_net = PolicyNetwork(FEATURE_DIM, hidden_sizes).to(torch_device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()
    buffer = ReplayBuffer(replay_size)
    visits: Dict[str, int] = defaultdict(int)

    epsilon = epsilon_start
    step_count = 0

    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False

        while not done:
            state_key = state.key()
            visits[state_key] += 1
            features = state_feature_vector(state)
            valid_actions = available_actions(
                state,
                surrender_allowed=env.surrender_allowed,
                double_allowed=env.double_allowed,
            )

            if random.random() < epsilon:
                action = random.choice(valid_actions)
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor([features], dtype=torch.float32, device=torch_device)
                    q_values = policy_net(state_tensor).squeeze(0)
                    mask = torch.tensor(
                        _actions_to_mask(valid_actions), dtype=torch.bool, device=torch_device
                    )
                    q_values_masked = q_values.masked_fill(~mask, -1e9)
                    action = int(torch.argmax(q_values_masked).item())

            next_state, reward, done, _ = env.step(action)
            if next_state is None:
                next_features = ZERO_FEATURES
                next_mask = ZERO_MASK
            else:
                next_features = state_feature_vector(next_state)
                next_valid_actions = available_actions(
                    next_state,
                    surrender_allowed=env.surrender_allowed,
                    double_allowed=env.double_allowed,
                )
                next_mask = _actions_to_mask(next_valid_actions)

            buffer.push(features, action, float(reward), next_features, done, next_mask)

            if len(buffer) >= max(batch_size, warmup):
                sample_size = min(batch_size, len(buffer))
                batch = buffer.sample(sample_size)

                states_tensor = torch.tensor(batch["states"], dtype=torch.float32, device=torch_device)
                actions_tensor = (
                    torch.tensor(batch["actions"], dtype=torch.int64, device=torch_device).unsqueeze(1)
                )
                rewards_tensor = torch.tensor(batch["rewards"], dtype=torch.float32, device=torch_device)
                next_states_tensor = torch.tensor(batch["next_states"], dtype=torch.float32, device=torch_device)
                dones_tensor = torch.tensor(batch["dones"], dtype=torch.bool, device=torch_device)
                next_masks_tensor = torch.tensor(batch["next_masks"], dtype=torch.bool, device=torch_device)

                current_q = policy_net(states_tensor).gather(1, actions_tensor).squeeze(1)

                with torch.no_grad():
                    next_q_values = target_net(next_states_tensor)
                    next_q_values = next_q_values.masked_fill(~next_masks_tensor, -1e9)
                    best_next = torch.max(next_q_values, dim=1).values
                    best_next = torch.where(
                        next_masks_tensor.any(dim=1), best_next, torch.zeros_like(best_next)
                    )
                    target = rewards_tensor + gamma * (~dones_tensor).float() * best_next

                loss = loss_fn(current_q, target)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 5.0)
                optimizer.step()

            step_count += 1
            if step_count % target_sync_interval == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if next_state is not None:
                state = next_state

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    target_net.load_state_dict(policy_net.state_dict())
    policy_net.eval()

    q_table: Dict[str, List[float]] = {}
    for state in _enumerate_reachable_states(env.surrender_allowed, env.double_allowed):
        feats = state_feature_vector(state)
        state_tensor = torch.tensor([feats], dtype=torch.float32, device=torch_device)
        with torch.no_grad():
            q_values = policy_net(state_tensor).squeeze(0).cpu().tolist()
        q_table[state.key()] = q_values

    return q_table, visits


def save_policy(
    path: str,
    q_table: Dict[str, Iterable[float]],
    visits: Dict[str, int],
    meta: Optional[Dict[str, object]] = None,
    rules: Optional[Mapping[str, object]] = None,
) -> None:
    """Persist the learned policy and metadata to disk."""

    rules_dict = dict(DEFAULT_GAME_RULES)
    double_allowed = True
    if rules:
        for key, value in rules.items():
            if key in rules_dict:
                rules_dict[key] = bool(value)
        if "double_allowed" in rules:
            double_allowed = bool(rules["double_allowed"])

    surrender_allowed = rules_dict["surrender_allowed"]

    policy: Dict[str, str] = {}
    for state_key, q_values in q_table.items():
        state = state_from_key(state_key)
        valid_actions = available_actions(
            state,
            surrender_allowed=surrender_allowed,
            double_allowed=double_allowed,
        )
        if not valid_actions:
            continue
        best_index = max(valid_actions, key=lambda i: q_values[i])
        policy[state_key] = ACTION_LABELS[best_index]

    resolved_meta = dict(meta) if meta else {}
    resolved_meta.setdefault(
        "rules",
        {
            "surrender_allowed": surrender_allowed,
            "double_allowed": double_allowed,
            "dealer_hits_on_soft_17": rules_dict["dealer_hits_on_soft_17"],
        },
    )
    resolved_meta["action_order"] = ACTION_LABELS

    output = {
        "meta": resolved_meta,
        "policy": policy,
        "q_values": {state: list(values) for state, values in q_table.items()},
        "visits": visits,
    }

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, sort_keys=True)


def load_policy(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data["policy"]

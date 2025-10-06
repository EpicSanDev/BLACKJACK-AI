"""Runtime helper to load and query the learned Blackjack policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

from .strategy import DEFAULT_GAME_RULES, evaluate_hand, get_expert_advice, normalise_hand


class AdvancedAdvisor:
    def __init__(self, policy: Dict[str, str], rules: Optional[Mapping[str, object]] = None) -> None:
        self.policy = dict(policy)
        base_rules = dict(DEFAULT_GAME_RULES)
        allow_double = True
        if rules:
            for key, value in rules.items():
                if key in base_rules:
                    base_rules[key] = bool(value)
                if key == "double_allowed":
                    allow_double = bool(value)
        self.rules = base_rules
        self.allow_double = allow_double

    @classmethod
    def from_file(cls, path: str | Path, rules: Optional[Mapping[str, object]] = None) -> "AdvancedAdvisor":
        with open(Path(path), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        policy = data.get("policy")
        if not isinstance(policy, dict):
            raise ValueError("Invalid policy file: 'policy' section missing.")
        if rules is None:
            meta_rules = data.get("meta", {}).get("rules")
            if isinstance(meta_rules, Mapping):
                rules = meta_rules
        return cls(policy, rules)

    def recommend(self, player_cards: Sequence[Mapping[str, object]], dealer_card: Mapping[str, object]) -> str:
        if not player_cards or not dealer_card:
            return "Hit"

        normalised_player = normalise_hand(player_cards)
        normalised_dealer = normalise_hand([dealer_card])[0]

        total, usable_ace = evaluate_hand(normalised_player)
        dealer_value = int(normalised_dealer["value"])

        first_move = len(normalised_player) == 2
        is_pair = False
        if first_move and len(normalised_player) == 2:
            is_pair = normalised_player[0]["rank"] == normalised_player[1]["rank"]
        state_key = (
            f"{total}|{int(usable_ace)}|{dealer_value}|{int(first_move)}|{int(is_pair)}"
        )

        action = self.policy.get(state_key)

        if action == "Double" and (not first_move or not self.allow_double):
            action = None
        if action == "Surrender" and (not first_move or not self.rules.get("surrender_allowed", False)):
            action = None

        if action is None:
            action = get_expert_advice(normalised_player, normalised_dealer, self.rules)

        return action

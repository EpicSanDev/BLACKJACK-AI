from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

CARD_VALUES: Dict[str, int] = {
    "ace": 11,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "jack": 10,
    "queen": 10,
    "king": 10,
}

DEFAULT_GAME_RULES: Dict[str, bool] = {
    "dealer_hits_on_soft_17": False,
    "double_after_split_allowed": True,
    "surrender_allowed": True,
}

# Strategy tables adapted from standard Blackjack strategy charts
STRATEGY_HARD = {
    17: {2: "S", 3: "S", 4: "S", 5: "S", 6: "S", 7: "S", 8: "S", 9: "S", 10: "S", 11: "S"},
    16: {2: "S", 3: "S", 4: "S", 5: "S", 6: "S", 7: "H", 8: "H", 9: "Rh", 10: "Rh", 11: "Rh"},
    15: {2: "S", 3: "S", 4: "S", 5: "S", 6: "S", 7: "H", 8: "H", 9: "H", 10: "Rh", 11: "H"},
    14: {2: "S", 3: "S", 4: "S", 5: "S", 6: "S", 7: "H", 8: "H", 9: "H", 10: "H", 11: "H"},
    13: {2: "S", 3: "S", 4: "S", 5: "S", 6: "S", 7: "H", 8: "H", 9: "H", 10: "H", 11: "H"},
    12: {2: "H", 3: "H", 4: "S", 5: "S", 6: "S", 7: "H", 8: "H", 9: "H", 10: "H", 11: "H"},
    11: {2: "D", 3: "D", 4: "D", 5: "D", 6: "D", 7: "D", 8: "D", 9: "D", 10: "D", 11: "H"},
    10: {2: "D", 3: "D", 4: "D", 5: "D", 6: "D", 7: "D", 8: "D", 9: "D", 10: "H", 11: "H"},
    9: {2: "H", 3: "D", 4: "D", 5: "D", 6: "D", 7: "H", 8: "H", 9: "H", 10: "H", 11: "H"},
    8: {2: "H", 3: "H", 4: "H", 5: "H", 6: "H", 7: "H", 8: "H", 9: "H", 10: "H", 11: "H"},
}

STRATEGY_SOFT = {
    20: {2: "S", 3: "S", 4: "S", 5: "S", 6: "S", 7: "S", 8: "S", 9: "S", 10: "S", 11: "S"},
    19: {2: "S", 3: "S", 4: "S", 5: "S", 6: "S", 7: "S", 8: "S", 9: "S", 10: "S", 11: "S"},
    18: {2: "S", 3: "S", 4: "S", 5: "S", 6: "S", 7: "S", 8: "S", 9: "H", 10: "H", 11: "H"},
    17: {2: "H", 3: "D", 4: "D", 5: "D", 6: "D", 7: "H", 8: "H", 9: "H", 10: "H", 11: "H"},
    16: {2: "H", 3: "H", 4: "D", 5: "D", 6: "D", 7: "H", 8: "H", 9: "H", 10: "H", 11: "H"},
    15: {2: "H", 3: "H", 4: "D", 5: "D", 6: "D", 7: "H", 8: "H", 9: "H", 10: "H", 11: "H"},
    14: {2: "H", 3: "H", 4: "H", 5: "D", 6: "D", 7: "H", 8: "H", 9: "H", 10: "H", 11: "H"},
    13: {2: "H", 3: "H", 4: "H", 5: "D", 6: "D", 7: "H", 8: "H", 9: "H", 10: "H", 11: "H"},
}

STRATEGY_PAIRS = {
    11: {2: "P", 3: "P", 4: "P", 5: "P", 6: "P", 7: "P", 8: "P", 9: "P", 10: "P", 11: "P"},
    10: {2: "S", 3: "S", 4: "S", 5: "S", 6: "S", 7: "S", 8: "S", 9: "S", 10: "S", 11: "S"},
    9: {2: "P", 3: "P", 4: "P", 5: "P", 6: "P", 7: "S", 8: "P", 9: "P", 10: "S", 11: "S"},
    8: {2: "P", 3: "P", 4: "P", 5: "P", 6: "P", 7: "P", 8: "P", 9: "P", 10: "P", 11: "P"},
    7: {2: "P", 3: "P", 4: "P", 5: "P", 6: "P", 7: "P", 8: "H", 9: "H", 10: "H", 11: "H"},
    6: {2: "P", 3: "P", 4: "P", 5: "P", 6: "P", 7: "H", 8: "H", 9: "H", 10: "H", 11: "H"},
    5: {2: "D", 3: "D", 4: "D", 5: "D", 6: "D", 7: "D", 8: "D", 9: "D", 10: "H", 11: "H"},
    4: {2: "H", 3: "H", 4: "H", 5: "P", 6: "P", 7: "H", 8: "H", 9: "H", 10: "H", 11: "H"},
    3: {2: "P", 3: "P", 4: "P", 5: "P", 6: "P", 7: "P", 8: "H", 9: "H", 10: "H", 11: "H"},
    2: {2: "P", 3: "P", 4: "P", 5: "P", 6: "P", 7: "P", 8: "H", 9: "H", 10: "H", 11: "H"},
}

ACTION_LOOKUP = {"H": "Hit", "S": "Stand", "D": "Double", "P": "Split"}
ACTION_LABELS = {
    "Hit": "Hit",
    "Stand": "Stand",
    "Double": "Double",
    "Split": "Split",
    "Surrender": "Surrender",
}
ACTION_LABELS_FR = {
    "Hit": "Tirer",
    "Stand": "Rester",
    "Double": "Doubler",
    "Split": "Séparer",
    "Surrender": "Abandonner",
}


def _normalise_card(card: Mapping[str, object]) -> Dict[str, object]:
    rank = str(card["rank"]).lower()
    value = card.get("value")
    if value is None:
        value = CARD_VALUES.get(rank)
    if value is None:
        raise KeyError(f"Unknown card rank: {rank}")
    return {"rank": rank, "value": int(value)}


def normalise_hand(cards: Iterable[Mapping[str, object]]) -> List[Dict[str, object]]:
    return [_normalise_card(card) for card in cards]


def _hand_total(hand: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    normalised = normalise_hand(hand)
    total = sum(int(card["value"]) for card in normalised)
    ace_count = sum(1 for card in normalised if card["rank"] == "ace")
    soft_aces = ace_count
    while total > 21 and soft_aces > 0:
        total -= 10
        soft_aces -= 1
    is_soft = soft_aces > 0
    return {"total": total, "is_soft": is_soft, "hand": normalised}


def evaluate_hand(hand: Sequence[Mapping[str, object]]) -> Tuple[int, bool]:
    state = _hand_total(hand)
    return int(state["total"]), bool(state["is_soft"])


def _is_pair(hand: Sequence[Mapping[str, object]]) -> bool:
    if len(hand) != 2:
        return False
    first, second = hand
    return first["rank"] == second["rank"]


def _pair_value(hand: Sequence[Mapping[str, object]]) -> int:
    return int(hand[0]["value"])


def _dealer_value(card: Mapping[str, object]) -> int:
    value = int(card["value"])
    return 11 if value == 11 else value


def describe_hand(hand: Sequence[Mapping[str, object]]) -> str:
    ranks = [str(card["rank"]) for card in normalise_hand(hand)]
    return " + ".join(ranks) if ranks else "-"


def get_expert_advice(
    player_cards: Sequence[Mapping[str, object]],
    dealer_card: Mapping[str, object],
    rules: Optional[Mapping[str, object]] = None,
) -> str:
    if not player_cards or not dealer_card:
        return "Hit"

    rules_dict: Dict[str, bool] = dict(DEFAULT_GAME_RULES)
    if rules:
        for key, value in rules.items():
            if key in rules_dict:
                rules_dict[key] = bool(value)

    dealer = _normalise_card(dealer_card)
    player_state = _hand_total(player_cards)
    player_hand = player_state["hand"]
    player_total = int(player_state["total"])
    is_soft = bool(player_state["is_soft"])
    pair = _is_pair(player_hand)
    dealer_val = _dealer_value(dealer)

    if pair:
        pair_val = _pair_value(player_hand)
        if pair_val in STRATEGY_PAIRS:
            action = STRATEGY_PAIRS[pair_val][dealer_val]
            if action == "P" and not rules_dict["double_after_split_allowed"] and player_hand[0]["rank"] == "4":
                return "Hit"
            return ACTION_LOOKUP.get(action, "Hit")

    if is_soft and player_total in STRATEGY_SOFT:
        action = STRATEGY_SOFT[player_total][dealer_val]
        return ACTION_LOOKUP.get(action, "Hit")

    if 8 <= player_total <= 17 and player_total in STRATEGY_HARD:
        action = STRATEGY_HARD[player_total][dealer_val]
        if action == "Rh":
            return "Surrender" if rules_dict.get("surrender_allowed", False) else "Hit"
        return ACTION_LOOKUP.get(action, "Hit")

    if player_total >= 18:
        return "Stand"
    return "Hit"

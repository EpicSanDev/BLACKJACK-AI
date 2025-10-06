from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .strategy import (
    ACTION_LABELS_FR,
    CARD_VALUES,
    DEFAULT_GAME_RULES,
    describe_hand,
    evaluate_hand,
    get_expert_advice,
)
from .advanced_advisor import AdvancedAdvisor
from .rl_policy import ACTION_LABELS, ACTION_INDEX, save_policy

SUITS: Tuple[str, ...] = ("clubs", "diamonds", "hearts", "spades")
RANKS: Tuple[str, ...] = tuple(CARD_VALUES.keys())


@dataclass(frozen=True)
class Card:
    rank: str
    suit: str

    def __post_init__(self) -> None:
        if self.rank not in CARD_VALUES:
            raise ValueError(f"Unknown rank: {self.rank}")
        if self.suit not in SUITS:
            raise ValueError(f"Unknown suit: {self.suit}")

    @property
    def value(self) -> int:
        return int(CARD_VALUES[self.rank])

    @property
    def asset_name(self) -> str:
        return f"{self.rank}_of_{self.suit}"

    def as_strategy_card(self) -> Dict[str, object]:
        return {"rank": self.rank, "value": self.value}


@dataclass
class Hand:
    cards: List[Card] = field(default_factory=list)

    def add(self, card: Card) -> None:
        self.cards.append(card)

    def clear(self) -> None:
        self.cards.clear()

    def strategy_cards(self) -> List[Dict[str, object]]:
        return [card.as_strategy_card() for card in self.cards]

    def total(self) -> Tuple[int, bool]:
        return evaluate_hand(self.strategy_cards())

    def is_blackjack(self) -> bool:
        if len(self.cards) != 2:
            return False
        total, _ = self.total()
        return total == 21

    def description(self) -> str:
        return describe_hand(self.strategy_cards())


class Shoe:
    def __init__(self, decks: int = 6) -> None:
        if decks < 1:
            raise ValueError("At least one deck is required")
        self.decks = decks
        self.cards: List[Card] = []
        self._reshuffle()

    def _reshuffle(self) -> None:
        self.cards = [
            Card(rank, suit)
            for _ in range(self.decks)
            for rank in RANKS
            for suit in SUITS
        ]
        random.shuffle(self.cards)

    def draw(self) -> Card:
        if not self.cards:
            self._reshuffle()
        return self.cards.pop()

    def penetration(self) -> float:
        total_cards = 52 * self.decks
        return 1.0 - (len(self.cards) / total_cards)

    def needs_shuffle(self, threshold: float = 0.75) -> bool:
        return self.penetration() >= threshold

    def shuffle_if_needed(self) -> None:
        if self.needs_shuffle():
            self._reshuffle()


STATE_BETTING = "betting"
STATE_PLAYER = "player_turn"
STATE_DEALER = "dealer_turn"
STATE_RESULT = "round_result"


class BlackjackGame:
    def __init__(
        self,
        *,
        starting_bankroll: float = 1000.0,
        min_bet: float = 10.0,
        max_bet: Optional[float] = None,
        decks: int = 6,
        rules: Optional[Mapping[str, object]] = None,
        shuffle_penetration: float = 0.75,
    ) -> None:
        if starting_bankroll <= 0:
            raise ValueError("Starting bankroll must be positive")
        if min_bet <= 0:
            raise ValueError("Minimum bet must be positive")

        self.rules = dict(DEFAULT_GAME_RULES)
        if rules:
            for key, value in rules.items():
                if key in self.rules:
                    self.rules[key] = bool(value)

        self.shoe = Shoe(decks=decks)
        self.shuffle_penetration = max(0.1, min(0.95, shuffle_penetration))

        self.bankroll = starting_bankroll
        self.min_bet = min_bet
        self.max_bet = max_bet if max_bet and max_bet > min_bet else None
        self.base_bet = min_bet
        self.current_wager = 0.0

        self.player_hand = Hand()
        self.dealer_hand = Hand()
        self.state = STATE_BETTING
        self.message = "Place your bet."
        self.result: Optional[str] = None
        self.hide_dealer_hole_card = True
        self.first_move = True
        self.last_reward = 0.0

    def reset_round(self) -> None:
        self.player_hand.clear()
        self.dealer_hand.clear()
        self.current_wager = 0.0
        self.result = None
        self.message = "Place your bet."
        self.state = STATE_BETTING
        self.hide_dealer_hole_card = True
        self.first_move = True
        self.last_reward = 0.0

    def adjust_bet(self, delta: float) -> None:
        if self.state not in (STATE_BETTING, STATE_RESULT):
            return
        new_bet = max(self.min_bet, self.base_bet + delta)
        if self.max_bet is not None:
            new_bet = min(new_bet, self.max_bet)
        new_bet = min(new_bet, self.bankroll)
        self.base_bet = round(new_bet, 2)

    def set_bet(self, value: float) -> None:
        if self.state not in (STATE_BETTING, STATE_RESULT):
            return
        value = max(self.min_bet, value)
        if self.max_bet is not None:
            value = min(value, self.max_bet)
        value = min(value, self.bankroll)
        self.base_bet = round(value, 2)

    def can_start_round(self) -> bool:
        return self.state in (STATE_BETTING, STATE_RESULT) and self.bankroll >= self.min_bet

    def start_round(self) -> None:
        if not self.can_start_round():
            self.message = "Bankroll insuffisant."
            return

        wager = min(self.base_bet, self.bankroll)
        if wager < self.min_bet:
            self.message = "Mise minimale non satisfaite."
            return

        self.shoe.shuffle_if_needed()
        self.player_hand.clear()
        self.dealer_hand.clear()
        self.result = None
        self.hide_dealer_hole_card = True
        self.first_move = True
        self.last_reward = 0.0

        self.current_wager = round(wager, 2)
        self.bankroll = round(self.bankroll - self.current_wager, 2)

        self.player_hand.add(self.shoe.draw())
        self.dealer_hand.add(self.shoe.draw())
        self.player_hand.add(self.shoe.draw())
        self.dealer_hand.add(self.shoe.draw())

        self.state = STATE_PLAYER
        self.message = "À vous de jouer."

        self._check_initial_blackjack()

    def _check_initial_blackjack(self) -> None:
        player_blackjack = self.player_hand.is_blackjack()
        dealer_blackjack = self.dealer_hand.is_blackjack()

        if not player_blackjack and not dealer_blackjack:
            return

        self.hide_dealer_hole_card = False
        self.state = STATE_RESULT

        if player_blackjack and dealer_blackjack:
            self.result = "push"
            payout = self.current_wager
            self.message = "Blackjack des deux côtés. Égalité."
        elif player_blackjack:
            self.result = "blackjack"
            payout = self.current_wager * 2.5
            self.message = "Blackjack ! Vous gagnez 3:2."
        else:
            self.result = "dealer_blackjack"
            payout = 0.0
            self.message = "Le croupier a un blackjack."

        self.bankroll = round(self.bankroll + payout, 2)

    def _player_total(self) -> Tuple[int, bool]:
        return self.player_hand.total()

    def _dealer_total(self) -> Tuple[int, bool]:
        return self.dealer_hand.total()

    def player_hit(self) -> None:
        if self.state != STATE_PLAYER:
            return
        self.player_hand.add(self.shoe.draw())
        total, _ = self._player_total()
        self.first_move = False
        if total > 21:
            self._finish_round("bust", "Vous dépassez 21.")

    def player_stand(self) -> None:
        if self.state != STATE_PLAYER:
            return
        self.first_move = False
        self._dealer_play()

    def player_double(self) -> None:
        if self.state != STATE_PLAYER or not self.first_move:
            return
        if self.bankroll < self.current_wager:
            self.message = "Fonds insuffisants pour doubler."
            return
        self.bankroll = round(self.bankroll - self.current_wager, 2)
        self.current_wager = round(self.current_wager * 2, 2)
        self.player_hand.add(self.shoe.draw())
        total, _ = self._player_total()
        self.hide_dealer_hole_card = False
        if total > 21:
            self._finish_round("bust", "Double raté : vous dépassez 21.")
        else:
            self._dealer_play()

    def player_surrender(self) -> None:
        if self.state != STATE_PLAYER or not self.first_move or not self.rules.get("surrender_allowed", True):
            return
        self.hide_dealer_hole_card = False
        refund = round(self.current_wager * 0.5, 2)
        self.bankroll = round(self.bankroll + refund, 2)
        self._finish_round("surrender", "Vous abandonnez : moitié de la mise retournée.")

    def _dealer_play(self) -> None:
        self.state = STATE_DEALER
        self.hide_dealer_hole_card = False

        dealer_total, dealer_soft = self._dealer_total()
        while dealer_total < 17 or (
            dealer_total == 17
            and dealer_soft
            and self.rules.get("dealer_hits_on_soft_17", False)
        ):
            self.dealer_hand.add(self.shoe.draw())
            dealer_total, dealer_soft = self._dealer_total()

        self._resolve_outcome()

    def _resolve_outcome(self) -> None:
        player_total, _ = self._player_total()
        dealer_total, _ = self._dealer_total()

        if dealer_total > 21:
            self._finish_round("dealer_bust", "Le croupier dépasse 21. Vous gagnez.")
            return
        if player_total > dealer_total:
            self._finish_round("win", "Vous gagnez.")
            return
        if player_total < dealer_total:
            self._finish_round("lose", "Le croupier gagne.")
            return
        self._finish_round("push", "Égalité.")

    def _finish_round(self, result: str, message: str) -> None:
        self.state = STATE_RESULT
        self.result = result
        payout = 0.0
        if result in {"win", "dealer_bust"}:
            payout = self.current_wager * 2
        elif result == "blackjack":
            payout = self.current_wager * 2.5
        elif result == "push":
            payout = self.current_wager
        elif result == "surrender":
            # bankroll already refunded half when surrender was executed
            payout = 0.0
        elif result == "dealer_blackjack":
            payout = 0.0
        elif result == "bust":
            payout = 0.0

        if payout:
            self.bankroll = round(self.bankroll + payout, 2)

        self.message = message

        reward_lookup = {
            "win": 1.0,
            "dealer_bust": 1.0,
            "blackjack": 1.5,
            "push": 0.0,
            "lose": -1.0,
            "bust": -1.0,
            "dealer_blackjack": -1.0,
            "surrender": -0.5,
        }
        self.last_reward = reward_lookup.get(result or "", 0.0)

    def round_summary(self) -> Dict[str, object]:
        player_total, player_soft = self._player_total()
        dealer_total, dealer_soft = self._dealer_total()
        return {
            "result": self.result,
            "player_total": player_total,
            "player_soft": player_soft,
            "dealer_total": dealer_total,
            "dealer_soft": dealer_soft,
            "player_hand": self.player_hand.description(),
            "dealer_hand": self.dealer_hand.description(),
            "wager": self.current_wager,
            "reward": self.last_reward,
        }

    def recommended_action(self, advisor: Optional["AdvisorBridge"]) -> Optional[str]:
        if self.state != STATE_PLAYER or advisor is None:
            return None
        dealer_card = self.dealer_hand.cards[0] if self.dealer_hand.cards else None
        if dealer_card is None:
            return None
        return advisor.recommend(self)

    def penetration(self) -> float:
        return self.shoe.penetration()


class AdvisorBridge:
    def __init__(
        self,
        policy_path: Optional[str] = None,
        *,
        rules: Optional[Mapping[str, object]] = None,
        online_learning: bool = False,
        learning_rate: float = 0.2,
        exploration: float = 0.1,
    ) -> None:
        self.rules = dict(DEFAULT_GAME_RULES)
        self.rules["double_allowed"] = True
        if rules:
            for key, value in rules.items():
                if key in self.rules:
                    self.rules[key] = bool(value)
                if key == "double_allowed":
                    self.rules[key] = bool(value)

        self.advanced: Optional[AdvancedAdvisor] = None
        if policy_path:
            policy_path = str(Path(policy_path).expanduser())
            try:
                self.advanced = AdvancedAdvisor.from_file(policy_path, rules=self.rules)
            except FileNotFoundError:
                raise

        self.online_learning = bool(online_learning)
        self.learning_rate = float(learning_rate)
        self.exploration = float(exploration)
        self.q_table: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.history: List[Tuple[str, str]] = []
        self.training_rounds = 0
        self.visit_counts: Dict[str, int] = defaultdict(int)

    def recommend(self, game: "BlackjackGame") -> Optional[str]:
        if not game.player_hand.cards or not game.dealer_hand.cards:
            return None

        dealer_card = game.dealer_hand.cards[0]
        player_strategy = [card.as_strategy_card() for card in game.player_hand.cards]
        dealer_strategy = dealer_card.as_strategy_card()
        state_key = self._state_key(player_strategy, dealer_strategy, first_move=game.first_move)
        if state_key is None:
            return None

        valid_actions = self._valid_actions(game)
        if not valid_actions:
            return None

        if self.online_learning:
            online_choice = self._recommend_online(state_key, valid_actions)
            if online_choice:
                return online_choice

        if self.advanced is not None:
            action = self.advanced.recommend(player_strategy, dealer_strategy)
        else:
            action = get_expert_advice(player_strategy, dealer_strategy, self.rules)

        if action not in valid_actions:
            fallback = [act for act in valid_actions if act != "Surrender"] or valid_actions
            action = fallback[0]
        return action

    def label_fr(self, action: Optional[str]) -> Optional[str]:
        if not action:
            return None
        return ACTION_LABELS_FR.get(action, action)

    def begin_round(self, game: "BlackjackGame") -> None:
        if not self.online_learning:
            return
        self.history.clear()

    def record_action(self, game: "BlackjackGame", action: str) -> None:
        if not self.online_learning:
            return
        dealer_card = game.dealer_hand.cards[0] if game.dealer_hand.cards else None
        if dealer_card is None:
            return
        player_strategy = [card.as_strategy_card() for card in game.player_hand.cards]
        dealer_strategy = dealer_card.as_strategy_card()
        state_key = self._state_key(player_strategy, dealer_strategy, first_move=game.first_move)
        if state_key is None:
            return
        self.history.append((state_key, action))
        self.visit_counts[state_key] += 1

    def finish_round(self, game: "BlackjackGame") -> None:
        if not self.online_learning or not self.history:
            self.history.clear()
            return

        reward = float(game.last_reward)
        for state_key, action in self.history:
            state_values = self.q_table.setdefault(state_key, {})
            current = state_values.get(action, 0.0)
            state_values[action] = current + self.learning_rate * (reward - current)
        self.history.clear()
        self.training_rounds += 1

    def load_online_policy(self, path: str | Path) -> None:
        if not self.online_learning:
            raise ValueError("Le chargement d'une policy en ligne nécessite --online-learning.")
        policy_path = Path(path).expanduser()
        if not policy_path.exists():
            raise FileNotFoundError(f"Policy introuvable: {policy_path}")
        with policy_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        q_values = data.get("q_values", {})
        if isinstance(q_values, dict):
            for state_key, values in q_values.items():
                if not isinstance(values, (list, tuple)):
                    continue
                state_store = self.q_table.setdefault(state_key, {})
                for idx, raw_value in enumerate(values):
                    if idx >= len(ACTION_LABELS):
                        break
                    state_store[ACTION_LABELS[idx]] = float(raw_value)

        policy = data.get("policy", {})
        if isinstance(policy, dict):
            for state_key, action_name in policy.items():
                if action_name in ACTION_INDEX:
                    state_store = self.q_table.setdefault(state_key, {})
                    state_store.setdefault(action_name, 0.0)

        visits = data.get("visits", {})
        if isinstance(visits, dict):
            for state_key, count in visits.items():
                try:
                    self.visit_counts[state_key] = int(count)
                except (TypeError, ValueError):
                    continue

        meta = data.get("meta", {})
        if isinstance(meta, dict):
            saved_rules = meta.get("rules")
            if isinstance(saved_rules, Mapping):
                mismatched = {
                    key: (self.rules.get(key), bool(saved_rules.get(key)))
                    for key in ("surrender_allowed", "double_allowed", "dealer_hits_on_soft_17")
                    if key in self.rules
                    and key in saved_rules
                    and bool(saved_rules.get(key)) != bool(self.rules.get(key))
                }
                if mismatched:
                    print(
                        "[AdvisorBridge] Avertissement: règles différentes entre la policy chargée et la configuration courante:",
                        mismatched,
                    )
            rounds = meta.get("training_rounds")
            if isinstance(rounds, (int, float)):
                self.training_rounds = max(self.training_rounds, int(rounds))

        print(f"[AdvisorBridge] Policy en ligne chargée depuis {policy_path}")

    def save_online_policy(self, path: str | Path) -> Path:
        if not self.online_learning:
            raise ValueError("La sauvegarde d'une policy en ligne nécessite --online-learning.")
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)

        q_snapshot: Dict[str, List[float]] = {}
        for state_key, actions in self.q_table.items():
            values = [0.0] * len(ACTION_LABELS)
            for action_name, score in actions.items():
                if action_name in ACTION_INDEX:
                    values[ACTION_INDEX[action_name]] = float(score)
            q_snapshot[state_key] = values

        visits = {state: int(self.visit_counts.get(state, 0)) for state in q_snapshot.keys()}
        meta = {
            "source": "blackjack_game_online",
            "training_rounds": self.training_rounds,
        }

        save_policy(
            str(target),
            q_snapshot,
            visits,
            meta=meta,
            rules=self.rules,
        )
        print(f"[AdvisorBridge] Policy en ligne sauvegardée vers {target}")
        return target

    def _recommend_online(self, state_key: str, valid_actions: Sequence[str]) -> Optional[str]:
        values = self.q_table.get(state_key)
        if not values:
            return None
        scored = [(action, values.get(action, 0.0)) for action in valid_actions]
        if not scored:
            return None
        if self.exploration > 0.0 and random.random() < self.exploration:
            return random.choice(valid_actions)
        best_action, _ = max(scored, key=lambda item: item[1])
        return best_action

    def _valid_actions(self, game: "BlackjackGame") -> List[str]:
        actions = ["Stand", "Hit"]
        if (
            game.first_move
            and self.rules.get("double_allowed", True)
            and game.bankroll >= game.current_wager
        ):
            actions.append("Double")
        if game.first_move and self.rules.get("surrender_allowed", True):
            actions.append("Surrender")
        return actions

    @staticmethod
    def _state_key(
        player_strategy: Sequence[Mapping[str, object]],
        dealer_strategy: Mapping[str, object],
        *,
        first_move: bool,
    ) -> Optional[str]:
        if not player_strategy or not dealer_strategy:
            return None

        total, usable_ace = evaluate_hand(player_strategy)
        dealer_value = int(dealer_strategy.get("value", 0))
        if dealer_value <= 0:
            return None
        is_pair = False
        if first_move and len(player_strategy) == 2:
            is_pair = player_strategy[0]["rank"] == player_strategy[1]["rank"]
        return (
            f"{total}|{int(usable_ace)}|{dealer_value}|"
            f"{int(first_move)}|{int(is_pair)}"
        )


__all__ = [
    "AdvisorBridge",
    "BlackjackGame",
    "Card",
    "Hand",
    "STATE_BETTING",
    "STATE_DEALER",
    "STATE_PLAYER",
    "STATE_RESULT",
    "Shoe",
    "SUITS",
    "RANKS",
]

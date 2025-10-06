from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pygame

from blackjack.game import (
    AdvisorBridge,
    BlackjackGame,
    STATE_BETTING,
    STATE_PLAYER,
    STATE_RESULT,
)

CARD_ASSETS = Path("dataset/png")
BACKGROUND_DIR = Path("dataset/backgrounds")
BACKGROUND_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

CARD_ASPECT_RATIO = 0.71  # width / height approximation
BUTTON_HEIGHT = 48
BUTTON_WIDTH = 160
BUTTON_SPACING = 12
PANEL_PADDING = 20

COLOR_WHITE = (240, 240, 240)
COLOR_BLACK = (10, 10, 10)
COLOR_SHADOW = (0, 0, 0, 100)
COLOR_PRIMARY = (38, 115, 198)
COLOR_PRIMARY_DISABLED = (90, 110, 130)
COLOR_BACKGROUND_FALLBACK = (21, 82, 56)
COLOR_PANEL = (0, 0, 0, 140)


class Button:
    def __init__(self, label: str, rect: pygame.Rect, callback) -> None:
        self.label = label
        self.rect = rect
        self.callback = callback
        self.enabled = True

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        color = COLOR_PRIMARY if self.enabled else COLOR_PRIMARY_DISABLED
        pygame.draw.rect(surface, color, self.rect, border_radius=8)
        text = font.render(self.label, True, COLOR_WHITE)
        text_rect = text.get_rect(center=self.rect.center)
        surface.blit(text, text_rect)

    def handle(self, event: pygame.event.Event) -> None:
        if not self.enabled:
            return
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.callback()


def _background_candidates(directory: Path) -> list[Path]:
    candidates = []
    for entry in directory.iterdir():
        if entry.is_file() and entry.suffix.lower() in BACKGROUND_EXTENSIONS:
            candidates.append(entry)
    return candidates


def load_background(size: Tuple[int, int], path: Optional[Path]) -> pygame.Surface:
    screen = pygame.Surface(size)
    screen.fill(COLOR_BACKGROUND_FALLBACK)
    source = None
    if path:
        guess = Path(path)
        if guess.is_file() and guess.suffix.lower() in BACKGROUND_EXTENSIONS:
            source = guess
    else:
        candidates = _background_candidates(BACKGROUND_DIR) if BACKGROUND_DIR.exists() else []
        if candidates:
            source = random.choice(candidates)
    if source and source.exists():
        try:
            image = pygame.image.load(str(source))
            image = image.convert_alpha() if image.get_alpha() else image.convert()
            return pygame.transform.smoothscale(image, size)
        except pygame.error:
            pass
    return screen


def load_card_images(card_dir: Path, target_height: int) -> Dict[str, pygame.Surface]:
    assets: Dict[str, pygame.Surface] = {}
    target_size = (int(target_height * CARD_ASPECT_RATIO), target_height)
    for path in card_dir.glob("*_of_*.png"):
        surface = pygame.image.load(str(path)).convert_alpha()
        surface = pygame.transform.smoothscale(surface, target_size)
        assets[path.stem] = surface
    return assets


def create_card_back(size: Tuple[int, int]) -> pygame.Surface:
    card = pygame.Surface(size, pygame.SRCALPHA)
    card.fill((170, 30, 55))
    pygame.draw.rect(card, (230, 230, 230), card.get_rect(), width=4, border_radius=12)
    pygame.draw.rect(card, (255, 255, 255, 70), card.get_rect().inflate(-16, -16), width=2, border_radius=8)
    return card


def get_card_surface(name: str, lookup: Dict[str, pygame.Surface]) -> Optional[pygame.Surface]:
    return lookup.get(name)


def format_currency(value: float) -> str:
    return f"{value:,.2f}".replace(",", " ")


class BlackjackApp:
    def __init__(self, args: argparse.Namespace) -> None:
        pygame.init()
        pygame.display.set_caption("Blackjack Trainer")
        self.screen = pygame.display.set_mode((args.width, args.height))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 44)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 26)

        card_height = int(args.height * 0.28)
        self.card_size = (int(card_height * CARD_ASPECT_RATIO), card_height)
        self.card_spacing = int(self.card_size[0] * 0.6)

        self.background = load_background((args.width, args.height), args.background)
        self.card_images = load_card_images(CARD_ASSETS, card_height)
        self.card_back = create_card_back(self.card_size)

        rules = {
            "dealer_hits_on_soft_17": args.dealer_hits_soft_17,
            "surrender_allowed": not args.disable_surrender,
        }
        self.game = BlackjackGame(
            starting_bankroll=args.bankroll,
            min_bet=args.min_bet,
            max_bet=args.max_bet,
            decks=args.decks,
            rules=rules,
            shuffle_penetration=args.shuffle_penetration,
        )
        advisor_rules = dict(rules)
        advisor_rules["double_allowed"] = not args.disable_double
        enable_advisor = bool(
            args.advanced_policy or args.use_chart_advisor or args.online_learning
        )
        if args.load_online_policy and not args.online_learning:
            raise ValueError("--load-online-policy nécessite d'activer --online-learning.")
        if args.save_online_policy and not args.online_learning:
            raise ValueError("--save-online-policy nécessite d'activer --online-learning.")
        self.advisor = (
            AdvisorBridge(
                policy_path=args.advanced_policy,
                rules=advisor_rules,
                online_learning=args.online_learning,
                learning_rate=args.learning_rate,
                exploration=args.exploration,
            )
            if enable_advisor
            else None
        )

        self.running = True
        self.round_concluded = False
        self.auto_play = bool(args.auto_play)
        if self.auto_play and self.advisor is None:
            raise ValueError(
                "Le mode auto-play nécessite un advisor. Activez --use-chart-advisor ou fournissez --advanced-policy."
            )
        self.auto_delay_ms = int(max(0.0, args.auto_delay) * 1000)
        self.auto_round_limit = (
            args.auto_rounds if args.auto_rounds and args.auto_rounds > 0 else None
        )
        self.auto_rounds_completed = 0
        self._last_auto_tick = 0
        self.save_online_policy_path = (
            Path(args.save_online_policy).expanduser()
            if args.save_online_policy
            else None
        )
        self.save_online_interval = max(0, args.save_online_interval)
        self._last_saved_training_round = 0

        if self.advisor and args.load_online_policy:
            self.advisor.load_online_policy(args.load_online_policy)

        btn_y = args.height - BUTTON_HEIGHT - PANEL_PADDING
        start_x = PANEL_PADDING
        self.buttons: Dict[str, Button] = {}
        self.buttons["deal"] = Button(
            "Distribuer",
            pygame.Rect(start_x, btn_y, BUTTON_WIDTH, BUTTON_HEIGHT),
            self.start_round,
        )
        self.buttons["hit"] = Button(
            "Tirer",
            pygame.Rect(start_x + (BUTTON_WIDTH + BUTTON_SPACING) * 1, btn_y, BUTTON_WIDTH, BUTTON_HEIGHT),
            lambda action="Hit": self._execute_action(action),
        )
        self.buttons["stand"] = Button(
            "Rester",
            pygame.Rect(start_x + (BUTTON_WIDTH + BUTTON_SPACING) * 2, btn_y, BUTTON_WIDTH, BUTTON_HEIGHT),
            lambda action="Stand": self._execute_action(action),
        )
        self.buttons["double"] = Button(
            "Doubler",
            pygame.Rect(start_x + (BUTTON_WIDTH + BUTTON_SPACING) * 3, btn_y, BUTTON_WIDTH, BUTTON_HEIGHT),
            lambda action="Double": self._execute_action(action),
        )
        self.buttons["surrender"] = Button(
            "Abandonner",
            pygame.Rect(start_x + (BUTTON_WIDTH + BUTTON_SPACING) * 4, btn_y, BUTTON_WIDTH, BUTTON_HEIGHT),
            lambda action="Surrender": self._execute_action(action),
        )
        self.buttons["next"] = Button(
            "Nouvelle main",
            pygame.Rect(start_x + (BUTTON_WIDTH + BUTTON_SPACING) * 5, btn_y, BUTTON_WIDTH, BUTTON_HEIGHT),
            self._new_round,
        )

        self.args = args

    def start_round(self) -> None:
        if self.game.state == STATE_RESULT:
            self.game.reset_round()
        self.game.start_round()
        self.round_concluded = False
        if self.advisor:
            self.advisor.begin_round(self.game)
        self._check_round_end()

    def adjust_bet(self, delta: float) -> None:
        self.game.adjust_bet(delta)

    def _new_round(self) -> None:
        self.game.reset_round()
        self.round_concluded = False

    def _action_available(self, action: str) -> bool:
        if self.game.state != STATE_PLAYER:
            return False
        if action == "Double":
            if self.args.disable_double:
                return False
            return self.game.first_move and self.game.bankroll >= self.game.current_wager
        if action == "Surrender":
            if self.args.disable_surrender:
                return False
            return self.game.first_move
        return True

    def _record_action(self, action: str) -> None:
        if self.advisor:
            self.advisor.record_action(self.game, action)

    def _execute_action(self, action: str) -> None:
        if not self._action_available(action):
            return
        self._record_action(action)
        if action == "Hit":
            self.game.player_hit()
        elif action == "Stand":
            self.game.player_stand()
        elif action == "Double":
            self.game.player_double()
        elif action == "Surrender":
            self.game.player_surrender()
        self._check_round_end()

    def _check_round_end(self) -> None:
        if self.round_concluded or self.game.state != STATE_RESULT:
            return
        if self.advisor:
            self.advisor.finish_round(self.game)
        self.round_concluded = True
        if self.auto_play:
            self.auto_rounds_completed += 1
            if self.auto_round_limit and self.auto_rounds_completed >= self.auto_round_limit:
                self.running = False
        self._persist_online_policy()

    def _persist_online_policy(self, force: bool = False) -> None:
        if (
            self.save_online_policy_path is None
            or not self.advisor
            or not self.advisor.online_learning
        ):
            return
        if not force:
            if self.save_online_interval <= 0:
                return
            if self.advisor.training_rounds == 0:
                return
            if (
                self.advisor.training_rounds % self.save_online_interval
            ) != 0:
                return
            if self.advisor.training_rounds == self._last_saved_training_round:
                return
        try:
            self.advisor.save_online_policy(self.save_online_policy_path)
            self._last_saved_training_round = self.advisor.training_rounds
        except Exception as exc:
            print(f"[AutoSave] Échec de la sauvegarde de la policy : {exc}")

    def _maybe_auto_play(self) -> None:
        if not self.auto_play or not self.running:
            return

        now = pygame.time.get_ticks()
        if now - self._last_auto_tick < self.auto_delay_ms:
            return

        state = self.game.state
        if state == STATE_RESULT:
            if self.auto_round_limit and self.auto_rounds_completed >= self.auto_round_limit:
                self.running = False
                return
            self._new_round()
            self._last_auto_tick = now
            return

        if state == STATE_BETTING:
            if self.game.can_start_round():
                self.start_round()
                self._last_auto_tick = now
            return

        if state != STATE_PLAYER:
            return

        action = self.game.recommended_action(self.advisor)
        if action is None:
            fallback = [
                candidate
                for candidate in ("Stand", "Hit", "Double", "Surrender")
                if self._action_available(candidate)
            ]
            action = fallback[0] if fallback else None
        if action:
            self._execute_action(action)
            self._last_auto_tick = now

    def draw(self) -> None:
        self.screen.blit(self.background, (0, 0))
        self._draw_panels()
        self._draw_dealer_area()
        self._draw_player_area()
        self._draw_buttons()
        pygame.display.flip()

    def _draw_panels(self) -> None:
        width, height = self.screen.get_size()
        panel_rect = pygame.Rect(PANEL_PADDING, PANEL_PADDING, width - PANEL_PADDING * 2, 120)
        overlay = pygame.Surface(panel_rect.size, pygame.SRCALPHA)
        overlay.fill(COLOR_PANEL)
        self.screen.blit(overlay, panel_rect.topleft)

        bankroll_text = f"Bankroll : {format_currency(self.game.bankroll)}"
        bet_text = f"Mise : {format_currency(self.game.base_bet)}"
        state_text = self.game.message
        penetration = f"Shoe : {self.game.penetration() * 100:.0f}%"

        self.screen.blit(self.font_medium.render(bankroll_text, True, COLOR_WHITE), (panel_rect.x + 16, panel_rect.y + 12))
        self.screen.blit(self.font_medium.render(bet_text, True, COLOR_WHITE), (panel_rect.x + 16, panel_rect.y + 48))
        self.screen.blit(self.font_small.render(penetration, True, COLOR_WHITE), (panel_rect.x + 16, panel_rect.y + 82))
        self.screen.blit(self.font_medium.render(state_text, True, COLOR_WHITE), (panel_rect.x + 320, panel_rect.y + 32))
        if self.advisor and self.advisor.online_learning:
            training_text = f"Apprentissage : {self.advisor.training_rounds} mains"
            self.screen.blit(
                self.font_small.render(training_text, True, COLOR_WHITE),
                (panel_rect.x + 320, panel_rect.y + 8),
            )
        if self.auto_play:
            if self.auto_round_limit:
                auto_text = f"Auto : {self.auto_rounds_completed}/{self.auto_round_limit}"
            else:
                auto_text = f"Auto : {self.auto_rounds_completed}"
            self.screen.blit(
                self.font_small.render(auto_text, True, COLOR_WHITE),
                (panel_rect.x + 320, panel_rect.y + 92),
            )

        if self.game.state == STATE_PLAYER and self.advisor is not None:
            action = self.game.recommended_action(self.advisor)
            if action:
                label_fr = self.advisor.label_fr(action)
                suggestion = f"Conseil: {label_fr} ({action})"
                self.screen.blit(
                    self.font_medium.render(suggestion, True, COLOR_WHITE),
                    (panel_rect.x + 320, panel_rect.y + 68),
                )

    def _draw_dealer_area(self) -> None:
        width, _ = self.screen.get_size()
        dealer_count = max(len(self.game.dealer_hand.cards), 1)
        base_x = width // 2 - (dealer_count - 1) * self.card_spacing // 2
        y = PANEL_PADDING + 140

        for idx, card in enumerate(self.game.dealer_hand.cards):
            pos = (base_x + idx * self.card_spacing, y)
            if idx == 1 and self.game.hide_dealer_hole_card:
                self.screen.blit(self.card_back, pos)
            else:
                surface = get_card_surface(card.asset_name, self.card_images)
                if surface:
                    self.screen.blit(surface, pos)

        total, soft = self.game.dealer_hand.total() if not self.game.hide_dealer_hole_card else (0, False)
        if not self.game.hide_dealer_hole_card:
            info = f"Total croupier: {total}{' (soft)' if soft else ''}"
            text = self.font_medium.render(info, True, COLOR_WHITE)
            self.screen.blit(text, (PANEL_PADDING, y + self.card_size[1] + 12))

    def _draw_player_area(self) -> None:
        width, height = self.screen.get_size()
        player_count = max(len(self.game.player_hand.cards), 1)
        base_x = width // 2 - (player_count - 1) * self.card_spacing // 2
        y = height // 2 + 20

        for idx, card in enumerate(self.game.player_hand.cards):
            pos = (base_x + idx * self.card_spacing, y)
            surface = get_card_surface(card.asset_name, self.card_images)
            if surface:
                self.screen.blit(surface, pos)

        if self.game.player_hand.cards:
            total, soft = self.game.player_hand.total()
            info = f"Total joueur: {total}{' (soft)' if soft else ''}"
            text = self.font_medium.render(info, True, COLOR_WHITE)
            self.screen.blit(text, (PANEL_PADDING, y - 40))

        if self.game.state == STATE_RESULT and self.game.result:
            summary = self.game.round_summary()
            summary_text = (
                f"Résultat: {summary['result']} | Joueur: {summary['player_hand']} ({summary['player_total']})"
                f" vs Croupier: {summary['dealer_hand']} ({summary['dealer_total']})"
            )
            self.screen.blit(self.font_small.render(summary_text, True, COLOR_WHITE), (PANEL_PADDING, y + self.card_size[1] + 20))

    def _draw_buttons(self) -> None:
        self._update_button_states()
        for button in self.buttons.values():
            button.draw(self.screen, self.font_medium)

    def _update_button_states(self) -> None:
        state = self.game.state

        self.buttons["deal"].enabled = state in (STATE_BETTING, STATE_RESULT) and self.game.can_start_round()
        self.buttons["hit"].enabled = state == STATE_PLAYER
        self.buttons["stand"].enabled = state == STATE_PLAYER
        self.buttons["double"].enabled = self._action_available("Double")
        self.buttons["surrender"].enabled = self._action_available("Surrender")
        self.buttons["next"].enabled = state == STATE_RESULT

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            self.running = False
            return
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.running = False
                return
            if event.key == pygame.K_UP:
                self.adjust_bet(self.args.bet_step)
            elif event.key == pygame.K_DOWN:
                self.adjust_bet(-self.args.bet_step)
            elif event.key in (pygame.K_SPACE, pygame.K_RETURN):
                if self.game.state in (STATE_BETTING, STATE_RESULT):
                    self.start_round()
                elif self.game.state == STATE_PLAYER:
                    self._execute_action("Hit")
            elif event.key == pygame.K_h:
                self._execute_action("Hit")
            elif event.key == pygame.K_s:
                self._execute_action("Stand")
            elif event.key == pygame.K_d:
                self._execute_action("Double")
            elif event.key == pygame.K_r:
                self._execute_action("Surrender")
        for button in self.buttons.values():
            button.handle(event)

    def run(self) -> None:
        while self.running:
            for event in pygame.event.get():
                self.handle_event(event)
            self._check_round_end()
            self._maybe_auto_play()
            self.draw()
            self.clock.tick(self.args.fps)
        self._persist_online_policy(force=True)
        pygame.quit()


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jeu de blackjack interactif pour entraîner l'advisor en temps réel.")
    parser.add_argument("--width", type=int, default=1280, help="Largeur de la fenêtre")
    parser.add_argument("--height", type=int, default=720, help="Hauteur de la fenêtre")
    parser.add_argument("--fps", type=int, default=60, help="Fréquence de rafraîchissement cible")
    parser.add_argument("--decks", type=int, default=6, help="Nombre de paquets dans le sabot")
    parser.add_argument("--bankroll", type=float, default=1000.0, help="Bankroll initiale")
    parser.add_argument("--min-bet", dest="min_bet", type=float, default=10.0, help="Mise minimale")
    parser.add_argument("--max-bet", dest="max_bet", type=float, default=None, help="Mise maximale")
    parser.add_argument("--bet-step", dest="bet_step", type=float, default=5.0, help="Incrément de mise avec les flèches")
    parser.add_argument("--shuffle-penetration", type=float, default=0.75, help="Pénétration du sabot avant mélange (0-1)")
    parser.add_argument("--advanced-policy", type=str, default=None, help="Chemin vers un policy JSON entraîné")
    parser.add_argument("--use-chart-advisor", action="store_true", help="Afficher les conseils chart même sans policy avancée")
    parser.add_argument("--dealer-hits-soft-17", action="store_true", help="Le croupier tire sur 17 soft")
    parser.add_argument("--disable-surrender", action="store_true", help="Désactiver l'abandon")
    parser.add_argument("--disable-double", action="store_true", help="Désactiver le double")
    parser.add_argument("--background", type=str, default=None, help="Chemin vers une image de fond personnalisée")
    parser.add_argument("--online-learning", action="store_true", help="Activer l'apprentissage en ligne basé sur les mains jouées")
    parser.add_argument("--learning-rate", type=float, default=0.2, help="Taux d'apprentissage pour la mise à jour des Q-valeurs")
    parser.add_argument("--exploration", type=float, default=0.1, help="Probabilité epsilon-greedy pour explorer de nouvelles actions")
    parser.add_argument(
        "--load-online-policy",
        type=str,
        default=None,
        help="Charger un fichier JSON contenant un Q-table pour reprendre l'apprentissage en ligne",
    )
    parser.add_argument(
        "--save-online-policy",
        type=str,
        default=None,
        help="Sauvegarder périodiquement la policy en ligne vers ce fichier JSON",
    )
    parser.add_argument(
        "--save-online-interval",
        type=int,
        default=0,
        help="Nombre de manches entre deux sauvegardes auto (0 = uniquement en fin de session)",
    )
    parser.add_argument("--auto-play", action="store_true", help="La table joue automatiquement en suivant l'advisor")
    parser.add_argument("--auto-delay", type=float, default=0.3, help="Délai entre deux actions auto (secondes)")
    parser.add_argument(
        "--auto-rounds",
        type=int,
        default=None,
        help="Arrêter le mode auto-play après N manches (illimité si absent)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    if not CARD_ASSETS.exists():
        raise FileNotFoundError(f"Répertoire d'assets introuvable: {CARD_ASSETS}")
    app = BlackjackApp(args)
    app.run()


if __name__ == "__main__":
    main()

"""CLI utilitaire pour provisionner une instance Vast.ai et lancer l'entraînement.

Usage basique :
    python tools/vast_train.py --api-key ... --git-url https://github.com/... -- --algo dqn --episodes 100000
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests


DEFAULT_BASE_URL = os.getenv("VAST_URL", "https://console.vast.ai")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class VastAIError(RuntimeError):
    """Erreur spécifique à l'API Vast.ai."""


class VastAIClient:
    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL, *, timeout: float = 30.0, max_retries: int = 5) -> None:
        if not api_key:
            raise ValueError("Une clé API Vast.ai est requise. Fournissez --api-key ou définissez VAST_API_KEY.")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "blackjack-ai-trainer/0.1",
            }
        )
        self.session.params.update({"api_key": api_key})

    def request(self, method: str, path: str, *, params: Optional[Dict[str, Any]] = None, json: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}/api/v0{path}"
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.request(method, url, params=params, json=json, timeout=self.timeout)
            except requests.RequestException as exc:  # pragma: no cover - réseau externe
                if attempt == self.max_retries:
                    raise VastAIError(f"Requête {method} {url} échouée après {attempt} tentatives: {exc}") from exc
                time.sleep(min(2 ** attempt, 10))
                continue

            if response.status_code == 429 and attempt < self.max_retries:
                time.sleep(min(2 ** attempt, 10))
                continue

            if response.status_code >= 400:
                raise VastAIError(
                    f"API Vast.ai a retourné {response.status_code} pour {method} {path}: {response.text}"
                )

            if "application/json" not in response.headers.get("Content-Type", ""):
                raise VastAIError(
                    f"Réponse inattendue pour {method} {path} (Content-Type={response.headers.get('Content-Type')})."
                )
            return response.json()
        raise VastAIError(f"Impossible de contacter Vast.ai pour {method} {path}")

    def search_offers(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        payload = {"select_cols": ["*"], "q": query}
        data = self.request("POST", "/bundles/", json=payload)
        offers = data.get("offers", [])
        if not isinstance(offers, list):
            raise VastAIError(f"Structure inattendue pour les offres: {offers!r}")
        return offers

    def create_instance(self, ask_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = self.request("PUT", f"/asks/{ask_id}/", json=payload)
        if not data.get("success", False):
            raise VastAIError(f"Création d'instance refusée: {data}")
        return data

    def get_instance(self, instance_id: int) -> Dict[str, Any]:
        data = self.request("GET", f"/instances/{instance_id}/", params={"owner": "me"})
        row = data.get("instances")
        if isinstance(row, list):
            if not row:
                raise VastAIError(f"Instance {instance_id} introuvable (liste vide renvoyée).")
            return row[0]
        if isinstance(row, dict):
            return row
        raise VastAIError(f"Structure inattendue pour l'instance {instance_id}: {row!r}")

    def wait_for_instance_running(
        self, instance_id: int, *, poll_interval: float = 15.0, timeout: float = 900.0
    ) -> Dict[str, Any]:
        start = time.time()
        last_status: Optional[str] = None
        while True:
            info = self.get_instance(instance_id)
            status = info.get("actual_status") or info.get("status")
            if status != last_status:
                print(f"Instance {instance_id} statut = {status}")
                last_status = status
            if status in {"running", "active", "ready"}:
                return info
            if status in {"terminated", "canceled", "dead"}:
                raise VastAIError(f"Instance {instance_id} arrêtée prématurément (statut {status}).")
            if time.time() - start > timeout:
                raise VastAIError(f"Instance {instance_id} indisponible après {timeout} secondes.")
            time.sleep(poll_interval)

    def run_remote_command(self, instance_id: int, command: str) -> Dict[str, Any]:
        payload = {"command": command}
        data = self.request("PUT", f"/instances/command/{instance_id}/", json=payload)
        if not data.get("success", False):
            raise VastAIError(f"Exécution distante refusée: {data}")
        return data


def detect_git_remote(cwd: Path) -> tuple[Optional[str], Optional[str]]:
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            check=True,
            cwd=cwd,
        )
        url = result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None
    try:
        branch_result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=cwd,
        )
        branch = branch_result.stdout.strip()
    except subprocess.CalledProcessError:
        branch = None
    return url or None, branch or None


def derive_repo_name(git_url: str) -> str:
    cleaned = git_url.rstrip("/")
    name = cleaned.split("/")[-1]
    if name.endswith(".git"):
        name = name[:-4]
    return name or "project"


def shell_join(parts: Iterable[str]) -> str:
    quoted = [shlex.quote(p) for p in parts]
    return " ".join(quoted)


def build_training_command(train_args: List[str], python_bin: str) -> str:
    base_cmd = [python_bin, "train_advanced_advisor.py"]
    base_cmd.extend(train_args)
    return shell_join(base_cmd)


def build_onstart_script(
    *,
    git_url: str,
    git_branch: str,
    remote_workspace: str,
    repo_name: str,
    use_venv: bool,
    venv_name: str,
    train_command: str,
    extra_setup: Optional[List[str]] = None,
    log_dir: str = "logs",
) -> str:
    workspace_q = shlex.quote(remote_workspace)
    repo_q = shlex.quote(repo_name)
    branch_q = shlex.quote(git_branch)
    url_q = shlex.quote(git_url)
    venv_q = shlex.quote(venv_name)
    log_dir_q = shlex.quote(log_dir)

    lines = [
        "#!/bin/bash",
        "set -euxo pipefail",
        f"WORKSPACE={workspace_q}",
        f"REPO_DIR={repo_q}",
        f"BRANCH={branch_q}",
        f"GIT_URL={url_q}",
        "RUN_ID=$(date +%Y%m%d-%H%M%S)",
        f"LOG_DIR={log_dir_q}",
        "mkdir -p \"$WORKSPACE\"",
        "cd \"$WORKSPACE\"",
        "if [ ! -d \"$REPO_DIR/.git\" ]; then",
        "  rm -rf \"$REPO_DIR\"",
        "  git clone --branch \"$BRANCH\" \"$GIT_URL\" \"$REPO_DIR\"",
        "else",
        "  cd \"$REPO_DIR\"",
        "  git fetch origin",
        "  git checkout \"$BRANCH\"",
        "  git reset --hard \"origin/$BRANCH\"",
        "  cd ..",
        "fi",
        "cd \"$REPO_DIR\"",
        "mkdir -p \"$LOG_DIR\"",
    ]

    if extra_setup:
        lines.extend(extra_setup)

    if use_venv:
        lines.extend(
            [
                "if [ ! -d \"$VENV_DIR\" ]; then",
                "  python3 -m venv \"$VENV_DIR\"",
                "fi",
                "source \"$VENV_DIR/bin/activate\"",
            ]
        )
    else:
        lines.append("unset PIP_REQUIRE_VIRTUALENV || true")

    lines.extend(
        [
            "pip install --upgrade pip",
            "if [ -f requirements.txt ]; then pip install -r requirements.txt; fi",
            "if [ -f requirements-train.txt ]; then pip install -r requirements-train.txt; fi",
        ]
    )

    run_line = f"{train_command} 2>&1 | tee \"$LOG_DIR/train-$RUN_ID.log\""
    lines.append(run_line)
    return "\n".join(lines)


def build_query(args: argparse.Namespace) -> Dict[str, Any]:
    query: Dict[str, Any] = {
        "verified": {"eq": True},
        "external": {"eq": False},
        "rentable": {"eq": True},
        "rented": {"eq": False},
        "type": "bid" if args.interruptible else "on-demand",
        "order": [[args.order_by, args.order_direction]],
        "limit": args.offer_limit,
    }
    if args.min_vram_gib is not None:
        query["gpu_ram"] = {"gte": int(args.min_vram_gib * 1000)}
    if args.min_total_vram_gib is not None:
        query["gpu_total_ram"] = {"gte": int(args.min_total_vram_gib * 1000)}
    if args.max_price is not None:
        query["dph_total"] = {"lte": float(args.max_price)}
    if args.gpu_name:
        query["gpu_name"] = {"eq": args.gpu_name}
    if args.num_gpus is not None:
        query["num_gpus"] = {"gte": args.num_gpus}
    if args.location:
        query["geolocation"] = {"eq": args.location}
    return query


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Provisionner et entraîner automatiquement sur Vast.ai.")
    parser.add_argument("--api-key", default=os.getenv("VAST_API_KEY"), help="Clé API Vast.ai (sinon variable VAST_API_KEY)")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="URL de base de l'API Vast.ai")
    parser.add_argument("--image", default="pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime", help="Image Docker à utiliser")
    parser.add_argument("--disk", type=int, default=64, help="Taille disque (Go)")
    parser.add_argument("--label", default="blackjack-ai-train", help="Label pour l'instance")
    parser.add_argument("--ask-id", type=int, help="ID précis d'offre Vast.ai à utiliser (bypass recherche)")
    parser.add_argument("--interruptible", action="store_true", help="Utiliser une offre spot (interruptible)")
    parser.add_argument("--bid-price", type=float, help="Prix max $/h pour une offre spot")
    parser.add_argument("--max-price", type=float, help="Prix max $/h pour l'offre on-demand")
    parser.add_argument("--min-vram-gib", type=float, help="Mémoire GPU minimale par carte (GiB)")
    parser.add_argument("--min-total-vram-gib", type=float, help="Mémoire GPU totale minimale (GiB)")
    parser.add_argument("--num-gpus", type=int, help="Nombre minimal de GPU")
    parser.add_argument("--gpu-name", help="Nom exact de GPU souhaité (ex: RTX 4090)")
    parser.add_argument("--location", help="Code pays ou région à privilégier")
    parser.add_argument("--order-by", default="dph_total", help="Champ de tri pour la recherche d'offres")
    parser.add_argument("--order-direction", choices=["asc", "desc"], default="asc", help="Ordre de tri")
    parser.add_argument("--offer-limit", type=int, default=20, help="Nombre max d'offres à récupérer")
    parser.add_argument("--remote-workspace", default="/workspace", help="Répertoire distant où cloner le projet")
    parser.add_argument("--remote-log-dir", default="logs", help="Sous-dossier distant pour les logs")
    parser.add_argument("--git-url", help="URL git du dépôt à cloner")
    parser.add_argument("--git-branch", help="Branche git à checkout")
    parser.add_argument("--python-bin", default="python3", help="Binaire Python à utiliser sur l'instance")
    parser.add_argument("--venv-name", default=".venv", help="Nom du dossier de venv distant")
    parser.add_argument("--no-venv", action="store_true", help="Ne pas créer/activer d'environnement virtuel")
    parser.add_argument("--dry-run", action="store_true", help="Affiche les actions et quitte sans toucher à Vast.ai")
    parser.add_argument("--skip-wait", action="store_true", help="Ne pas attendre que l'instance soit RUNNING")
    parser.add_argument("--extra-setup", action="append", default=[], help="Commande shell supplémentaire ajoutée avant l'entraînement")
    parser.add_argument("train_args", nargs=argparse.REMAINDER, help="Arguments passés à train_advanced_advisor.py (utiliser -- avant)")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    git_url = args.git_url
    git_branch = args.git_branch
    if not git_url or not git_branch:
        detected_url, detected_branch = detect_git_remote(PROJECT_ROOT)
        git_url = git_url or detected_url
        git_branch = git_branch or detected_branch

    if not git_url:
        raise SystemExit("Impossible de déterminer l'URL git. Fournissez --git-url.")
    if not git_branch:
        git_branch = "main"

    repo_name = derive_repo_name(git_url)
    train_args = args.train_args or []
    if train_args and train_args[0] == "--":
        train_args = train_args[1:]

    train_command = build_training_command(train_args, args.python_bin)

    use_venv = not args.no_venv
    extra_setup = [
        "if ! command -v git >/dev/null 2>&1; then apt-get update && apt-get install -y git; fi",
    ]
    if args.extra_setup:
        extra_setup.extend(args.extra_setup)
    if use_venv:
        extra_setup.append(
            "if ! python3 -m venv --help >/dev/null 2>&1; then apt-get update && apt-get install -y python3-venv; fi"
        )
        extra_setup.append(f"VENV_DIR={shlex.quote(args.venv_name)}")
    else:
        extra_setup.append("VENV_DIR=")

    onstart_script = build_onstart_script(
        git_url=git_url,
        git_branch=git_branch,
        remote_workspace=args.remote_workspace,
        repo_name=repo_name,
        use_venv=use_venv,
        venv_name=args.venv_name,
        train_command=train_command,
        extra_setup=extra_setup,
        log_dir=args.remote_log_dir,
    )

    if args.dry_run:
        print("--- Onstart script ---")
        print(onstart_script)
        return 0

    client = VastAIClient(api_key=args.api_key, base_url=args.base_url)

    if args.ask_id is not None:
        selected_offer = {"id": args.ask_id}
    else:
        query = build_query(args)
        offers = client.search_offers(query)
        if not offers:
            raise SystemExit("Aucune offre Vast.ai ne correspond aux critères.")
        selected_offer = offers[0]
        print("Offre sélectionnée:")
        print(json.dumps({k: selected_offer.get(k) for k in ("id", "gpu_name", "num_gpus", "dph_total", "gpu_ram", "geolocation")}, indent=2))

    ask_id = int(selected_offer["id"])

    payload: Dict[str, Any] = {
        "client_id": "me",
        "image": args.image,
        "disk": args.disk,
        "label": args.label,
        "onstart": onstart_script,
        "env": {},
        "runtype": "ssh",
        "force": False,
        "cancel_unavail": True,
    }

    if args.bid_price is not None:
        payload["price"] = float(args.bid_price)
    else:
        payload["price"] = None

    response = client.create_instance(ask_id, payload)
    instance_id = response.get("new_contract")
    if instance_id is None:
        raise VastAIError(f"Réponse inattendue lors de la création: {response}")

    print(f"Instance Vast.ai créée: ID={instance_id}")

    if args.skip_wait:
        return 0

    info = client.wait_for_instance_running(int(instance_id))
    ssh_host = info.get("ssh_host")
    ssh_port = info.get("ssh_port")
    if ssh_host and ssh_port:
        print(f"Connexion SSH: ssh root@{ssh_host} -p {ssh_port}")
    else:
        print("Informations SSH indisponibles immédiatement. Consultez la console Vast.ai si nécessaire.")
    print("Entraînement lancé via script onstart. Consultez les logs dans", f"{args.remote_workspace}/{repo_name}/{args.remote_log_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

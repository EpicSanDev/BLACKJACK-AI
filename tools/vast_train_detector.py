"""Provisionne une instance Vast.ai pour entraîner le détecteur de cartes via `model/train_model.py`."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from typing import Any, Dict, List, Optional

from tools.vast_train import (
    DEFAULT_BASE_URL,
    DEFAULT_GIT_URL,
    PROJECT_ROOT,
    VastAIClient,
    VastAIError,
    build_onstart_script,
    build_query,
    detect_git_remote,
    derive_repo_name,
    shell_join,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Provisionner automatiquement une instance Vast.ai et lancer model/train_model.py"
    )
    parser.add_argument("--api-key", default=os.getenv("VAST_API_KEY"), help="Clé API Vast.ai (ou variable VAST_API_KEY)")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="URL de base de l'API Vast.ai")
    parser.add_argument("--image", default="pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime", help="Image Docker à utiliser")
    parser.add_argument("--disk", type=int, default=150, help="Taille disque (Go) pour stocker dataset+poids")
    parser.add_argument("--label", default="blackjack-detector-train", help="Label attribué à l'instance")
    parser.add_argument("--ask-id", type=int, help="ID d'offre Vast.ai à utiliser explicitement")
    parser.add_argument("--interruptible", action="store_true", help="Louer une offre spot (interruptible)")
    parser.add_argument("--bid-price", type=float, help="Prix max en $/h pour une offre spot")
    parser.add_argument("--max-price", type=float, help="Plafond $/h pour une offre on-demand")
    parser.add_argument("--min-vram-gib", type=float, help="VRAM minimale par GPU (GiB)")
    parser.add_argument("--min-total-vram-gib", type=float, help="VRAM totale minimale (GiB)")
    parser.add_argument("--num-gpus", type=int, help="Nombre minimal de GPU")
    parser.add_argument("--gpu-name", help="Nom exact de GPU souhaité")
    parser.add_argument("--location", help="Filtre pays/région (champ geolocation)")
    parser.add_argument("--order-by", default="dph_total", help="Champ de tri lors de la recherche d'offres")
    parser.add_argument("--order-direction", choices=["asc", "desc"], default="asc", help="Ordre de tri")
    parser.add_argument("--offer-limit", type=int, default=20, help="Nombre max d'offres renvoyées par la recherche")
    parser.add_argument("--remote-workspace", default="/workspace", help="Répertoire distant de travail")
    parser.add_argument("--remote-log-dir", default="logs", help="Sous-dossier distant recevant les logs")
    parser.add_argument("--git-url", help="URL git du dépôt à cloner")
    parser.add_argument("--git-branch", help="Branche git à checkout")
    parser.add_argument("--python-bin", default="python3", help="Binaire Python utilisé sur l'instance")
    parser.add_argument("--venv-name", default=".venv", help="Nom du dossier de venv distant")
    parser.add_argument("--no-venv", action="store_true", help="Ne pas créer/activer de venv")
    parser.add_argument("--dry-run", action="store_true", help="Afficher le script onstart et quitter")
    parser.add_argument("--skip-wait", action="store_true", help="Ne pas patienter jusqu'au statut RUNNING")
    parser.add_argument(
        "--extra-setup",
        action="append",
        default=[],
        help="Commande shell supplémentaire insérée avant la création du venv (peut être répétée)",
    )
    parser.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="Arguments passés à model/train_model.py (placer -- avant)",
    )
    return parser.parse_args(argv)


def build_detector_command(train_args: List[str], python_bin: str) -> str:
    args = list(train_args)
    if args and args[0] == "--":
        args = args[1:]
    command_parts = [python_bin, "model/train_model.py"] + args
    return shell_join(command_parts)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    git_url = args.git_url
    git_branch = args.git_branch
    if not git_url or not git_branch:
        detected_url, detected_branch = detect_git_remote(PROJECT_ROOT)
        git_url = git_url or detected_url
        git_branch = git_branch or detected_branch

    if not git_url:
        git_url = DEFAULT_GIT_URL
    if not git_branch:
        git_branch = "main"

    repo_name = derive_repo_name(git_url)
    train_command = build_detector_command(args.train_args or [], args.python_bin)

    use_venv = not args.no_venv
    extra_setup: List[str] = [
        "apt-get update",
        "apt-get install -y git libgl1 libglib2.0-0",
    ]
    if args.extra_setup:
        extra_setup.extend(args.extra_setup)
    if use_venv:
        extra_setup.append(
            "if ! python3 -m venv --help >/dev/null 2>&1; then apt-get install -y python3-venv; fi"
        )
        extra_setup.append(f"VENV_DIR={shlex.quote(args.venv_name)}")
    else:
        extra_setup.append("VENV_DIR=")

    dataset_commands = [f"{shlex.quote(args.python_bin)} tools/download_card_assets.py"]

    onstart = build_onstart_script(
        git_url=git_url,
        git_branch=git_branch,
        remote_workspace=args.remote_workspace,
        repo_name=repo_name,
        use_venv=use_venv,
        venv_name=args.venv_name,
        train_command=train_command,
        extra_setup=extra_setup,
        pre_train_commands=dataset_commands,
        log_dir=args.remote_log_dir,
    )

    if args.dry_run:
        print("--- Onstart script ---")
        print(onstart)
        return 0

    client = VastAIClient(api_key=args.api_key, base_url=args.base_url)

    if args.ask_id is not None:
        candidate_offers = [{"id": args.ask_id}]
    else:
        query = build_query(args)
        offers = client.search_offers(query)
        if not offers:
            raise SystemExit("Aucune offre Vast.ai ne correspond aux critères fournis.")
        candidate_offers = offers
        print("Offre retenue:")
        keys = ("id", "gpu_name", "num_gpus", "dph_total", "gpu_total_ram", "geolocation")
        print(json.dumps({k: offers[0].get(k) for k in keys}, indent=2))

    base_payload: Dict[str, Any] = {
        "client_id": "me",
        "image": args.image,
        "disk": args.disk,
        "label": args.label,
        "onstart": onstart,
        "env": {},
        "runtype": "ssh",
        "force": False,
        "cancel_unavail": True,
    }
    if args.bid_price is not None:
        base_payload["price"] = float(args.bid_price)
    else:
        base_payload["price"] = None

    last_error: Optional[Exception] = None
    instance_id: Optional[int] = None

    for idx, offer in enumerate(candidate_offers, start=1):
        ask_id = int(offer["id"])
        payload = dict(base_payload)
        print(f"Tentative de provisioning avec l'offre {ask_id} (essai {idx}/{len(candidate_offers)})")
        try:
            response = client.create_instance(ask_id, payload)
        except VastAIError as exc:
            last_error = exc
            message = str(exc)
            if "no_such_ask" in message or "not available" in message:
                print(f"Offre {ask_id} indisponible, on tente la suivante...")
                continue
            raise

        instance_id = response.get("new_contract")
        if instance_id is None:
            raise RuntimeError(f"Création d'instance inattendue: {response}")
        break

    if instance_id is None:
        if last_error is not None:
            raise last_error
        raise SystemExit("Impossible de provisionner une instance Vast.ai avec les offres proposées.")

    print(f"Instance Vast.ai créée: ID={instance_id}")

    if args.skip_wait:
        return 0

    info = client.wait_for_instance_running(int(instance_id))
    ssh_host = info.get("ssh_host")
    ssh_port = info.get("ssh_port")
    if ssh_host and ssh_port:
        print(f"Connexion SSH: ssh root@{ssh_host} -p {ssh_port}")
    else:
        print("Coordonnées SSH non disponibles immédiatement. Consultez la console Vast.ai si besoin.")
    print(
        "Entraînement lancé via onstart. Logs:" ,
        f"{args.remote_workspace}/{repo_name}/{args.remote_log_dir}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

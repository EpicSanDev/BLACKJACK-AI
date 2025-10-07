"""Surveille des instances Vast.ai, rapatrie les artefacts puis détruit les contrats."""

from __future__ import annotations

import argparse
import base64
import os
import shlex
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from tools.vast_train import (  # noqa: E402
    DEFAULT_BASE_URL,
    DEFAULT_GIT_URL,
    VastAIClient,
    VastAIError,
    derive_repo_name,
)


TERMINATED_STATUSES = {"terminated", "canceled", "dead"}


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{timestamp()}] {message}")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Monitore une ou plusieurs instances Vast.ai, télécharge un artefact "
            "et détruit automatiquement l'instance lorsque l'artefact est stable."
        )
    )
    parser.add_argument("--api-key", default=os.getenv("VAST_API_KEY"), help="Clé API Vast.ai")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="URL de base de l'API Vast.ai")
    parser.add_argument(
        "--instance-id",
        dest="instance_ids",
        action="append",
        type=int,
        help="ID d'instance à surveiller (peut être répété)",
    )
    parser.add_argument(
        "--label",
        help="Label d'instance à filtrer lorsqu'aucun --instance-id n'est fourni",
    )
    parser.add_argument(
        "--remote-root",
        help="Dossier distant contenant le dépôt (ex: /workspace/BLACKJACK-AI)",
    )
    parser.add_argument(
        "--remote-workspace",
        default="/workspace",
        help="Répertoire distant racine où le dépôt est cloné",
    )
    parser.add_argument(
        "--git-url",
        help="URL git utilisée pour déduire le nom du dépôt distant (fallback vast_train)",
    )
    parser.add_argument(
        "--artifact-path",
        default="model/advanced_policy.json",
        help="Chemin (relatif au dépôt ou absolu) de l'artefact à rapatrier",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Répertoire local où sauvegarder les artefacts",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=45.0,
        help="Intervalle (s) entre deux vérifications",
    )
    parser.add_argument(
        "--stable-polls",
        type=int,
        default=2,
        help="Nombre de vérifications consécutives avec la même taille de fichier avant collecte",
    )
    parser.add_argument(
        "--download-log",
        action="store_true",
        help="Télécharger aussi le dernier log d'entraînement (pattern logs/train-*.log)",
    )
    parser.add_argument(
        "--log-pattern",
        default="logs/train-*.log",
        help="Motif (relatif au dépôt) pour localiser le log à rapatrier",
    )
    parser.add_argument(
        "--keep-instance",
        action="store_true",
        help="Ne pas détruire l'instance après récupération",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Afficher les instances visées sans lancer de commandes distantes",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def resolve_remote_root(args: argparse.Namespace) -> str:
    if args.remote_root:
        return args.remote_root
    git_url = args.git_url or DEFAULT_GIT_URL
    repo_name = derive_repo_name(git_url)
    return str(Path(args.remote_workspace).joinpath(repo_name))


def resolve_remote_path(remote_root: str, path: str) -> str:
    if Path(path).is_absolute():
        return path
    return str(Path(remote_root) / path)


def select_instances(client: VastAIClient, args: argparse.Namespace) -> List[Dict[str, object]]:
    if args.instance_ids:
        selected: List[Dict[str, object]] = []
        for instance_id in args.instance_ids:
            info = client.get_instance(int(instance_id))
            info.setdefault("id", int(instance_id))
            selected.append(info)
        return selected

    rows = client.list_instances()
    if args.label:
        rows = [row for row in rows if row.get("label") == args.label]
    if not rows:
        raise SystemExit("Aucune instance ne correspond aux critères fournis.")
    return rows


def run_remote_text(client: VastAIClient, instance_id: int, command: str) -> str:
    response = client.run_remote_command(instance_id, command)
    output = response.get("output", "")
    return output.strip()


def check_artifact(client: VastAIClient, instance_id: int, artifact_path: str) -> Optional[Tuple[int, int]]:
    quoted = shlex.quote(artifact_path)
    command = f"bash -lc 'if [ -f {quoted} ]; then stat -c \"%s %Y\" {quoted}; else echo __MISSING__; fi'"
    try:
        result = run_remote_text(client, instance_id, command)
    except VastAIError:
        return None
    if not result or "__MISSING__" in result:
        return None
    parts = result.split()
    if len(parts) < 2:
        return None
    try:
        size = int(parts[0])
        mtime = int(parts[1])
    except ValueError:
        return None
    return size, mtime


def fetch_remote_file(client: VastAIClient, instance_id: int, remote_path: str) -> bytes:
    script = """
python3 - <<'PY'
import base64
from pathlib import Path
path = Path(%r)
data = path.read_bytes()
print(base64.b64encode(data).decode('ascii'))
PY
""" % (remote_path,)
    payload = run_remote_text(client, instance_id, script)
    try:
        return base64.b64decode(payload.encode("ascii"))
    except Exception as exc:  # pragma: no cover - protection
        raise VastAIError(f"Impossible de décoder '{remote_path}': {exc}") from exc


def locate_latest_log(client: VastAIClient, instance_id: int, remote_root: str, pattern: str) -> Optional[str]:
    script = """
python3 - <<'PY'
from pathlib import Path
root = Path(%r)
pattern = %r
paths = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
print(paths[0] if paths else "")
PY
""" % (remote_root, pattern)
    result = run_remote_text(client, instance_id, script)
    return result or None


def save_file(content: bytes, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    return destination


def monitor_instance(
    client: VastAIClient,
    instance: Dict[str, object],
    *,
    artifact_path: str,
    output_dir: Path,
    poll_interval: float,
    stable_polls: int,
    download_log: bool,
    log_pattern: str,
    keep_instance: bool,
    remote_root: str,
) -> None:
    instance_id = int(instance.get("id") or instance.get("instance_id") or instance.get("contract") or 0)
    if not instance_id:
        raise SystemExit(f"Impossible de déterminer l'ID d'instance depuis {instance}.")

    label = instance.get("label") or "(sans label)"
    log(f"Instance {instance_id} détectée (label={label})")

    last_status: Optional[str] = None
    stable_signature: Optional[Tuple[int, int]] = None
    consecutive_matches = 0

    while True:
        try:
            info = client.get_instance(instance_id)
        except VastAIError as exc:
            log(f"Erreur de récupération de l'instance {instance_id}: {exc}")
            time.sleep(poll_interval)
            continue

        status = (info.get("actual_status") or info.get("status") or "").lower()
        if status != last_status:
            log(f"Statut instance -> {status}")
            last_status = status

        if status in TERMINATED_STATUSES:
            log(f"Instance {instance_id} arrêtée avant récupération de l'artefact.")
            return

        signature = check_artifact(client, instance_id, artifact_path)
        if signature:
            size, mtime = signature
            log(f"Artefact détecté (taille={size} o, mtime={mtime})")
            if signature == stable_signature:
                consecutive_matches += 1
            else:
                stable_signature = signature
                consecutive_matches = 1

            if consecutive_matches >= stable_polls:
                log("Artefact stable, lancement de la récupération.")
                data = fetch_remote_file(client, instance_id, artifact_path)
                filename = Path(artifact_path).name
                timestamp_suffix = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
                destination = output_dir / f"{instance_id}-{timestamp_suffix}-{filename}"
                save_file(data, destination)
                log(f"Artefact sauvegardé dans {destination}")

                if download_log:
                    remote_log = locate_latest_log(client, instance_id, remote_root, log_pattern)
                    if remote_log:
                        try:
                            log_bytes = fetch_remote_file(client, instance_id, remote_log)
                        except VastAIError as exc:
                            log(f"Impossible de récupérer le log {remote_log}: {exc}")
                        else:
                            log_dest = destination.with_suffix(".log")
                            save_file(log_bytes, log_dest)
                            log(f"Log sauvegardé dans {log_dest}")
                    else:
                        log("Aucun log trouvé correspondant au motif fourni.")

                if not keep_instance:
                    try:
                        client.destroy_instance(instance_id)
                    except VastAIError as exc:
                        log(f"Erreur lors de la destruction de l'instance {instance_id}: {exc}")
                    else:
                        log(f"Instance {instance_id} détruite.")
                return
        else:
            consecutive_matches = 0

        time.sleep(poll_interval)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)

    if not args.api_key:
        raise SystemExit("Aucune clé API fournie (argument --api-key ou variable VAST_API_KEY)")

    output_dir = args.output_dir.expanduser()
    remote_root = resolve_remote_root(args)
    artifact_path = resolve_remote_path(remote_root, args.artifact_path)

    client = VastAIClient(api_key=args.api_key, base_url=args.base_url)
    instances = select_instances(client, args)

    if args.dry_run:
        for info in instances:
            log(f"Dry-run: instance {info.get('id')} label={info.get('label')}")
        return 0

    for info in instances:
        monitor_instance(
            client,
            info,
            artifact_path=artifact_path,
            output_dir=output_dir,
            poll_interval=args.poll_interval,
            stable_polls=args.stable_polls,
            download_log=args.download_log,
            log_pattern=args.log_pattern,
            keep_instance=args.keep_instance,
            remote_root=remote_root,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())

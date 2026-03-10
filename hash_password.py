from __future__ import annotations

import argparse
import getpass

from app.auth import hash_password


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera un hash PBKDF2 para YOLO_WS_AUTH_PASSWORD_HASH."
    )
    parser.add_argument(
        "--password",
        type=str,
        default="",
        help="Contrasena en texto plano. Si no se envia, se pedira por consola.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    password = args.password or getpass.getpass("Password: ")
    if not password:
        raise SystemExit("La contrasena no puede estar vacia.")
    print(hash_password(password))


if __name__ == "__main__":
    main()


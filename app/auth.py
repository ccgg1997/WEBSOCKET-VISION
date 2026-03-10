from __future__ import annotations

import hashlib
import os
import secrets


DEFAULT_ITERATIONS = 390000


def hash_password(password: str, iterations: int = DEFAULT_ITERATIONS) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, iterations
    )
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def _parse_hash(encoded_hash: str) -> tuple[int, bytes, bytes]:
    try:
        scheme, iterations_raw, salt_hex, digest_hex = encoded_hash.split("$", 3)
    except ValueError as exc:
        raise ValueError("Formato invalido para YOLO_WS_AUTH_PASSWORD_HASH.") from exc

    if scheme != "pbkdf2_sha256":
        raise ValueError("Solo se soporta el esquema pbkdf2_sha256.")

    return int(iterations_raw), bytes.fromhex(salt_hex), bytes.fromhex(digest_hex)


def verify_password(password: str, encoded_hash: str) -> bool:
    iterations, salt, expected_digest = _parse_hash(encoded_hash)
    computed_digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, iterations
    )
    return secrets.compare_digest(computed_digest, expected_digest)


class AuthManager:
    def __init__(
        self,
        username: str,
        password: str = "",
        password_hash: str = "",
    ) -> None:
        self._username = username
        self._password = password
        self._password_hash = password_hash
        if self._password_hash:
            _parse_hash(self._password_hash)

    def verify(self, username: str, password: str) -> bool:
        if not secrets.compare_digest(username, self._username):
            return False

        if self._password_hash:
            return verify_password(password, self._password_hash)

        return secrets.compare_digest(password, self._password)

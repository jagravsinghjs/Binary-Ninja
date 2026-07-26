"""
security.py
Password hashing with stdlib only (hashlib.pbkdf2_hmac) — no new dependency
to install under a hackathon deadline. PBKDF2-SHA256, 260k iterations
(OWASP's current minimum recommendation), random 16-byte salt per password.

Stored format: "pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>"
so the iteration count can be bumped later without breaking old hashes.
"""

import hashlib
import hmac
import secrets

_ITERATIONS = 260_000


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _ITERATIONS)
    return f"pbkdf2_sha256${_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, iterations, salt_hex, hash_hex = stored.split("$")
        if scheme != "pbkdf2_sha256":
            return False
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except (ValueError, AttributeError):
        return False
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iterations))
    return hmac.compare_digest(actual, expected)


def new_token() -> str:
    return secrets.token_urlsafe(32)

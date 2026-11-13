"""Server-side authentication and rate limiting for the Yodha API.

Standard library only (no new dependencies):
  - Passwords: PBKDF2-SHA256, stored as  pbkdf2_sha256$<iterations>$<salt>$<hash>
  - Tokens:    HMAC-SHA256-signed  <payload>.<signature>  with an expiry (stateless)
  - Limits:    in-memory sliding-window rate limiter, per process

Configuration (environment variables, set as Container Apps secrets):
  YODHA_USERS        comma-separated  email:hash  pairs
  YODHA_AUTH_SECRET  random string (32+ chars) used to sign tokens
  YODHA_TOKEN_TTL    token lifetime in seconds (default 43200 = 12 h)

If YODHA_USERS or YODHA_AUTH_SECRET is missing, every login is refused (fail closed).

Generate a user entry:
  python auth.py hash someone@example.com
"""
import base64
import collections
import getpass
import hashlib
import hmac
import json
import os
import secrets
import sys
import threading
import time

from fastapi import Header, HTTPException, Request

PBKDF2_ITERATIONS = 600_000
MIN_SECRET_LEN = 32


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


# ── Passwords ────────────────────────────────────────────────────────────────
def hash_password(password: str, iterations: int = PBKDF2_ITERATIONS) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"pbkdf2_sha256${iterations}${_b64e(salt)}${_b64e(dk)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters, salt, expected = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), _b64d(salt), int(iters))
        return hmac.compare_digest(dk, _b64d(expected))
    except (ValueError, TypeError):
        return False


# Verified against when the email is unknown, so response time doesn't reveal valid emails.
_DUMMY_HASH = hash_password(secrets.token_hex(16))


def parse_users(raw: str) -> dict:
    users = {}
    for entry in filter(None, (e.strip() for e in (raw or "").split(","))):
        email, sep, pw_hash = entry.partition(":")
        if sep and email.strip() and pw_hash.startswith("pbkdf2_sha256$"):
            users[email.strip().lower()] = pw_hash.strip()
        else:
            print("auth: ignoring malformed YODHA_USERS entry", file=sys.stderr)
    return users


# ── Tokens ───────────────────────────────────────────────────────────────────
def issue_token(email: str, secret: str, ttl: int) -> tuple:
    exp = int(time.time()) + ttl
    payload = _b64e(json.dumps({"sub": email, "exp": exp}, separators=(",", ":")).encode())
    sig = _b64e(hmac.new(secret.encode(), payload.encode(), hashlib.sha256).digest())
    return f"{payload}.{sig}", exp


def verify_token(token: str, secret: str):
    """Returns the email for a valid, unexpired token, else None."""
    try:
        payload, sig = token.split(".")
        expected = _b64e(hmac.new(secret.encode(), payload.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, expected):
            return None
        data = json.loads(_b64d(payload))
        if int(data["exp"]) < time.time():
            return None
        return data["sub"]
    except (ValueError, KeyError, TypeError):
        return None


# ── Rate limiting ────────────────────────────────────────────────────────────
class RateLimiter:
    """Sliding-window limiter. In-memory, so limits apply per replica."""

    def __init__(self, max_calls: int, window_s: float, max_keys: int = 10_000):
        self.max_calls, self.window_s, self.max_keys = max_calls, window_s, max_keys
        self._hits = collections.defaultdict(collections.deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and q[0] <= now - self.window_s:
                q.popleft()
            if len(q) >= self.max_calls:
                return False
            q.append(now)
            if len(self._hits) > self.max_keys:  # bound memory: drop keys with no hit inside the window
                cutoff = now - self.window_s
                for k in [k for k, v in self._hits.items() if not v or v[-1] <= cutoff]:
                    del self._hits[k]
            return True


class BodySizeLimitMiddleware:
    """Reject oversized request bodies before they are read into memory. Field-level limits
    only apply after the whole body has been parsed, and that happens before the login check,
    so without this anyone could exhaust the server's memory with a few huge requests."""

    def __init__(self, app, max_bytes: int):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        declared = dict(scope["headers"]).get(b"content-length", b"")
        if declared.isdigit() and int(declared) > self.max_bytes:
            body = b'{"detail":"Request too large"}'
            await send({"type": "http.response.start", "status": 413,
                        "headers": [(b"content-type", b"application/json"),
                                    (b"content-length", str(len(body)).encode()),
                                    (b"connection", b"close")]})
            await send({"type": "http.response.body", "body": body})
            return
        received = 0

        async def limited_receive():   # covers bodies sent without a Content-Length
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise HTTPException(413, "Request too large")
            return message

        await self.app(scope, limited_receive, send)


def client_ip(request: Request) -> str:
    # Azure Container Apps ingress appends the real client IP as the LAST X-Forwarded-For
    # entry; earlier entries are client-supplied and can be spoofed.
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


# ── Configuration + FastAPI dependencies ─────────────────────────────────────
class AuthConfig:
    def __init__(self):
        self.users = parse_users(os.environ.get("YODHA_USERS", ""))
        self.secret = os.environ.get("YODHA_AUTH_SECRET", "")
        self.ttl = int(os.environ.get("YODHA_TOKEN_TTL", "43200"))
        if len(self.secret) < MIN_SECRET_LEN:
            print(f"auth: YODHA_AUTH_SECRET missing or shorter than {MIN_SECRET_LEN} chars — all logins refused", file=sys.stderr)
            self.secret = ""
        if not self.users:
            print("auth: YODHA_USERS empty — all logins refused", file=sys.stderr)

    @property
    def enabled(self) -> bool:
        return bool(self.users and self.secret)

    def authenticate(self, email: str, password: str) -> bool:
        stored = self.users.get(email.strip().lower())
        ok = verify_password(password, stored or _DUMMY_HASH)
        return ok and stored is not None and self.enabled


config = AuthConfig()
login_limiter = RateLimiter(max_calls=5, window_s=60)    # per IP: brute-force protection
# Per user: cost protection. One voice turn = 1 conversation call + 1 /tts call per sentence.
api_limiter = RateLimiter(max_calls=int(os.environ.get("YODHA_RATE_LIMIT_PER_MIN", "120")), window_s=60)


def require_user(authorization: str = Header(default="")) -> str:
    """FastAPI dependency: validates the Bearer token and applies the per-user rate limit."""
    scheme, _, token = authorization.partition(" ")
    email = verify_token(token, config.secret) if scheme.lower() == "bearer" and config.secret else None
    if not email:
        raise HTTPException(401, "Not authenticated", headers={"WWW-Authenticate": "Bearer"})
    if not api_limiter.allow(email):
        raise HTTPException(429, "Too many requests. Please wait a minute.")
    return email


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "hash":
        pw = getpass.getpass("Password: ")
        if len(pw) < 10:
            sys.exit("Use a password of at least 10 characters.")
        if pw != getpass.getpass("Repeat:   "):
            sys.exit("Passwords do not match.")
        print(f"{sys.argv[2].strip().lower()}:{hash_password(pw)}")
    elif len(sys.argv) == 2 and sys.argv[1] == "secret":
        print(secrets.token_urlsafe(48))
    else:
        sys.exit("usage: python auth.py hash <email>   |   python auth.py secret")

"""Utilitaires HTTP JSON (avec retry/backoff) et lecture .env partagés."""

import http.client
import json
import logging
import math
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_BACKOFF_SECONDS = 1.0
MAX_RETRY_AFTER_SECONDS = 60.0
RETRYABLE_HTTP_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
SENSITIVE_HEADER_NAMES = frozenset(
    {"x-goog-api-key", "x-api-key", "api-key", "authorization", "proxy-authorization"}
)

_default_logger = logging.getLogger(__name__)


def _extract_sensitive_values(headers: dict[str, str]) -> tuple[str, ...]:
    """Extrait les valeurs secrètes (en-têtes d'authentification et jetons Bearer/Basic)."""
    found: set[str] = set()
    for key, raw_value in headers.items():
        if key.lower() not in SENSITIVE_HEADER_NAMES:
            continue
        value = raw_value.strip()
        if not value:
            continue
        found.add(value)
        if " " in value:
            _, _, token = value.partition(" ")
            cleaned_token = token.strip()
            if cleaned_token:
                found.add(cleaned_token)
    return tuple(sorted(found, key=len, reverse=True))


def _exception_contains_secret(exc: BaseException, secrets: tuple[str, ...]) -> bool:
    """Indique si l'exception brute contient un secret susceptible de fuiter dans un traceback."""
    if not secrets:
        return False
    candidates = [str(exc)]
    for attr in ("reason", "msg"):
        value = getattr(exc, attr, None)
        if value is not None:
            candidates.append(str(value))
    return any(secret in text for secret in secrets for text in candidates)


def sanitize_error_message(
    message: str,
    secret: str | Iterable[str] | None = None,
) -> str:
    """Masque un ou plusieurs secrets avant d'exposer ou journaliser un message d'erreur."""
    if secret is None:
        return message
    secrets: tuple[str, ...]
    if isinstance(secret, str):
        secrets = (secret,) if secret else ()
    else:
        secrets = tuple(sorted({s for s in secret if s}, key=len, reverse=True))
    sanitized = message
    for value in secrets:
        if value in sanitized:
            sanitized = sanitized.replace(value, "***")
    return sanitized


def extract_http_error_detail(exc: urllib.error.HTTPError) -> str:
    """Extrait un détail lisible depuis le corps d'une réponse HTTPError et ferme le flux."""
    try:
        raw_body = exc.read().decode("utf-8", errors="replace").strip()
    except (OSError, AttributeError, ValueError):
        return ""
    finally:
        try:
            exc.close()
        except Exception:
            pass
    if not raw_body:
        return ""
    try:
        parsed = json.loads(raw_body)
    except json.JSONDecodeError:
        return raw_body[:300]
    if isinstance(parsed, dict):
        error_obj = parsed.get("error")
        if isinstance(error_obj, dict) and isinstance(error_obj.get("message"), str):
            return error_obj["message"]
    return raw_body[:300]


def _parse_retry_after_seconds(exc: urllib.error.HTTPError) -> float | None:
    """Lit un délai numérique Retry-After (en secondes), plafonné à MAX_RETRY_AFTER_SECONDS."""
    headers = getattr(exc, "headers", None)
    if headers is None:
        return None
    raw_value = headers.get("Retry-After")
    if not isinstance(raw_value, str):
        return None
    try:
        seconds = float(raw_value.strip())
    except ValueError:
        return None
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return min(seconds, MAX_RETRY_AFTER_SECONDS)


def request_json(
    url: str,
    payload: dict[str, Any] | None = None,
    extra_headers: dict[str, str] | None = None,
    timeout: int = 60,
    *,
    error_cls: type[Exception] = RuntimeError,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    logger_instance: logging.Logger | None = None,
) -> dict[str, Any]:
    """Envoie une requête HTTP GET ou POST JSON avec retry sur erreurs transitoires."""
    if isinstance(timeout, bool) or timeout <= 0:
        raise error_cls("timeout doit être un entier strictement positif.")
    if isinstance(max_retries, bool) or max_retries < 0:
        raise error_cls("max_retries doit être un entier supérieur ou égal à 0.")
    if (
        isinstance(backoff_seconds, bool)
        or not math.isfinite(backoff_seconds)
        or backoff_seconds < 0
    ):
        raise error_cls("backoff_seconds doit être un nombre fini supérieur ou égal à 0.")

    log = logger_instance or _default_logger

    try:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
    except (TypeError, ValueError) as exc:
        raise error_cls(f"Impossible de sérialiser le payload JSON : {exc}") from exc

    headers = {"Content-Type": "application/json"} if data is not None else {}
    if extra_headers:
        headers.update(extra_headers)
    secrets = _extract_sensitive_values(headers)

    raw_response = ""
    for attempt in range(max_retries + 1):
        req = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method="POST" if data is not None else "GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw_response = resp.read().decode("utf-8")
            break
        except urllib.error.HTTPError as exc:
            retry_after = _parse_retry_after_seconds(exc)
            detail = sanitize_error_message(extract_http_error_detail(exc), secrets)
            suffix = f" ({detail})" if detail else ""
            if exc.code in RETRYABLE_HTTP_STATUS_CODES and attempt < max_retries:
                backoff_delay = backoff_seconds * (2**attempt)
                delay = max(backoff_delay, retry_after) if retry_after is not None else backoff_delay
                log.warning(
                    "Erreur HTTP transitoire %s (tentative %d/%d), nouvelle tentative dans %.1fs.",
                    exc.code,
                    attempt + 1,
                    max_retries + 1,
                    delay,
                )
                sleep(delay)
                continue
            raise error_cls(f"Erreur HTTP {exc.code}{suffix}.") from (
                None if _exception_contains_secret(exc, secrets) else exc
            )
        except urllib.error.URLError as exc:
            reason = sanitize_error_message(str(exc.reason), secrets)
            if attempt < max_retries:
                delay = backoff_seconds * (2**attempt)
                log.warning(
                    "Erreur réseau (tentative %d/%d) : %s — nouvelle tentative dans %.1fs.",
                    attempt + 1,
                    max_retries + 1,
                    reason,
                    delay,
                )
                sleep(delay)
                continue
            raise error_cls(f"Erreur réseau : {reason}") from (
                None if _exception_contains_secret(exc, secrets) else exc
            )
        except TimeoutError as exc:
            if attempt < max_retries:
                delay = backoff_seconds * (2**attempt)
                log.warning(
                    "Timeout HTTP (%ds, tentative %d/%d), nouvelle tentative dans %.1fs.",
                    timeout,
                    attempt + 1,
                    max_retries + 1,
                    delay,
                )
                sleep(delay)
                continue
            raise error_cls(f"Délai d'attente dépassé ({timeout}s).") from exc
        except http.client.HTTPException as exc:
            detail = sanitize_error_message(str(exc), secrets)
            if attempt < max_retries:
                delay = backoff_seconds * (2**attempt)
                log.warning(
                    "Erreur protocole HTTP (tentative %d/%d) : %s — nouvelle tentative dans %.1fs.",
                    attempt + 1,
                    max_retries + 1,
                    detail,
                    delay,
                )
                sleep(delay)
                continue
            raise error_cls(f"Erreur protocole HTTP : {detail}") from (
                None if _exception_contains_secret(exc, secrets) else exc
            )
        except (OSError, UnicodeDecodeError) as exc:
            detail = sanitize_error_message(str(exc), secrets)
            raise error_cls(f"Erreur d'entrée/sortie : {detail}") from (
                None if _exception_contains_secret(exc, secrets) else exc
            )

    try:
        decoded = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise error_cls(f"Réponse HTTP non JSON : {exc}") from exc

    if not isinstance(decoded, dict):
        raise error_cls(f"Format de réponse HTTP inattendu : {type(decoded).__name__}.")
    return decoded


def read_env_file_key(
    env_path: Path,
    key_name: str,
    *,
    error_cls: type[Exception] = OSError,
) -> str | None:
    """Lit une variable dans un fichier .env sans modifier os.environ."""
    if not env_path.is_file():
        return None
    try:
        content = env_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise error_cls(
            f"Impossible de lire le fichier d'environnement {env_path} : {exc}"
        ) from exc

    found: str | None = None
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("export ", "export\t")):
            line = line[6:].lstrip()
        if "=" not in line:
            continue
        name, raw_value = line.split("=", 1)
        if name.strip() != key_name:
            continue
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        found = value.strip()
    return found

"""OpenAI client construction for the CVE-to-ATT&CK pipeline.

Configuration precedence
------------------------
1. Environment variables.
2. An optional JSON secrets file (``secrets.json`` by default).

The module deliberately does not contain a hard-coded proxy, API key, model,
or endpoint.  Standard ``HTTP_PROXY``/``HTTPS_PROXY`` variables are honoured
by HTTPX when ``OPENAI_TRUST_ENV`` is true (the default).  An explicit
``OPENAI_PROXY`` value takes precedence when supplied.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


_DEFAULT_TIMEOUT_SECONDS = 300.0
_DEFAULT_CONNECT_TIMEOUT_SECONDS = 60.0
_DEFAULT_MAX_RETRIES = 2
_DEFAULT_SECRETS_FILE = "secrets.json"


def _parse_bool(value: Optional[str], *, default: bool) -> bool:
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"Invalid boolean value {value!r}; expected one of "
        "true/false, yes/no, on/off, or 1/0."
    )


def _parse_float(name: str, value: Optional[str], *, default: float) -> float:
    if value is None or not value.strip():
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {value!r}.") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than zero, got {parsed}.")
    return parsed


def _parse_int(name: str, value: Optional[str], *, default: int) -> int:
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}.") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be zero or greater, got {parsed}.")
    return parsed


def _load_optional_secrets(path: Path) -> Dict[str, Any]:
    """Load a JSON secrets file when present, without requiring one."""
    if not path.exists():
        return {}
    if not path.is_file():
        raise RuntimeError(f"OpenAI secrets path is not a file: {path}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid JSON in OpenAI secrets file {path}: "
            f"line {exc.lineno}, column {exc.colno}."
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"Unable to read OpenAI secrets file {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise RuntimeError(f"OpenAI secrets file must contain a JSON object: {path}")
    return raw


def _nonempty(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _resolve_config() -> Dict[str, Any]:
    secrets_path = Path(
        os.getenv("OPENAI_SECRETS_FILE", _DEFAULT_SECRETS_FILE)
    ).expanduser()
    secrets = _load_optional_secrets(secrets_path)

    def value(name: str, default: Optional[str] = None) -> Optional[str]:
        env_value = _nonempty(os.getenv(name))
        if env_value is not None:
            return env_value
        secret_value = _nonempty(secrets.get(name))
        return secret_value if secret_value is not None else default

    api_key = value("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. Set it as an environment variable "
            "or place it in the optional secrets file selected by "
            "OPENAI_SECRETS_FILE (default: ./secrets.json)."
        )

    timeout_seconds = _parse_float(
        "OPENAI_TIMEOUT",
        value("OPENAI_TIMEOUT"),
        default=_DEFAULT_TIMEOUT_SECONDS,
    )
    connect_timeout_seconds = _parse_float(
        "OPENAI_CONNECT_TIMEOUT",
        value("OPENAI_CONNECT_TIMEOUT"),
        default=_DEFAULT_CONNECT_TIMEOUT_SECONDS,
    )
    max_retries = _parse_int(
        "OPENAI_MAX_RETRIES",
        value("OPENAI_MAX_RETRIES"),
        default=_DEFAULT_MAX_RETRIES,
    )
    http2 = _parse_bool(value("OPENAI_HTTP2"), default=False)
    trust_env = _parse_bool(value("OPENAI_TRUST_ENV"), default=True)

    return {
        "api_key": api_key,
        "base_url": value("OPENAI_BASE_URL"),
        "proxy": value("OPENAI_PROXY"),
        "timeout_seconds": timeout_seconds,
        "connect_timeout_seconds": connect_timeout_seconds,
        "max_retries": max_retries,
        "http2": http2,
        "trust_env": trust_env,
        "secrets_path": str(secrets_path),
        "secrets_file_present": secrets_path.is_file(),
        "api_key_source": (
            "environment"
            if _nonempty(os.getenv("OPENAI_API_KEY")) is not None
            else "secrets_file"
        ),
    }


def get_openai_runtime_config() -> Dict[str, Any]:
    """Return a provenance-safe configuration summary.

    The API key itself is never returned.  This function can be stored in an
    experiment manifest to document endpoint, timeout, retry, and proxy usage.
    """
    cfg = _resolve_config()
    return {
        "base_url": cfg["base_url"],
        "proxy_configured": bool(cfg["proxy"]),
        "timeout_seconds": cfg["timeout_seconds"],
        "connect_timeout_seconds": cfg["connect_timeout_seconds"],
        "max_retries": cfg["max_retries"],
        "http2": cfg["http2"],
        "trust_env": cfg["trust_env"],
        "secrets_file_present": cfg["secrets_file_present"],
        "api_key_source": cfg["api_key_source"],
    }


def _make_http_client(httpx_module: Any, cfg: Mapping[str, Any]) -> Any:
    timeout = httpx_module.Timeout(
        cfg["timeout_seconds"], connect=cfg["connect_timeout_seconds"]
    )
    kwargs: Dict[str, Any] = {
        "timeout": timeout,
        "http2": cfg["http2"],
        "trust_env": cfg["trust_env"],
    }

    proxy = cfg.get("proxy")
    if proxy:
        # HTTPX >= 0.28 uses ``proxy``; older supported releases used
        # ``proxies``.  Keep a narrow compatibility fallback.
        try:
            return httpx_module.Client(proxy=proxy, **kwargs)
        except TypeError:
            return httpx_module.Client(proxies=proxy, **kwargs)

    return httpx_module.Client(**kwargs)


def get_openai_client() -> Any:
    """Create a configured ``openai.OpenAI`` client.

    Dependencies are imported lazily so non-LLM pipeline stages can run even
    when the optional OpenAI packages are not installed.
    """
    try:
        import httpx  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'httpx'. Install project requirements before "
            "running an LLM stage."
        ) from exc

    try:
        from openai import OpenAI  # type: ignore
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Missing or incompatible dependency 'openai'. Install a current "
            "OpenAI Python package before running an LLM stage."
        ) from exc

    cfg = _resolve_config()
    http_client = _make_http_client(httpx, cfg)

    client_kwargs: Dict[str, Any] = {
        "api_key": cfg["api_key"],
        "http_client": http_client,
        "timeout": cfg["timeout_seconds"],
        "max_retries": cfg["max_retries"],
    }
    if cfg["base_url"]:
        client_kwargs["base_url"] = cfg["base_url"]

    return OpenAI(**client_kwargs)

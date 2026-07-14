from __future__ import annotations

from contextvars import ContextVar


_USER_ID_CTX: ContextVar[str | None] = ContextVar("user_id", default=None)
_USER_CITY_CTX: ContextVar[str | None] = ContextVar("user_city", default=None)


def _normalize(value: str | None) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def set_request_context(user_id: str | None = None, city: str | None = None) -> None:
    _USER_ID_CTX.set(_normalize(user_id))
    _USER_CITY_CTX.set(_normalize(city))


def clear_request_context() -> None:
    _USER_ID_CTX.set(None)
    _USER_CITY_CTX.set(None)


def get_request_user_id() -> str | None:
    return _USER_ID_CTX.get()


def get_request_user_city() -> str | None:
    return _USER_CITY_CTX.get()

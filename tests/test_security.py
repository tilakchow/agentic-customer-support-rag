"""Tests for proxy trust and IP spoofing prevention in rate limiting."""

from fastapi import Request

from agenticrag.api import get_client_ip, get_client_rate_limit_key
from agenticrag.config import Settings


def make_request(client_ip: str, headers: dict[str, str] | None = None) -> Request:
    raw_headers = []
    if headers:
        for k, v in headers.items():
            raw_headers.append((k.lower().encode("latin-1"), v.encode("latin-1")))
    return Request(
        scope={
            "type": "http",
            "client": (client_ip, 12345),
            "headers": raw_headers,
        }
    )


def test_spoofed_forwarded_for_without_trusted_proxy(monkeypatch):
    """When client is not in TRUSTED_PROXIES, X-Forwarded-For is strictly ignored."""
    import agenticrag.api as api_module

    untrusted_settings = Settings(trusted_proxies=[])
    monkeypatch.setattr(api_module, "get_settings", lambda: untrusted_settings)

    # Attacker connects from 203.0.113.1 and claims to be 198.51.100.2
    req = make_request("203.0.113.1", {"x-forwarded-for": "198.51.100.2"})
    assert get_client_ip(req) == "203.0.113.1"
    assert get_client_rate_limit_key(req) == "203.0.113.1"


def test_spoofed_forwarded_for_with_trusted_proxy(monkeypatch):
    """When connecting socket is a trusted proxy, real client IP is resolved from X-Forwarded-For."""
    import agenticrag.api as api_module

    trusted_settings = Settings(trusted_proxies=["10.0.0.1"])
    monkeypatch.setattr(api_module, "get_settings", lambda: trusted_settings)

    # Proxy 10.0.0.1 forwards real client 203.0.113.5
    req = make_request("10.0.0.1", {"x-forwarded-for": "203.0.113.5"})
    assert get_client_ip(req) == "203.0.113.5"


def test_multiple_hops_takes_rightmost_untrusted_ip(monkeypatch):
    """Chain with multiple hops picks the rightmost untrusted IP, ignoring spoofed leftmost IPs."""
    import agenticrag.api as api_module

    # Internal load balancers 10.0.0.1 and 10.0.0.2 are trusted
    trusted_settings = Settings(trusted_proxies=["10.0.0.1", "10.0.0.2"])
    monkeypatch.setattr(api_module, "get_settings", lambda: trusted_settings)

    # Chain: [client-spoofed-ip, real-client-ip, intermediate-trusted-proxy]
    req = make_request("10.0.0.1", {"x-forwarded-for": "1.1.1.1, 198.51.100.42, 10.0.0.2"})
    assert get_client_ip(req) == "198.51.100.42"


def test_settings_trusted_proxies_comma_separated(monkeypatch):
    """TRUSTED_PROXIES parses comma-separated environment variables into list."""
    monkeypatch.setenv("TRUSTED_PROXIES", "127.0.0.1, 10.0.0.1, 172.16.0.1")
    s = Settings()
    assert s.trusted_proxies == ["127.0.0.1", "10.0.0.1", "172.16.0.1"]

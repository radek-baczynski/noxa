from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse


class SSRFError(ValueError):
    pass


def validate_public_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise SSRFError(f"Unsupported URL scheme: {parsed.scheme}")
    if not parsed.hostname:
        raise SSRFError("URL must include a hostname")
    hostname = parsed.hostname.lower()
    if hostname in {"localhost", "127.0.0.1", "::1"}:
        raise SSRFError("Private/localhost URLs are not allowed")
    try:
        addr_infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise SSRFError(f"Could not resolve hostname: {hostname}") from exc
    for info in addr_infos:
        ip_str = info[4][0]
        ip = ipaddress.ip_address(ip_str)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise SSRFError(f"URL resolves to non-public IP: {ip_str}")
    return url


def validate_public_urls(urls: list[str]) -> list[str]:
    return [validate_public_url(u) for u in urls]

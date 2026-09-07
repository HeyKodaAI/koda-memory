"""Secret detection — identifies credentials in user messages.

When a user says something like "my Netflix password is abc123" or
"here's my GitHub token: ghp_xxxx", Koda should route that to the
encrypted Credential Vault instead of plain-text memory.

Detection strategy:
1. Regex patterns for known credential formats (API keys, tokens, etc.)
2. Keyword context analysis ("password is", "my token:", "key for", etc.)
3. Confidence scoring — only vault when we're reasonably sure

This module NEVER logs or stores the detected secret value.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("koda_memory.secret_detector")


@dataclass(frozen=True)
class SecretDetection:
    """Result of scanning a message for secrets."""
    detected: bool
    confidence: float          # 0.0–1.0
    secret_value: str          # The actual credential — handle with care
    secret_type: str           # "api_key", "password", "token", "ssh_key", etc.
    service_hint: str          # Best guess at the service ("github", "netflix", etc.)
    credential_name_hint: str  # Suggested name for the vault entry
    original_message: str      # The full message (for stripping the secret before memory)

    def __repr__(self) -> str:
        """Never print the actual secret value."""
        return (
            f"SecretDetection(detected={self.detected}, confidence={self.confidence:.2f}, "
            f"type={self.secret_type!r}, service={self.service_hint!r})"
        )


# ---------------------------------------------------------------------------
# Known credential patterns (regex → type)
# ---------------------------------------------------------------------------

_CREDENTIAL_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    # Anthropic API keys
    (re.compile(r"\b(sk-ant-api\d{2}-[\w-]{40,})\b"), "api_key", "anthropic"),
    # OpenAI API keys
    (re.compile(r"\b(sk-[A-Za-z0-9]{32,})\b"), "api_key", "openai"),
    # GitHub tokens (classic + fine-grained)
    (re.compile(r"\b(ghp_[A-Za-z0-9]{36,})\b"), "token", "github"),
    (re.compile(r"\b(github_pat_[A-Za-z0-9_]{22,})\b"), "token", "github"),
    # GitHub OAuth
    (re.compile(r"\b(gho_[A-Za-z0-9]{36,})\b"), "oauth_token", "github"),
    # AWS access keys
    (re.compile(r"\b(AKIA[0-9A-Z]{16})\b"), "api_key", "aws"),
    # AWS secret keys (usually 40 chars, base64-ish)
    (re.compile(r"\b([A-Za-z0-9/+=]{40})\b"), "secret_key", "aws"),
    # Slack tokens
    (re.compile(r"\b(xox[bpsar]-[\w-]{10,})\b"), "token", "slack"),
    # Discord tokens
    (re.compile(r"\b([A-Za-z0-9_-]{24}\.[A-Za-z0-9_-]{6}\.[A-Za-z0-9_-]{27,})\b"), "token", "discord"),
    # Generic bearer/API tokens (long alphanumeric strings 40+ chars)
    (re.compile(r"\b([A-Za-z0-9_-]{40,})\b"), "token", "unknown"),
    # SSH private keys
    (re.compile(r"(-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----)", re.MULTILINE), "ssh_key", "ssh"),
]

# ---------------------------------------------------------------------------
# Keyword patterns — "my X password is Y" style phrases
# ---------------------------------------------------------------------------

_KEYWORD_PATTERNS: list[tuple[re.Pattern, str]] = [
    # "my [service] password is [value]"
    (re.compile(
        r"(?:my|the)\s+(\w+)\s+password\s+(?:is|=|:)\s*[\"']?(.+?)[\"']?\s*$",
        re.IGNORECASE | re.MULTILINE,
    ), "password"),
    # "password for [service] is [value]"
    (re.compile(
        r"password\s+(?:for|to)\s+(\w+)\s+(?:is|=|:)\s*[\"']?(.+?)[\"']?\s*$",
        re.IGNORECASE | re.MULTILINE,
    ), "password"),
    # "[service] password: [value]"
    (re.compile(
        r"(\w+)\s+password\s*[:=]\s*[\"']?(.+?)[\"']?\s*$",
        re.IGNORECASE | re.MULTILINE,
    ), "password"),
    # "my [service] token/key is [value]"
    (re.compile(
        r"(?:my|the)\s+(\w+)\s+(?:token|key|api[_\s]?key|secret)\s+(?:is|=|:)\s*[\"']?(.+?)[\"']?\s*$",
        re.IGNORECASE | re.MULTILINE,
    ), "token"),
    # "token/key for [service]: [value]"
    (re.compile(
        r"(?:token|key|api[_\s]?key|secret)\s+(?:for|to)\s+(\w+)\s*[:=]\s*[\"']?(.+?)[\"']?\s*$",
        re.IGNORECASE | re.MULTILINE,
    ), "token"),
    # "pin for [service] is [value]"
    (re.compile(
        r"(?:pin|code)\s+(?:for|to)\s+(\w+)\s+(?:is|=|:)\s*[\"']?(.+?)[\"']?\s*$",
        re.IGNORECASE | re.MULTILINE,
    ), "pin"),
    # "my [service] pin is [value]"
    (re.compile(
        r"(?:my|the)\s+(\w+)\s+(?:pin|code)\s+(?:is|=|:)\s*[\"']?(.+?)[\"']?\s*$",
        re.IGNORECASE | re.MULTILINE,
    ), "pin"),
    # "remember that my [service] login is [value]"
    (re.compile(
        r"(?:remember|store|save|keep)\s+(?:that\s+)?(?:my|the)\s+(\w+)\s+(?:password|token|key|login|credentials?)\s+(?:is|are|=|:)\s*[\"']?(.+?)[\"']?\s*$",
        re.IGNORECASE | re.MULTILINE,
    ), "password"),
]

# Services we recognize in natural language
_SERVICE_ALIASES: dict[str, str] = {
    "google": "google", "gmail": "gmail", "youtube": "google",
    "netflix": "netflix", "hulu": "hulu", "disney": "disney+",
    "amazon": "amazon", "aws": "aws", "prime": "amazon",
    "github": "github", "gitlab": "gitlab", "bitbucket": "bitbucket",
    "slack": "slack", "discord": "discord", "teams": "teams",
    "spotify": "spotify", "apple": "apple", "icloud": "icloud",
    "twitter": "twitter", "x": "twitter",
    "facebook": "facebook", "meta": "meta", "instagram": "instagram",
    "openai": "openai", "anthropic": "anthropic", "claude": "anthropic",
    "chatgpt": "openai",
    "wifi": "wifi", "router": "router",
    "bank": "banking", "chase": "chase", "bofa": "bank_of_america",
    "venmo": "venmo", "paypal": "paypal", "cashapp": "cashapp",
    "ssh": "ssh", "server": "server", "vpn": "vpn",
    "govee": "govee", "homeassistant": "home_assistant", "ha": "home_assistant",
}


class SecretDetector:
    """Scans user messages for credential-like content.

    Usage:
        detector = SecretDetector()
        result = detector.scan("my github token is ghp_abc123def456...")
        if result.detected:
            vault.store(name=result.credential_name_hint, ...)
    """

    def __init__(self, min_confidence: float = 0.6) -> None:
        self._min_confidence = min_confidence

    def scan_all(self, message: str) -> list[SecretDetection]:
        """Return every recognized credential, with overlapping matches deduplicated.

        Store each returned value in a vault before discarding the original.
        Detection is heuristic; this is not a guarantee against every secret format.
        """
        matches = []
        for pattern, kind, service in _CREDENTIAL_PATTERNS:
            for match in pattern.finditer(message):
                value = match.group(1)
                if service == "unknown" and (len(value) < 40 or value.isalpha()):
                    continue
                confidence = 0.95 if service != "unknown" else 0.7
                if confidence >= self._min_confidence:
                    matches.append((match.start(1), match.end(1), confidence, kind, service, value))
        for pattern, kind in _KEYWORD_PATTERNS:
            for match in pattern.finditer(message):
                value = match.group(2).strip()
                service = match.group(1).lower().strip()
                if len(value) >= 3 and 0.85 >= self._min_confidence:
                    matches.append((match.start(2), match.end(2), 0.85, kind,
                                    _SERVICE_ALIASES.get(service, service), value))
        # Prefer a complete containing match over fragments (e.g. a private key).
        selected = []
        for m in sorted(matches, key=lambda x: (x[0], -(x[1] - x[0]), -x[2])):
            if any(m[0] >= old[0] and m[1] <= old[1] for old in selected):
                continue
            selected.append(m)
        return [SecretDetection(True, conf, value, kind, service,
                                f"{service}-{kind}", message)
                for _, _, conf, kind, service, value in selected]

    def scan(self, message: str) -> SecretDetection:
        """Return the first detection; use scan_all to vault multiple credentials."""
        results = self.scan_all(message)
        if results:
            return results[0]
        return SecretDetection(False, 0.0, "", "", "", "", message)

    def redact_all(self, message: str) -> str:
        """Redact every recognized credential without claiming it was stored."""
        values = {d.secret_value for d in self.scan_all(message)}
        for value in sorted(values, key=len, reverse=True):
            message = message.replace(value, "[REDACTED]")
        return message

    def redact(self, message: str, detection: SecretDetection) -> str:
        """Compatibility API: redact the supplied value AND all other detections."""
        redacted = self.redact_all(message)
        if detection.detected and detection.secret_value:
            redacted = redacted.replace(detection.secret_value, "[REDACTED]")
        return redacted

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
    (re.compile(r"(-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----)", re.MULTILINE), "ssh_key", "ssh"),
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

    def scan(self, message: str) -> SecretDetection:
        """Scan a message for credentials.

        Returns a SecretDetection with detected=True if a credential
        was found with confidence >= min_confidence.
        """
        # 1. Try regex patterns first (highest confidence)
        for pattern, secret_type, service in _CREDENTIAL_PATTERNS:
            match = pattern.search(message)
            if match:
                value = match.group(1)
                # Skip very short matches from the generic pattern
                if service == "unknown" and len(value) < 40:
                    continue
                # Skip if it looks like a normal word
                if service == "unknown" and value.isalpha():
                    continue

                confidence = 0.95 if service != "unknown" else 0.7
                name_hint = f"{service}-{secret_type}"

                return SecretDetection(
                    detected=confidence >= self._min_confidence,
                    confidence=confidence,
                    secret_value=value,
                    secret_type=secret_type,
                    service_hint=service,
                    credential_name_hint=name_hint,
                    original_message=message,
                )

        # 2. Try keyword patterns ("my X password is Y")
        for pattern, secret_type in _KEYWORD_PATTERNS:
            match = pattern.search(message)
            if match:
                service_raw = match.group(1).lower().strip()
                value = match.group(2).strip()

                # Don't vault obviously non-secret text
                if len(value) < 3:
                    continue

                service = _SERVICE_ALIASES.get(service_raw, service_raw)
                confidence = 0.85
                name_hint = f"{service}-{secret_type}"

                return SecretDetection(
                    detected=True,
                    confidence=confidence,
                    secret_value=value,
                    secret_type=secret_type,
                    service_hint=service,
                    credential_name_hint=name_hint,
                    original_message=message,
                )

        # 3. No credential detected
        return SecretDetection(
            detected=False,
            confidence=0.0,
            secret_value="",
            secret_type="",
            service_hint="",
            credential_name_hint="",
            original_message=message,
        )

    def redact(self, message: str, detection: SecretDetection) -> str:
        """Replace the secret value in a message with [REDACTED].

        Use this before storing the message in regular memory,
        so the credential doesn't leak into plain-text storage.
        """
        if not detection.detected or not detection.secret_value:
            return message
        return message.replace(detection.secret_value, "[CREDENTIAL_STORED_IN_VAULT]")

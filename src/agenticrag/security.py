"""Security guardrails: prompt injection detection, PII redactor, and canary string leak protection."""

import re

# Unique canary token embedded within internal system instructions
CANARY_TOKEN = "CANARY_GUARD_7f9c2d1b"

# --- PII Redaction Patterns ---
_CREDIT_CARD_REGEX = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
_SSN_REGEX = re.compile(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b")
_EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_REGEX = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b")
_API_KEY_REGEX = re.compile(
    r"\b(?:gsk_[a-zA-Z0-9]{20,}|pcsk_[a-zA-Z0-9]{20,}|tvly-[a-zA-Z0-9]{20,}|Bearer\s+[a-zA-Z0-9_\-\.]{20,})\b"
)


def redact_pii(text: str) -> str:
    """Masks payment card numbers, SSNs, emails, phone numbers, and API tokens."""
    if not text:
        return ""

    redacted = _API_KEY_REGEX.sub("[REDACTED_SECRET]", text)
    redacted = _EMAIL_REGEX.sub("[REDACTED_EMAIL]", redacted)
    redacted = _SSN_REGEX.sub("[REDACTED_SSN]", redacted)
    redacted = _CREDIT_CARD_REGEX.sub("[REDACTED_CARD]", redacted)
    redacted = _PHONE_REGEX.sub("[REDACTED_PHONE]", redacted)
    return redacted


# --- Injection & Jailbreak Detection ---
_INJECTION_PATTERNS = [
    r"ignore\s+.*?(instructions|guidelines|rules|constraints)",
    r"disregard\s+.*?(instructions|rules|prompts|guidelines|constraints)",
    r"forget\s+.*?(safety\s+rules|rules|instructions|prompts)",
    r"(reveal|print|output|dump|show|repeat|display|echo)\s+.*?(system\s+prompt|developer\s+prompt|base\s+instructions|preamble|sys\.prompt|system\s+instructions|developer\s+secrets)",
    r"print\s+everything\s+before\s+the\s+start",
    r"(what|tell\s+me\s+what)\s+.*?(system\s+prompt|developer\s+prompt|base\s+instructions)",
    r"(you\s+are\s+now|act\s+as|in)\s+(dan|jailbreak\s+mode|jailbreak|unfiltered|an\s+ai\s+without\s+rules)",
    r"system\s*override",
    r"emergency\s+(administrative\s+)?(maintenance\s+mode|override)",
    r"simulate\s+a\s+developer\s+console",
    r"translate\s+.*?(system\s+instructions|system\s+prompt|base\s+instructions)",
    r"repeat\s+the\s+words\s+above\s+starting\s+with",
    r"format\s+your\s+.*?(system\s+prompt|initial\s+prompt)",
    r"new\s+roleplay\s+persona",
    r"bypass\s+(mode\s+\d+|guardrails|safety\s+filters)",
    r"canary(\s+token|\s+string|\s+key|\s+guard)?",
    r"<\|im_start\|>",
    r"<\|system\|>",
    r"\[SYSTEM\]",
    r"BEGIN\s+SYSTEM\s+INSTRUCTIONS",
]

_COMPILED_INJECTION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE) for pattern in _INJECTION_PATTERNS
]


def is_prompt_injection(text: str) -> tuple[bool, str]:
    """Detects hostile prompt injections, jailbreak attempts, and system prompt exfiltration."""
    if not text:
        return False, ""

    normalized = text.strip()
    for pattern in _COMPILED_INJECTION_PATTERNS:
        match = pattern.search(normalized)
        if match:
            return True, f"Matched injection pattern: {match.group(0)}"

    return False, ""


def filter_canary_and_leaks(output_text: str) -> str:
    """Detects and suppresses canary token leaks or raw system prompt disclosure."""
    if not output_text:
        return ""

    if CANARY_TOKEN in output_text or "CUSTOMER_SUPPORT_SYSTEM_PROMPT" in output_text:
        return "I am an automated customer support assistant. I cannot disclose internal system prompts or hidden developer instructions."

    # Prevent regurgitating system prompt phrases directly
    leak_signatures = [
        "Internal Security Canary:",
        "CRITICAL SECURITY INSTRUCTION:",
        "Response Guidelines:\n1. Always respond politely",
    ]
    for sig in leak_signatures:
        if sig in output_text:
            return "I am an automated customer support assistant. I cannot disclose internal system prompts or hidden developer instructions."

    return redact_pii(output_text)

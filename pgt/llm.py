# pgt/llm.py
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

RE_ALL = re.compile

# ---- debug: log only once for the first real OpenAI API call ----
_OPENAI_FIRST_CALL_LOGGED = False

EXTRACTION_PIPELINE_VERSION = "llm-extraction-v2.1.0"
PROMPT_VERSION = "cve-extraction-prompt-v2.1.0"
DEFAULT_MODEL = "gpt-4o-mini-2024-07-18"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEED = 20260805
DEFAULT_MAX_COMPLETION_TOKENS = 1400
DEFAULT_MAX_ATTEMPTS = 2
DEFAULT_RETRY_BASE_SECONDS = 0.6



def _log_openai_first_call(msg: str) -> None:
    """Log only once per process to avoid spamming."""
    global _OPENAI_FIRST_CALL_LOGGED
    if _OPENAI_FIRST_CALL_LOGGED:
        return
    print(msg, flush=True)


# ---------------------------
# Optional local configuration loader
# ---------------------------

def _load_secrets() -> Dict[str, str]:
    """
    Load secrets from ./secrets.json (project root).
    This file MUST be gitignored.
    """
    p = Path("secrets.json")
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception:
        pass
    return {}


_SECRETS = _load_secrets()


def _get_cfg(name: str, default: Optional[str] = None) -> Optional[str]:
    # Environment variables are explicit run-time configuration and therefore
    # take precedence over the optional local secrets file.
    v = os.getenv(name)
    if v:
        return v
    if name in _SECRETS and _SECRETS[name]:
        return _SECRETS[name]
    return default


def _get_int_cfg(name: str, default: int) -> int:
    raw = _get_cfg(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _get_float_cfg(name: str, default: float) -> float:
    raw = _get_cfg(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


# ---------------------------
# Generic rule-based fallback (smoke tests only)
# ---------------------------

def _find_evidence(sentences: Dict[str, str], pattern: str, flags=re.IGNORECASE) -> List[str]:
    rx = RE_ALL(pattern, flags)
    return [eid for eid, sentence in sentences.items() if rx.search(sentence)]


def _first_nonempty(*values: Optional[str]) -> Optional[str]:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _rule_based_extract(input_id: str, sentences: Dict[str, str]) -> Dict[str, Any]:
    """Create a conservative, product-agnostic fallback record.

    This path exists only to exercise file and schema interfaces without an API
    call.  Formal experiments disable it.  The rules use broad vulnerability
    language and contain no CVE-family, product, protocol, or technique-specific
    exceptions.
    """

    preconditions: List[Dict[str, Any]] = []
    entry: List[Dict[str, Any]] = []
    vuln_type: List[Dict[str, Any]] = []
    behaviors: List[Dict[str, Any]] = []
    relations: List[Dict[str, Any]] = []
    impacts: List[Dict[str, Any]] = []

    remote_ids = _find_evidence(
        sentences,
        r"remote attacker|network-adjacent|over (?:a|the) network|send(?:ing)? (?:a )?(?:crafted )?(?:request|packet|message|input)|network service|remote request",
    )
    local_ids = _find_evidence(
        sentences,
        r"local attacker|local user|authenticated user|requires? authentication|with (?:local|valid) credentials",
    )
    interaction_ids = _find_evidence(
        sentences,
        r"user interaction|victim (?:opens|visits|loads|views)|persuad(?:e|es|ed|ing) (?:a )?user",
    )
    if remote_ids:
        preconditions.append({
            "condition": "attacker can reach the affected component or submit remote input",
            "evidence_ids": remote_ids,
            "confidence": 0.45,
        })
    elif local_ids:
        preconditions.append({
            "condition": "attacker has local or authenticated access",
            "evidence_ids": local_ids,
            "confidence": 0.45,
        })
    elif interaction_ids:
        preconditions.append({
            "condition": "successful exploitation requires user interaction",
            "evidence_ids": interaction_ids,
            "confidence": 0.45,
        })

    network_entry_ids = _find_evidence(
        sentences,
        r"request|packet|message|header|parameter|network service|endpoint|socket|protocol|API",
    )
    file_entry_ids = _find_evidence(
        sentences,
        r"crafted file|document|archive|image|media file|configuration file|uploaded file|attachment",
    )
    local_entry_ids = _find_evidence(
        sentences,
        r"command line|environment variable|local interface|system call|device input|console",
    )
    if network_entry_ids:
        entry.append({
            "vector": "network or application request",
            "detail": "attacker-controlled request, message, or parameter",
            "evidence_ids": network_entry_ids,
            "confidence": 0.4,
        })
    elif file_entry_ids:
        entry.append({
            "vector": "crafted file or document",
            "detail": "attacker-controlled file content",
            "evidence_ids": file_entry_ids,
            "confidence": 0.4,
        })
    elif local_entry_ids:
        entry.append({
            "vector": "local input interface",
            "detail": "attacker-controlled local input",
            "evidence_ids": local_entry_ids,
            "confidence": 0.4,
        })

    weakness_patterns = [
        ("injection", "command or code injection", r"command injection|code injection|SQL injection|script injection|template injection|expression injection"),
        ("memory_corruption", "memory-safety violation", r"buffer overflow|out-of-bounds|use-after-free|double free|memory corruption|integer overflow"),
        ("path_traversal", "path traversal", r"path traversal|directory traversal"),
        ("deserialization", "unsafe deserialization", r"unsafe deserialization|insecure deserialization|deserializ"),
        ("access_control", "authentication or authorization weakness", r"authentication bypass|authorization bypass|improper access control|missing authorization"),
        ("race_condition", "race condition", r"race condition|time-of-check|TOCTOU"),
        ("input_validation", "improper input validation", r"improper input validation|insufficient validation|fails? to validate|does not validate"),
    ]
    for weakness_type, subtype, pattern in weakness_patterns:
        evidence_ids = _find_evidence(sentences, pattern)
        if evidence_ids:
            vuln_type.append({
                "type": weakness_type,
                "subtype": subtype,
                "evidence_ids": evidence_ids,
                "confidence": 0.5,
            })
            break

    impact_patterns = [
        ("code_execution", "execute arbitrary code|remote code execution|code execution"),
        ("privilege_escalation", "gain elevated privileges|privilege escalation|execute with .* privileges"),
        ("denial_of_service", "denial of service|cause a crash|application crash|resource exhaustion"),
        ("information_disclosure", "information disclosure|sensitive information|read arbitrary files|leak(?:age)? of"),
        ("data_modification", "modify arbitrary|write arbitrary|delete arbitrary|tamper with"),
        ("security_bypass", "bypass authentication|bypass authorization|bypass security"),
    ]
    for impact_type, pattern in impact_patterns:
        evidence_ids = _find_evidence(sentences, pattern)
        if evidence_ids:
            impacts.append({
                "type": impact_type,
                "evidence_ids": evidence_ids,
                "confidence": 0.5,
            })
            break

    exploit_ids = _find_evidence(
        sentences,
        r"allows? (?:a |an )?(?:remote |local |authenticated )?attacker|could allow|may allow|by sending|via (?:a )?crafted|successful exploitation",
    )
    behavior_evidence = list(
        dict.fromkeys(
            exploit_ids
            + (entry[0]["evidence_ids"] if entry else [])
            + (vuln_type[0]["evidence_ids"] if vuln_type else [])
            + (impacts[0]["evidence_ids"] if impacts else [])
        )
    )
    if behavior_evidence:
        impact_name = impacts[0]["type"] if impacts else None
        target = _first_nonempty(
            entry[0].get("detail") if entry else None,
            "affected component",
        )
        behaviors.append({
            "action": "submit crafted input and trigger the vulnerable operation",
            "target": target,
            "impact": impact_name,
            "evidence_ids": behavior_evidence,
            "confidence": 0.35,
        })

    def add_relation(
        src: str,
        relation_type: str,
        dst: str,
        left: Dict[str, Any],
        right: Dict[str, Any],
    ) -> None:
        left_ids = [value for value in left.get("evidence_ids", []) if isinstance(value, str)]
        right_ids = [value for value in right.get("evidence_ids", []) if isinstance(value, str)]
        shared = [value for value in left_ids if value in set(right_ids)]
        evidence_ids = shared or list(dict.fromkeys(left_ids + right_ids))
        if not evidence_ids:
            return
        confidence = min(float(left.get("confidence", 0.5)), float(right.get("confidence", 0.5)))
        relations.append({
            "src": src,
            "type": relation_type,
            "dst": dst,
            "evidence_ids": evidence_ids,
            "confidence": max(0.0, min(1.0, confidence)),
        })

    if preconditions and entry:
        add_relation("P1", "enables", "EN1", preconditions[0], entry[0])
    if preconditions and behaviors:
        add_relation("P1", "enables", "B1", preconditions[0], behaviors[0])
    if entry and vuln_type:
        add_relation("EN1", "characterized_by", "VT1", entry[0], vuln_type[0])
    if entry and behaviors:
        add_relation("EN1", "enables", "B1", entry[0], behaviors[0])
    if vuln_type and behaviors:
        add_relation("VT1", "enables", "B1", vuln_type[0], behaviors[0])
    if behaviors and impacts:
        add_relation("B1", "causes", "I1", behaviors[0], impacts[0])

    return {
        "input_id": input_id,
        "preconditions": preconditions,
        "entry": entry,
        "vuln_type": vuln_type,
        "behaviors": behaviors,
        "relations": relations,
        "impacts": impacts,
        "_validation_errors": [],
        "_fallback_warnings": (
            [] if any((preconditions, entry, vuln_type, behaviors, impacts))
            else ["generic fallback found no structured vulnerability signal"]
        ),
    }


# ---------------------------
# Schema / post-processing
# ---------------------------

def _extraction_schema() -> Dict[str, Any]:
    return {
        "name": "cve_extraction",
        "schema": {
            "type": "object",
            "properties": {
                "preconditions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "condition": {"type": "string"},
                            "evidence_ids": {"type": "array", "items": {"type": "string"}},
                            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        },
                        "required": ["condition", "evidence_ids", "confidence"],
                        "additionalProperties": False,
                    },
                },
                "entry": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "vector": {"type": "string"},
                            "detail": {"type": "string"},
                            "evidence_ids": {"type": "array", "items": {"type": "string"}},
                            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        },
                        "required": ["vector", "detail", "evidence_ids", "confidence"],
                        "additionalProperties": False,
                    },
                },
                "vuln_type": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "subtype": {"type": "string"},
                            "evidence_ids": {"type": "array", "items": {"type": "string"}},
                            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        },
                        "required": ["type", "subtype", "evidence_ids", "confidence"],
                        "additionalProperties": False,
                    },
                },
                "behaviors": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string"},
                            "target": {"type": ["string", "null"]},
                            "impact": {"type": ["string", "null"]},
                            "evidence_ids": {"type": "array", "items": {"type": "string"}},
                            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        },
                        "required": ["action", "target", "impact", "evidence_ids", "confidence"],
                        "additionalProperties": False,
                    },
                },
                "relations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "src": {"type": "string"},
                            "type": {
                                "type": "string",
                                "enum": [
                                    "enables",
                                    "characterized_by",
                                    "causes",
                                    "leads_to",
                                ],
                            },
                            "dst": {"type": "string"},
                            "evidence_ids": {"type": "array", "items": {"type": "string"}},
                            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        },
                        "required": ["src", "type", "dst", "evidence_ids", "confidence"],
                        "additionalProperties": False,
                    },
                },

                "impacts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "evidence_ids": {"type": "array", "items": {"type": "string"}},
                            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        },
                        "required": ["type", "evidence_ids", "confidence"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["preconditions", "entry", "vuln_type", "behaviors", "relations", "impacts"],
            "additionalProperties": False,
        },
        "strict": True,
    }

def _norm(s: Any) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    return s

def _to_snake(s: str) -> str:
    s = _norm(s).lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s

def _as_json_null_if_string_null(v: Any) -> Any:
    """Convert common 'null-like' strings to real None (JSON null)."""
    if v is None:
        return None
    if isinstance(v, str):
        t = v.strip().lower()
        if t in {"null", "none", "nil", "n/a", "na", ""}:
            return None
    return v

def _short_vector(v: Any, max_words: int = 8, max_len: int = 48) -> str:
    """Keep vector as a short phrase; if too long, truncate safely."""
    v = _norm(v)
    if not v:
        return ""
    # Cut at punctuation first
    for sep in [".", ";", "。", "；", "\n"]:
        if sep in v:
            v = v.split(sep, 1)[0].strip()
    if len(v) <= max_len:
        return v
    words = v.split()
    v2 = " ".join(words[:max_words]).strip()
    return v2 if v2 else v[:max_len].strip()

def _normalize_impact_type(raw: Any) -> str:
    hits = _detect_impact_types(raw)
    if not hits:
        return "unspecified"
    # choose a primary label for single-field uses (behaviors.impact)
    priority = [
        "code_execution",
        "privilege_escalation",
        "unauthorized_access",
        "bypass",
        "information_disclosure",
        "integrity_violation",
        "denial_of_service",
    ]
    for p in priority:
        if p in hits:
            return p
    return "unspecified"


def _normalize_vuln_type(type_raw: Any, subtype_raw: Any) -> tuple[str, str]:
    t = _norm(type_raw).lower()
    st = _norm(subtype_raw).lower()

    # If LLM puts impact-ish labels into vuln_type, don't keep them as vuln type.
    if any(k in t for k in ["remote code execution", "arbitrary code execution", "code execution", "rce"]):
        return "unspecified", ""

    # CSRF
    if "csrf" in t or "cross site request forgery" in t:
        return "csrf", "csrf"

    # Injection family
    if "sql injection" in t or "sqli" in t or "sql injection" in st:
        return "injection", "sql_injection"
    if "command injection" in t or "os command" in t or "command injection" in st:
        return "injection", "command_injection"
    if "xss" in t or "cross site scripting" in t or "xss" in st:
        return "injection", "xss"
    if "ssrf" in t or "server side request forgery" in t or "ssrf" in st:
        return "injection", "ssrf"

    # Memory corruption family
    if "use after free" in t or "use-after-free" in t or "use after free" in st:
        return "memory_corruption", "use_after_free"
    if "double free" in t or "double-free" in t or "double free" in st:
        return "memory_corruption", "double_free"
    if "null dereference" in t or "null pointer" in t or "null dereference" in st:
        return "memory_corruption", "null_dereference"
    if any(k in t for k in ["buffer overflow", "heap overflow", "stack overflow", "memory corruption"]):
        return "memory_corruption", _to_snake(st) or "buffer_overflow"

    # Auth / logic
    if any(k in t for k in ["auth bypass", "authentication bypass", "authorization bypass"]) or (t == "bypass" and not st):
        return "auth_bypass", "auth_bypass"

    # DoS
    if "denial of service" in t or t == "dos":
        return "dos", "dos"

    # fallback
    t2 = _to_snake(type_raw) or "unspecified"
    st2 = _to_snake(subtype_raw) if subtype_raw else ""
    return t2, st2


def _detect_impact_types(text: Any) -> set[str]:
    s = _norm(text).lower()
    out: set[str] = set()

    # code execution first (higher priority)
    if any(k in s for k in [
        "execute arbitrary code",
        "execute arbitrary command",
        "execute arbitrary commands",
        "execute arbitrary os command",
        "execute arbitrary os commands",
        "remote code execution",
        "arbitrary code execution",
        "code execution",
        " rce",
        "rce ",
        "os command",
        "os commands",
        "command execution",
    ]):
        out.add("code_execution")

    # privilege escalation
    if any(k in s for k in [
        "privilege escalation",
        "elevation of privilege",
        "gain privileges",
        "administrator privileges",
        "admin privileges",
        "root privilege",
    ]):
        out.add("privilege_escalation")

    # unauthorized access / takeover
    if any(k in s for k in [
        "unauthorized access",
        "admin access",
        "account takeover",
    ]):
        out.add("unauthorized_access")

    # bypass
    if any(k in s for k in [
        "bypass configured filters",
        "bypass content filters",
        "bypass filters",
        "content filter",
        "content filters",
        "allow malicious content",
        "malicious content to pass",
        "bypass",
    ]):
        out.add("bypass")

    # info disclosure
    if any(k in s for k in [
        "information disclosure",
        "sensitive information",
        "leak",
        "exposure",
    ]):
        out.add("information_disclosure")

    # integrity
    if any(k in s for k in [
        "tamper",
        "modify",
        "write arbitrary",
        "delete arbitrary",
        "integrity",
    ]):
        out.add("integrity_violation")

    # dos last (so RCE+DoS keeps both)
    if any(k in s for k in [
        "denial of service",
        "dos",
        "crash",
        "availability",
        "hang",
        "panic",
    ]):
        out.add("denial_of_service")

    return out


# Explicit terminal consequences that may be recovered deterministically from
# the original evidence text when the model encodes the consequence only as a
# Behavior target (for example, action="execute", target="arbitrary code").
# These patterns are generic outcome phrases, not CVE-, product-, protocol-,
# or ATT&CK-technique-specific rules.
_EXPLICIT_IMPACT_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    (
        "code_execution",
        re.compile(
            r"\b(?:remote\s+code\s+execution|arbitrary\s+code\s+execution|"
            r"execute(?:s|d|ing)?\s+arbitrary\s+(?:native\s+)?code|"
            r"execute(?:s|d|ing)?\s+arbitrary\s+(?:os\s+)?commands?|"
            r"command\s+execution)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "privilege_escalation",
        re.compile(
            r"\b(?:privilege\s+escalation|elevation\s+of\s+privilege|"
            r"gain(?:s|ed|ing)?\s+(?:root|administrator|admin|system|elevated|higher)\s+privileges?|"
            r"obtain(?:s|ed|ing)?\s+(?:root|administrator|admin|system|elevated|higher)\s+privileges?|"
            r"execute(?:s|d|ing)?\s+with\s+(?:root|administrator|admin|system|elevated|higher)\s+privileges?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "information_disclosure",
        re.compile(
            r"\b(?:information\s+disclosure|disclos(?:e|es|ed|ing)\s+(?:sensitive|confidential|arbitrary)\s+(?:information|data)|"
            r"expos(?:e|es|ed|ing)\s+(?:sensitive|confidential)\s+(?:information|data)|"
            r"read(?:s|ing)?\s+arbitrary\s+files?|leak(?:s|ed|ing)?\s+(?:sensitive|confidential)\s+(?:information|data))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "integrity_violation",
        re.compile(
            r"\b(?:(?:modify|overwrite|delete|write)(?:s|d|ing)?\s+arbitrary\s+(?:files?|data)|"
            r"tamper(?:s|ed|ing)?\s+with\s+(?:files?|data)|"
            r"data\s+(?:modification|destruction|corruption))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "bypass",
        re.compile(
            r"\b(?:bypass|bypasses|bypassed|bypassing|circumvent|circumvents|circumvented|circumventing)\s+"
            r"(?:authentication|authorization|security|access\s+controls?|filters?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "unauthorized_access",
        re.compile(
            r"\b(?:gain(?:s|ed|ing)?\s+unauthorized\s+access|account\s+takeover|take\s+over\s+(?:an\s+)?account)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "denial_of_service",
        re.compile(
            r"\b(?:denial\s+of\s+service|DoS|"
            r"caus(?:e|es|ed|ing)\s+(?:a\s+)?(?:crash|hang|panic|resource\s+exhaustion))\b",
            re.IGNORECASE,
        ),
    ),
)

_UNKNOWN_IMPACT_RE = re.compile(
    r"\b(?:unknown|unspecified|undetermined)\s+(?:security\s+)?impacts?\b|"
    r"\bimpacts?\s+(?:is|are|remains?|was|were)\s+(?:unknown|unspecified|undetermined)\b",
    re.IGNORECASE,
)


def _explicit_impacts_from_evidence(
    evidence_texts: Optional[Dict[str, str]],
    valid_ids: set[str],
) -> List[Tuple[str, List[str]]]:
    """Return explicitly stated terminal consequences in evidence order.

    The function never maps an outcome to an ATT&CK technique.  It only
    normalizes literal consequence phrases already present in the CVE text.
    Evidence units whose impact is explicitly unknown/unspecified are ignored.
    """
    if not evidence_texts:
        return []

    by_type: Dict[str, List[str]] = {}
    order: List[str] = []
    for evidence_id, raw_text in evidence_texts.items():
        if evidence_id not in valid_ids or not isinstance(raw_text, str):
            continue
        text = raw_text.strip()
        if not text or _UNKNOWN_IMPACT_RE.search(text):
            continue
        for impact_type, pattern in _EXPLICIT_IMPACT_PATTERNS:
            if not pattern.search(text):
                continue
            if impact_type not in by_type:
                by_type[impact_type] = []
                order.append(impact_type)
            if evidence_id not in by_type[impact_type]:
                by_type[impact_type].append(evidence_id)

    return [(impact_type, by_type[impact_type]) for impact_type in order]


_NODE_REF_RE = re.compile(r"^(P|EN|VT|B|I)(\d+)$", re.IGNORECASE)
_RELATION_PREFIX_BY_FIELD = {
    "preconditions": "P",
    "entry": "EN",
    "vuln_type": "VT",
    "behaviors": "B",
    "impacts": "I",
}
_CANONICAL_RELATION_BY_PAIR = {
    ("P", "EN"): "enables",
    ("P", "B"): "enables",
    ("EN", "VT"): "characterized_by",
    ("EN", "B"): "enables",
    ("VT", "B"): "enables",
    ("B", "I"): "causes",
}


def _normalise_node_ref(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    if "::" in text:
        text = text.rsplit("::", 1)[-1]
    text = re.sub(r"[\s_:-]+", "", text)
    replacements = (
        ("PRECONDITION", "P"),
        ("VULNTYPE", "VT"),
        ("BEHAVIOR", "B"),
        ("BEHAVIOUR", "B"),
        ("IMPACT", "I"),
        ("ENTRY", "EN"),
    )
    for long_name, prefix in replacements:
        if text.startswith(long_name):
            text = prefix + text[len(long_name):]
            break
    match = _NODE_REF_RE.fullmatch(text)
    if not match:
        return None
    return f"{match.group(1).upper()}{int(match.group(2))}"


def _node_ref_prefix(ref: str) -> str:
    match = _NODE_REF_RE.fullmatch(ref)
    return match.group(1).upper() if match else ""


def _sanitize_relations(
    raw_relations: Any,
    alias_map: Dict[str, str],
    valid_ids: set[str],
) -> List[Dict[str, Any]]:
    if not isinstance(raw_relations, list):
        return []

    out: List[Dict[str, Any]] = []
    seen = set()
    for relation in raw_relations:
        if not isinstance(relation, dict):
            continue
        raw_src = _normalise_node_ref(relation.get("src", relation.get("source")))
        raw_dst = _normalise_node_ref(relation.get("dst", relation.get("target")))
        if raw_src is None or raw_dst is None:
            continue
        src = alias_map.get(raw_src)
        dst = alias_map.get(raw_dst)
        if src is None or dst is None or src == dst:
            continue

        pair = (_node_ref_prefix(src), _node_ref_prefix(dst))
        canonical_type = _CANONICAL_RELATION_BY_PAIR.get(pair)
        if canonical_type is None:
            continue

        supplied_type = _to_snake(relation.get("type", relation.get("rel", "")))
        if supplied_type not in {"enables", "characterized_by", "causes", "leads_to"}:
            continue

        eids: List[str] = []
        seen_eids = set()
        for eid in relation.get("evidence_ids") or []:
            if isinstance(eid, str) and eid in valid_ids and eid not in seen_eids:
                seen_eids.add(eid)
                eids.append(eid)
        if not eids:
            continue

        try:
            confidence = float(relation.get("confidence", 0.8))
        except (TypeError, ValueError):
            confidence = 0.8
        if confidence != confidence:
            confidence = 0.8
        confidence = max(0.0, min(1.0, confidence))

        key = (src, canonical_type, dst)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "src": src,
            "type": canonical_type,
            "dst": dst,
            "evidence_ids": eids,
            "confidence": confidence,
        })
    return out


def _sanitize_evidence_ids(
    extraction: Dict[str, Any],
    valid_ids: set[str],
    evidence_texts: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Sanitize an extraction dict:
    - Keep only items whose evidence_ids intersect valid_ids
    - Clamp confidence into [0,1]
    - Normalize kind-specific fields
    - Preserve behaviors.impact = None (JSON null) instead of coercing to "unspecified"
    - Recover only explicitly stated terminal consequences from the original evidence text
    - Build B->I relations only when Behavior and Impact share direct evidence
    - Do NOT inject a forced "unspecified" impact
    """

    def _clamp_conf(v: Any, default: float = 0.8) -> float:
        try:
            c = float(v)
        except Exception:
            c = float(default)
        if c != c:  # NaN check
            c = float(default)
        return max(0.0, min(1.0, c))

    def _filter_eids(eids: Any) -> List[str]:
        out: List[str] = []
        for eid in (eids or []):
            if isinstance(eid, str) and eid in valid_ids:
                out.append(eid)
        # keep original order, drop dup
        seen = set()
        dedup = []
        for eid in out:
            if eid not in seen:
                dedup.append(eid)
                seen.add(eid)
        return dedup

    def _filter(items: List[Dict[str, Any]], kind: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for it in (items or []):
            if not isinstance(it, dict):
                continue

            eids = _filter_eids(it.get("evidence_ids"))
            if not eids:
                # no valid evidence -> drop this item entirely
                continue
            it["evidence_ids"] = eids

            # clamp confidence
            it["confidence"] = _clamp_conf(it.get("confidence", 0.8), default=0.8)

            # --- kind-specific normalization ---
            if kind == "entry":
                it["vector"] = _short_vector(it.get("vector"))
                it["detail"] = _norm(it.get("detail"))

            elif kind == "vuln_type":
                t, st = _normalize_vuln_type(it.get("type"), it.get("subtype"))
                # NOTE: if you prefer to drop unspecified vuln_type entirely, do it upstream in prompt;
                # here we keep canonicalized values only.
                it["type"] = t
                it["subtype"] = st

            elif kind == "behaviors":
                # "null" string -> None
                it["target"] = _as_json_null_if_string_null(it.get("target"))
                it["impact"] = _as_json_null_if_string_null(it.get("impact"))
                it["action"] = _norm(it.get("action"))

                # preserve None; only normalize if non-empty
                if it.get("impact") is None:
                    it["impact"] = None
                else:
                    norm_imp = _normalize_impact_type(it.get("impact"))
                    # 关键：把 "unspecified" 当作未知，直接写成 null
                    it["impact"] = None if norm_imp == "unspecified" else norm_imp


            elif kind == "impacts":
                # Keep only canonical types (or None -> drop later)
                it["type"] = _normalize_impact_type(it.get("type"))

            # preconditions / relations are handled elsewhere; if present, keep untouched
            out.append(it)

        return out

    # Preserve original object positions so relation aliases remain valid even
    # when an unsupported item is removed during evidence sanitisation.
    original_collections = {
        field: list(extraction.get(field) or [])
        for field in _RELATION_PREFIX_BY_FIELD
    }
    raw_relations = extraction.get("relations")

    # --- apply filtering ---
    extraction["preconditions"] = _filter(original_collections["preconditions"], "preconditions")
    extraction["entry"] = _filter(original_collections["entry"], "entry")
    extraction["vuln_type"] = _filter(original_collections["vuln_type"], "vuln_type")
    extraction["behaviors"] = _filter(original_collections["behaviors"], "behaviors")
    extraction["impacts"] = _filter(original_collections["impacts"], "impacts")

    alias_map: Dict[str, str] = {}
    for field, prefix in _RELATION_PREFIX_BY_FIELD.items():
        original_items = original_collections[field]
        kept_items = extraction[field]
        new_index_by_identity = {id(item): index for index, item in enumerate(kept_items, start=1)}
        for old_index, item in enumerate(original_items, start=1):
            new_index = new_index_by_identity.get(id(item))
            if new_index is not None:
                alias_map[f"{prefix}{old_index}"] = f"{prefix}{new_index}"
                alias_map[f"{prefix}{new_index}"] = f"{prefix}{new_index}"

    extraction["relations"] = _sanitize_relations(raw_relations, alias_map, valid_ids)

    # ---- post-fix: impacts enrichment (without forcing "unspecified") ----
    try:
        impacts = extraction.get("impacts") or []
        if not isinstance(impacts, list):
            impacts = []

        # normalize existing impacts: drop invalid evidence, clamp confidence, drop empty
        norm_impacts: List[Dict[str, Any]] = []
        for it in impacts:
            if not isinstance(it, dict):
                continue
            t = _normalize_impact_type(it.get("type"))
            eids = _filter_eids(it.get("evidence_ids"))
            if not eids:
                continue

            # If t is still "unspecified", keep it only if you *really* want it.
            # Here: we will keep it temporarily but prefer to remove if we have better signals.
            norm_impacts.append({
                "type": t,
                "evidence_ids": eids,
                "confidence": _clamp_conf(it.get("confidence", 0.8), default=0.8),
            })
        impacts = norm_impacts

        beh = extraction.get("behaviors") or []
        ent = extraction.get("entry") or []

        # collect signals from model fields and from literal terminal-consequence
        # phrases in the original evidence units.  Evidence-derived signals are
        # deliberately generic and require an explicit textual outcome.
        signals: List[Tuple[str, List[str]]] = []
        signals.extend(_explicit_impacts_from_evidence(evidence_texts, valid_ids))

        # behaviors: use impact + detect from action/target
        for b in beh:
            if not isinstance(b, dict):
                continue
            b_eids = _filter_eids(b.get("evidence_ids"))
            if not b_eids:
                continue

            hits = set()
            hits |= _detect_impact_types(b.get("impact"))
            hits |= _detect_impact_types(b.get("action"))
            hits |= _detect_impact_types(b.get("target"))
            for t in hits:
                signals.append((t, b_eids))

        # entry: detect from detail text
        for e in ent:
            if not isinstance(e, dict):
                continue
            e_eids = _filter_eids(e.get("evidence_ids"))
            if not e_eids:
                continue

            hits = _detect_impact_types(e.get("detail"))
            for t in hits:
                signals.append((t, e_eids))

        # Build a set of existing types
        existing_types = set()
        for it in impacts:
            if isinstance(it, dict) and it.get("type"):
                existing_types.add(it["type"])

        # If impacts is empty OR only contains unspecified, we will prefer signals
        only_unspec = (not impacts) or all(
            (isinstance(it, dict) and it.get("type") == "unspecified") for it in impacts
        )
        if only_unspec:
            impacts = [it for it in impacts if isinstance(it, dict) and it.get("type") != "unspecified"]
            existing_types = {it["type"] for it in impacts if isinstance(it, dict) and it.get("type")}

        # add signal-derived impacts
        for t, eids in signals:
            if not t or t == "unspecified":
                continue
            if t in existing_types:
                continue
            impacts.append({"type": t, "evidence_ids": eids, "confidence": 0.95})
            existing_types.add(t)

        # If the model represented a literal terminal consequence as a Behavior
        # target (for example action="execute", target="arbitrary code"), fill
        # Behavior.impact only when the same evidence unit explicitly states the
        # consequence.  This is normalization, not inference.
        explicit_by_type = {
            impact_type: set(evidence_ids)
            for impact_type, evidence_ids in _explicit_impacts_from_evidence(
                evidence_texts, valid_ids
            )
        }
        for behavior in beh:
            if not isinstance(behavior, dict) or behavior.get("impact") is not None:
                continue
            behavior_eids = _filter_eids(behavior.get("evidence_ids"))
            for impact_type, impact_eids in explicit_by_type.items():
                if any(eid in impact_eids for eid in behavior_eids):
                    behavior["impact"] = impact_type
                    break

        # Complete explicit B->I links when both endpoints cite the same evidence.
        # No relation is added merely because a Behavior and Impact coexist.
        relations = extraction.get("relations") or []
        if not isinstance(relations, list):
            relations = []
        existing_relation_keys = {
            (str(rel.get("src")), str(rel.get("type")), str(rel.get("dst")))
            for rel in relations
            if isinstance(rel, dict)
        }
        for b_index, behavior in enumerate(beh, start=1):
            if not isinstance(behavior, dict):
                continue
            behavior_eids = _filter_eids(behavior.get("evidence_ids"))
            for i_index, impact in enumerate(impacts, start=1):
                if not isinstance(impact, dict):
                    continue
                impact_eids = _filter_eids(impact.get("evidence_ids"))
                shared = [eid for eid in behavior_eids if eid in set(impact_eids)]
                key = (f"B{b_index}", "causes", f"I{i_index}")
                if not shared or key in existing_relation_keys:
                    continue
                relations.append({
                    "src": key[0],
                    "type": key[1],
                    "dst": key[2],
                    "evidence_ids": shared,
                    "confidence": min(
                        _clamp_conf(behavior.get("confidence", 0.8)),
                        _clamp_conf(impact.get("confidence", 0.8)),
                    ),
                })
                existing_relation_keys.add(key)

        # IMPORTANT: do NOT inject a default "unspecified" if still empty
        extraction["impacts"] = impacts
        extraction["relations"] = relations

    except Exception:
        # keep whatever impacts we already had
        pass

    return extraction



def _build_messages(input_id: str, sentences: Dict[str, str]) -> List[Dict[str, str]]:
    evidence_lines = "\n".join([f"{eid}: {text}" for eid, text in sentences.items()])
    system = (
        "You are a deterministic cybersecurity information extraction engine.\n"
        "Extract only facts explicitly supported by the supplied CVE evidence units.\n"
        "The output must follow the provided strict JSON schema.\n"
        "\n"
        "Evidence rules:\n"
        "1) Use only evidence identifiers present in the input.\n"
        "2) Cite the smallest set of directly supporting evidence units for every item.\n"
        "3) Do not infer attacker capabilities, mechanisms, targets, or impacts that are not stated.\n"
        "4) If a field is unsupported, return an empty array or JSON null as allowed.\n"
        "\n"
        "Element rules:\n"
        "5) entry.vector is an attack surface, interface, protocol, file, endpoint, or component; "
        "put product/version context in entry.detail.\n"
        "6) behaviors must be atomic: action is a concise verb phrase, target is the acted-on object "
        "or null, and impact is a short supported outcome or null.\n"
        "7) A terminal consequence explicitly stated in the evidence MUST also appear in impacts. "
        "Examples include arbitrary-code or command execution, privilege escalation, information disclosure, "
        "denial of service, arbitrary file/data modification or deletion, security bypass, and unauthorized access.\n"
        "8) Keep action, target, and terminal consequence distinct. For example, in 'execute arbitrary code', "
        "use a Behavior such as action='execute', target='arbitrary code', impact='code_execution', and also "
        "emit an Impact with type='code_execution'.\n"
        "9) If the text says the impact is unknown, unspecified, or undetermined, do not invent an Impact.\n"
        "10) Do not use the literal category 'unspecified'; use an empty list instead.\n"
        "11) Preserve narrative order within each array.\n"
        "\n"
        "Relation rules:\n"
        "12) Refer to extracted items by their output position: P1/P2 for preconditions, "
        "EN1/EN2 for entries, VT1/VT2 for vulnerability types, B1/B2 for behaviors, "
        "and I1/I2 for impacts.\n"
        "13) Emit a relation only when the evidence directly supports the link. Allowed layer links are "
        "P->EN, P->B, EN->VT, EN->B, VT->B, and B->I.\n"
        "14) Use relation type 'enables' for P->EN, P->B, EN->B, and VT->B; "
        "'characterized_by' for EN->VT; and 'causes' for B->I.\n"
        "15) When a Behavior and an Impact are directly linked by the same evidence, emit B->I with type 'causes'.\n"
        "16) Relation evidence_ids must support the connection, not merely one endpoint.\n"
        "17) Confidence values must be between 0 and 1.\n"
    ).replace("\n\n", "\n")

    user = (
        f"input_id: {input_id}\n"
        "Evidence units:\n"
        f"{evidence_lines}\n\n"
        "Return the structured extraction and explicit supported relations.\n"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _prompt_sha256(messages: List[Dict[str, str]]) -> str:
    payload = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------
# Main entry
# ---------------------------

def call_llm_extract(input_id: str, sentences: Dict[str, str]) -> Dict[str, Any]:
    model = _get_cfg("OPENAI_EXTRACTION_MODEL") or _get_cfg("OPENAI_MODEL", DEFAULT_MODEL)
    temperature = _get_float_cfg("OPENAI_EXTRACTION_TEMPERATURE", DEFAULT_TEMPERATURE)
    seed = _get_int_cfg("OPENAI_EXTRACTION_SEED", DEFAULT_SEED)
    max_tokens = _get_int_cfg("OPENAI_EXTRACTION_MAX_TOKENS", DEFAULT_MAX_COMPLETION_TOKENS)
    max_attempts = max(1, _get_int_cfg("OPENAI_EXTRACTION_MAX_ATTEMPTS", DEFAULT_MAX_ATTEMPTS))
    retry_base = max(0.0, _get_float_cfg("OPENAI_EXTRACTION_RETRY_BASE_SECONDS", DEFAULT_RETRY_BASE_SECONDS))
    valid_ids = set(sentences.keys())

    messages = _build_messages(input_id, sentences)
    prompt_hash = _prompt_sha256(messages)
    base_provenance: Dict[str, Any] = {
        "pipeline_version": EXTRACTION_PIPELINE_VERSION,
        "prompt_version": PROMPT_VERSION,
        "requested_model": model,
        "temperature": temperature,
        "seed": seed,
        "max_completion_tokens": max_tokens,
        "response_format": "strict_json_schema",
        "max_attempts": max_attempts,
        "retry_base_seconds": retry_base,
        "prompt_sha256": prompt_hash,
    }

    openai_runtime: Dict[str, Any] = {}
    try:
        from .openai_client import get_openai_client, get_openai_runtime_config
        openai_runtime = get_openai_runtime_config()
        client = get_openai_client()
    except Exception as e:
        fb = _sanitize_evidence_ids(_rule_based_extract(input_id, sentences), valid_ids, sentences)
        fb["_used_llm"] = False
        fb["_runtime_errors"] = [f"llm_client_init_failed: {type(e).__name__}: {e}"]
        fb["_provenance"] = {
            **base_provenance,
            "openai_runtime": openai_runtime,
            "mode": "rule_based_fallback",
            "fallback_reason": "client_initialisation_failed",
        }
        return fb

    schema = _extraction_schema()
    last_err: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                seed=seed,
                response_format={"type": "json_schema", "json_schema": schema},
                max_completion_tokens=max_tokens,
            )
            content = resp.choices[0].message.content or "{}"
            data = json.loads(content)

            extraction: Dict[str, Any] = {
                "input_id": input_id,
                "preconditions": data.get("preconditions", []),
                "entry": data.get("entry", []),
                "vuln_type": data.get("vuln_type", []),
                "behaviors": data.get("behaviors", []),
                "relations": data.get("relations", []),
                "impacts": data.get("impacts", []),
                "_validation_errors": [],
                "_used_llm": True,
            }
            extraction = _sanitize_evidence_ids(extraction, valid_ids, sentences)
            extraction["_provenance"] = {
                **base_provenance,
                "openai_runtime": openai_runtime,
                "mode": "llm",
                "attempts_used": attempt,
                "returned_model": getattr(resp, "model", None),
                "system_fingerprint": getattr(resp, "system_fingerprint", None),
                "response_id": getattr(resp, "id", None),
            }
            return extraction

        except Exception as e:
            last_err = e
            if attempt < max_attempts:
                time.sleep(retry_base * attempt)

    fb = _sanitize_evidence_ids(_rule_based_extract(input_id, sentences), valid_ids, sentences)
    fb["_used_llm"] = False
    fb["_runtime_errors"] = (
        [f"llm_failed: {type(last_err).__name__}: {last_err}"] if last_err else []
    )
    fb["_provenance"] = {
        **base_provenance,
        "openai_runtime": openai_runtime,
        "mode": "rule_based_fallback",
        "attempts_used": max_attempts,
        "fallback_reason": "all_llm_attempts_failed",
    }
    return fb


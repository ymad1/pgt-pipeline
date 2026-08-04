# pgt/llm.py
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

RE_ALL = re.compile

# ---- debug: log only once for the first real OpenAI API call ----
_OPENAI_FIRST_CALL_LOGGED = False


def _log_openai_first_call(msg: str) -> None:
    """Log only once per process to avoid spamming."""
    global _OPENAI_FIRST_CALL_LOGGED
    if _OPENAI_FIRST_CALL_LOGGED:
        return
    print(msg, flush=True)


# ---------------------------
# Local secret loader (NO env var required)
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
    # priority: secrets.json -> env -> default
    if name in _SECRETS and _SECRETS[name]:
        return _SECRETS[name]
    v = os.getenv(name)
    return v if v else default


# ---------------------------
# Rule-based fallback (kept as backup)
# ---------------------------

def _find_evidence(sentences: Dict[str, str], pattern: str, flags=re.IGNORECASE) -> List[str]:
    rx = RE_ALL(pattern, flags)
    hits: List[str] = []
    for eid, s in sentences.items():
        if rx.search(s):
            hits.append(eid)
    return hits


def _rule_based_extract(input_id: str, sentences: Dict[str, str]) -> Dict[str, Any]:
    preconditions = []
    entry = []
    vuln_type = []
    behaviors = []
    relations = []
    impacts = []
    errors: List[str] = []

    ev_jndi = _find_evidence(sentences, r"\bJNDI\b")
    ev_ldap = _find_evidence(sentences, r"\bLDAP\b")
    if ev_jndi or ev_ldap:
        vuln_type.append({
            "type": "injection",
            "subtype": "JNDI/LDAP endpoint control",
            "evidence_ids": sorted(set(ev_jndi + ev_ldap)),
            "confidence": 0.6,
        })

    ev_rce = _find_evidence(sentences, r"execute arbitrary code|remote code execution|code execution")
    if ev_rce:
        impacts.append({
            "type": "code_execution",
            "evidence_ids": ev_rce,
            "confidence": 0.6,
        })

    ev_entry = _find_evidence(sentences, r"configuration|log messages|parameters|SMBv1|server")
    if ev_entry:
        entry.append({
            "vector": "external input surface",
            "detail": "config/logs/params or network service",
            "evidence_ids": ev_entry,
            "confidence": 0.4,
        })

    ev_attacker_ctrl = _find_evidence(sentences, r"attacker controlled|attacker-controlled")
    if ev_attacker_ctrl:
        preconditions.append({
            "condition": "attacker controls an external endpoint / input",
            "evidence_ids": ev_attacker_ctrl,
            "confidence": 0.6,
        })

    ev_crafted = _find_evidence(sentences, r"crafted packets|crafted")
    if ev_crafted:
        preconditions.append({
            "condition": "attacker can send crafted network input",
            "evidence_ids": ev_crafted,
            "confidence": 0.6,
        })

    ev_allows = _find_evidence(sentences, r"\ballows\b")
    ev_via = _find_evidence(sentences, r"\bvia\b")
    if ev_allows and ev_rce:
        ev = sorted(set(ev_allows + ev_rce + ev_via))
        behaviors.append({
            "action": "exploit",
            "target": "vulnerable service/component",
            "impact": "code execution",
            "evidence_ids": ev,
            "confidence": 0.55,
        })

    ev_not_protect = _find_evidence(sentences, r"do not protect|does not protect|not protect")
    if ev_not_protect and (ev_jndi or ev_ldap or ev_attacker_ctrl):
        ev = sorted(set(ev_not_protect + ev_jndi + ev_ldap + ev_attacker_ctrl))
        behaviors.append({
            "action": "trigger jndi lookup / remote fetch",
            "target": "attacker-controlled LDAP/JNDI endpoint",
            "impact": "possible code execution",
            "evidence_ids": ev,
            "confidence": 0.55,
        })

    if ev_rce and not any(b.get("impact") == "code execution" for b in behaviors):
        behaviors.append({
            "action": "exploit",
            "target": None,
            "impact": "code execution",
            "evidence_ids": ev_rce,
            "confidence": 0.35,
        })

    return {
        "input_id": input_id,
        "preconditions": preconditions,
        "entry": entry,
        "vuln_type": vuln_type,
        "behaviors": behaviors,
        "relations": relations,
        "impacts": impacts,
        "_validation_errors": errors,
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
                        "properties": {},
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


from typing import Any, Dict, List, Tuple

def _sanitize_evidence_ids(extraction: Dict[str, Any], valid_ids: set[str]) -> Dict[str, Any]:
    """
    Sanitize an extraction dict:
    - Keep only items whose evidence_ids intersect valid_ids
    - Clamp confidence into [0,1]
    - Normalize kind-specific fields
    - Preserve behaviors.impact = None (JSON null) instead of coercing to "unspecified"
    - Build impacts from signals if present, but do NOT inject a forced "unspecified" impact
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

    # --- apply filtering ---
    extraction["preconditions"] = _filter(extraction.get("preconditions", []), "preconditions")
    extraction["entry"] = _filter(extraction.get("entry", []), "entry")
    extraction["vuln_type"] = _filter(extraction.get("vuln_type", []), "vuln_type")
    extraction["behaviors"] = _filter(extraction.get("behaviors", []), "behaviors")
    extraction["impacts"] = _filter(extraction.get("impacts", []), "impacts")

    # relations: ensure list
    rel = extraction.get("relations")
    if not isinstance(rel, list):
        extraction["relations"] = []
    else:
        extraction["relations"] = rel

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

        # collect signals from behaviors + entry.detail
        signals: List[Tuple[str, List[str]]] = []

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
            impacts.append({"type": t, "evidence_ids": eids, "confidence": 0.8})
            existing_types.add(t)

        # IMPORTANT: do NOT inject a default "unspecified" if still empty
        extraction["impacts"] = impacts

    except Exception:
        # keep whatever impacts we already had
        pass

    return extraction



def _build_messages(input_id: str, sentences: Dict[str, str]) -> List[Dict[str, str]]:
    evidence_lines = "\n".join([f"{eid}: {text}" for eid, text in sentences.items()])
    system = (
        "You are a security information extraction engine.\n"
        "Extract structured fields from CVE evidence sentences into the required JSON schema.\n"
        "\n"
        "Rules:\n"
        "1) Only use evidence_ids that appear in the evidence list (E1, E2, ...).\n"
        "2) Do NOT invent facts. Only extract what is explicitly supported by evidence sentences.\n"
        "3) Evidence alignment MUST be precise:\n"
        "   - For each extracted item, cite the 1-2 most directly supporting evidence sentences.\n"
        "   - Do NOT default to E1 when multiple evidence sentences exist.\n"
        "   - Do NOT cite an evidence sentence unless it explicitly contains the fact.\n"
        "4) entry.vector must describe an attack surface / interface / component (e.g., XML-RPC endpoint, HTTP update channel, TLS handshake, DCCP packet).\n"
        "   - Do NOT use product name or version range as entry.vector.\n"
        "   - Put product/version information in entry.detail instead.\n"
        "5) impacts and vuln_type:\n"
        "   - Avoid using the literal string 'unspecified'.\n"
        "   - If you cannot infer a clean category from evidence, output an empty list for that section.\n"
        "6) behaviors should be atomic and readable:\n"
        "   - action: concise verb phrase (e.g., download, spoof, execute, bypass, decrypt)\n"
        "   - target: the object acted on (or null if truly unknown)\n"
        "   - impact: a short outcome phrase if supported (or null if unknown)\n"
        "7) confidence must be in [0,1].\n"
    )

    user = (
        f"input_id: {input_id}\n"
        "Evidence sentences:\n"
        f"{evidence_lines}\n\n"
        "Return JSON matching the required schema.\n"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ---------------------------
# Main entry
# ---------------------------

def call_llm_extract(input_id: str, sentences: Dict[str, str]) -> Dict[str, Any]:
    model = _get_cfg("OPENAI_MODEL", "gpt-4o-mini")
    valid_ids = set(sentences.keys())

    try:
        from .openai_client import get_openai_client
        client = get_openai_client()
    except Exception as e:
        fb = _rule_based_extract(input_id, sentences)
        fb["_used_llm"] = False
        fb["_validation_errors"] = [f"llm_client_init_failed: {type(e).__name__}: {e}"]
        return fb

    messages = _build_messages(input_id, sentences)
    schema = _extraction_schema()

    last_err: Optional[Exception] = None
    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                response_format={"type": "json_schema", "json_schema": schema},
                max_completion_tokens=1200,
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
            return _sanitize_evidence_ids(extraction, valid_ids)

        except Exception as e:
            last_err = e
            time.sleep(0.6 * (attempt + 1))

    fb = _rule_based_extract(input_id, sentences)
    fb["_used_llm"] = False
    fb["_validation_errors"] = [f"llm_failed: {type(last_err).__name__}: {last_err}"] if last_err else []
    return fb

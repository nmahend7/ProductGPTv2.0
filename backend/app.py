# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from openai import APIError, RateLimitError, APITimeoutError
import os
import json
import re
import time
from collections import defaultdict, deque
from dotenv import load_dotenv

# ---------------------------
# Setup
# ---------------------------
load_dotenv()
app = Flask(__name__)

# CORS (single origin or CSV via env; "*" allowed)
_origins_env = os.getenv("FRONTEND_ORIGINS") or os.getenv("FRONTEND_ORIGIN", "*")
_allowed_origins = "*" if (_origins_env.strip() == "*") else [o.strip() for o in _origins_env.split(",") if o.strip()]

CORS(
    app,
    resources={r"/*": {"origins": _allowed_origins}},
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

@app.after_request
def add_cors_headers(resp):
    origin = request.headers.get("Origin")
    if _allowed_origins == "*":
        resp.headers.setdefault("Access-Control-Allow-Origin", "*")
    elif origin and origin in _allowed_origins:
        resp.headers.setdefault("Access-Control-Allow-Origin", origin)
    resp.headers.setdefault("Vary", "Origin")
    resp.headers.setdefault("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    resp.headers.setdefault("Access-Control-Allow-Headers", "Content-Type, Authorization")
    return resp

@app.route("/health")
def health():
    return {"status": "ok"}, 200

# ---------------------------
# OpenAI client & config
# ---------------------------
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # fast default
MODERATION_MODEL = os.getenv("MODERATION_MODEL", "omni-moderation-latest")

OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "45"))
ENABLE_REPAIR_PASS = os.getenv("ENABLE_REPAIR_PASS", "1") == "1"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    # Fail fast so you don't chase 500s later
    raise RuntimeError("Missing OPENAI_API_KEY env var.")

client = OpenAI(api_key=OPENAI_API_KEY, timeout=OPENAI_TIMEOUT)

MAX_TOKENS_GENERATE = int(os.getenv("MAX_TOKENS_GENERATE", "1500"))
MAX_TOKENS_PRD = int(os.getenv("MAX_TOKENS_PRD", "1700"))
MODEL_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.2"))

# ---------------------------
# Utilities
# ---------------------------
def slugify(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"(^-|-$)", "", s)
    return s[:40] or "item"

def strip_code_fences(text: str) -> str:
    """
    Remove a single leading/trailing fenced block like:
      ```json ... ```
      ```markdown ... ```
      ```md ... ```
      ``` ... ```
    without destroying inline backticks inside the content.
    """
    if not text:
        return ""
    t = text.strip()
    # Leading fence
    m = re.match(r"^```(?:(json|markdown|md)\s*)?\n?", t, flags=re.IGNORECASE)
    if m:
        t = t[m.end():]
        # Trailing fence: last line starting with ```
        t = re.sub(r"\n?```[\s\t]*$", "", t, flags=re.IGNORECASE)
    return t.strip()

# Simple product-ish heuristic
_DOMAIN_HINTS = re.compile(
    r"\b(api|auth|login|signup|user|dashboard|admin|mobile|android|ios|web|frontend|backend|"
    r"database|schema|analytics|billing|payment|subscription|notification|workflow|"
    r"integration|pipeline|microservice|cloud|storage|query|latency|throughput|"
    r"cache|search|design|ui|ux|prototype|wireframe|security|compliance|gdpr|hipaa)\b",
    re.IGNORECASE,
)

def is_meaningful_goal(goal: str) -> bool:
    if not goal or len(goal.strip()) < 12:
        return False
    words = goal.strip().split()
    if len(words) < 3:
        return False
    letters = re.findall(r"[A-Za-z]", goal)
    if len(letters) < 8:
        return False
    if len(words) < 12 and not _DOMAIN_HINTS.search(goal):
        return False
    return True

def normalize_estimate(s: str) -> str:
    if not s:
        return ""
    t = s.strip().lower()
    t = t.replace("–", "-")
    t = t.replace("days", "d").replace("day", "d")
    t = t.replace("hrs", "h").replace("hours", "h").replace("hour", "h")
    t = re.sub(r"\s+", "", t)
    if re.match(r"^\d+(-\d+)?[dh]$", t):
        return t
    return t

def ensure_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        return [x] if x.strip() else []
    return list(x)

def moderate_text(text: str) -> bool:
    try:
        r = client.moderations.create(model=MODERATION_MODEL, input=text)
        return not getattr(r.results[0], "flagged", False)
    except Exception:
        return True  # fail-open

# ---------------------------
# Legacy parser
# ---------------------------
def legacy_parse_response(text):
    epics = []
    current_epic = None
    lines = text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Epic "):
            if current_epic:
                epics.append(current_epic)
            epic_title = line.split(":", 1)[1].strip()
            i += 1
            epic_description = ""
            if i < len(lines):
                epic_description = lines[i].replace("Description:", "").strip()
            current_epic = {
                "epic": epic_title,
                "description": epic_description,
                "stories": []
            }
        elif line and line[0].isdigit() and "Summary:" in line:
            story_summary = line.split("Summary:", 1)[1].strip()
            i += 1
            story_description = ""
            acceptance_criteria = []
            while i < len(lines) and "Acceptance Criteria:" not in lines[i]:
                if "Description:" in lines[i]:
                    story_description = lines[i].split("Description:", 1)[1].strip()
                i += 1
            if i < len(lines) and "Acceptance Criteria:" in lines[i]:
                i += 1
                while i < len(lines) and not lines[i].strip().startswith(tuple("0123456789")) and not lines[i].strip().startswith("Epic "):
                    crit_line = lines[i].strip()
                    if crit_line.startswith("- "):
                        acceptance_criteria.append(crit_line[2:].strip())
                    i += 1
                i -= 1
            current_epic["stories"].append({
                "summary": story_summary,
                "description": story_description,
                "acceptanceCriteria": acceptance_criteria
            })
        i += 1
    if current_epic:
        epics.append(current_epic)
    return epics

# ---------------------------
# Schema validation & normalization
# ---------------------------
ALLOWED_TYPES = {"Design", "Engineering"}
ALLOWED_PRIORITIES = {"P0", "P1", "P2"}
ALLOWED_AREAS = {"Product", "Design", "Engineering"}

def _dedupe_story_ids(epics):
    seen = set()
    counters = defaultdict(int)
    for ep in epics:
        for st in ep["stories"]:
            sid = st["storyId"]
            if sid in seen:
                counters[sid] += 1
                st["storyId"] = f"{sid}-{counters[sid]}"
            seen.add(st["storyId"])

def _prune_dependency_cycles(epics):
    graph = defaultdict(set)
    for ep in epics:
        for st in ep["stories"]:
            for d in st["dependencies"]:
                graph[d].add(st["storyId"])
    indeg = defaultdict(int)
    nodes = set()
    for u, vs in graph.items():
        nodes.add(u)
        for v in vs:
            nodes.add(v)
            indeg[v] += 1
    q = deque([n for n in nodes if indeg[n] == 0])
    visited = set()
    while q:
        n = q.popleft()
        visited.add(n)
        for v in graph.get(n, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    cyclic_nodes = nodes - visited
    if not cyclic_nodes:
        return
    for ep in epics:
        for st in ep["stories"]:
            st["dependencies"] = [d for d in st["dependencies"] if not (d in cyclic_nodes and st["storyId"] in graph[d])]

def _clamp_lengths(story):
    ac = ensure_list(story["acceptanceCriteria"])
    tc = ensure_list(story["testCases"])
    story["acceptanceCriteria"] = ac[:10]
    story["testCases"] = tc[:12]

# --- Heuristics for inference ---
_UI_HINT = re.compile(r"\b(ui|screen|figma|wireframe|visual|prototype|ux|design|mockup|handoff)\b", re.I)
_API_HINT = re.compile(r"\b(api|endpoint|service|controller|handler|sdk|server|schema|db|queue|topic|cron|job|etl|pipeline)\b", re.I)
_AUTH_HINT = re.compile(r"\b(auth|login|sign[- ]?in|oauth|oidc|token|session|mfa|two[- ]?factor)\b", re.I)
_CONNECT_HINT = re.compile(r"\b(pair|bluetooth|connect(ion)?|vehicle\s*(link|pair|connect)|provision(ing)?)\b", re.I)

# Feature tokens
FEATURE_PATTERNS = {
    "lock": re.compile(r"\block(ing)?\b", re.I),
    "unlock": re.compile(r"\bunlock(ing)?\b", re.I),
    "start": re.compile(r"\bstart(ing)?\b", re.I),
    "stop": re.compile(r"\bstop(ping)?\b", re.I),
    "horn": re.compile(r"\bhorn\b", re.I),
    "lights": re.compile(r"\blight(s)?\b", re.I),
    "climate": re.compile(r"\bclimate|ac|a/c|heater|heating|cooling\b", re.I),
    "trunk": re.compile(r"\btrunk|boot\b", re.I),
    "windows": re.compile(r"\bwindow(s)?\b", re.I),
}

def _extract_features(text: str):
    feats = set()
    t = (text or "")
    for name, pat in FEATURE_PATTERNS.items():
        if pat.search(t):
            feats.add(name)
    return feats

def _infer_missing_type(st):
    if not st.get("type"):
        txt = f"{st.get('summary','')} {st.get('description','')}"
        if _UI_HINT.search(txt):
            st["type"] = "Design"
        elif _API_HINT.search(txt):
            st["type"] = "Engineering"

# Priority inference hints
_SECURITY_HINT = re.compile(r"\b(auth|oauth|oidc|token|mfa|2fa|encryption|tls|cert|key|secret|"
                            r"privacy|pii|gdpr|ccpa|hipaa|soc2|audit|security|compliance)\b", re.I)
_BACKEND_CRITICAL_HINT = re.compile(r"\b(api|endpoint|controller|service|sdk|server|schema|db|queue|topic|"
                                    r"cron|job|etl|pipeline|provision|connect(ion)?)\b", re.I)
_UI_POLISH_HINT = re.compile(r"\b(copy|microcopy|typography|spacing|animation|polish|nudge|"
                             r"tooltip|empty\s*state|color|theme)\b", re.I)

def _infer_missing_priority(st):
    text = f"{st.get('summary','')} {st.get('description','')} {' '.join(st.get('labels',[]))}"
    if _SECURITY_HINT.search(text) or _BACKEND_CRITICAL_HINT.search(text):
        return "P0"
    if _UI_POLISH_HINT.search(text):
        return "P2"
    return "P1"

# Robust normalization for priority & type
def _normalize_priority(value: str) -> str:
    if not value:
        return ""
    v = str(value).strip().lower()
    v = v.replace(" ", "").replace("-", "")
    mapping = {
        "p0": "P0", "0": "P0", "blocker": "P0", "critical": "P0", "highest": "P0", "urgent": "P0",
        "p1": "P1", "1": "P1", "high": "P1", "major": "P1", "normal": "P1", "default": "P1",
        "p2": "P2", "2": "P2", "medium": "P2", "low": "P2", "minor": "P2", "nicetohave": "P2",
        "nice-to-have": "P2",
    }
    return mapping.get(v, ""

)

def _normalize_type(value: str) -> str:
    if not value:
        return ""
    v = str(value).strip().lower()
    if any(k in v for k in ["design", "ux", "ui", "visual", "prototype", "wireframe", "figma"]):
        return "Design"
    if any(k in v for k in ["engineering", "eng", "backend", "frontend", "api", "mobile", "ios", "android", "server", "data", "etl", "infra", "platform"]):
        return "Engineering"
    return ""

def _auto_infer_dependencies(epics):
    by_id = {}
    auth_ids, connect_ids = set(), set()
    feature_to_eng = defaultdict(list)
    feature_to_design = defaultdict(list)

    for ep in epics:
        for st in ep["stories"]:
            by_id[st["storyId"]] = st
            text = f"{st.get('summary','')} {st.get('description','')} {' '.join(st.get('labels',[]))}"
            feats = _extract_features(text)

            if _AUTH_HINT.search(text):
                auth_ids.add(st["storyId"])
            if _CONNECT_HINT.search(text):
                connect_ids.add(st["storyId"])

            st_type = (st.get("type") or "").lower()
            if not st_type:
                _infer_missing_type(st)
                st_type = (st.get("type") or "").lower()

            if feats:
                if st_type == "engineering" or _API_HINT.search(text):
                    for f in feats:
                        feature_to_eng[f].append(st["storyId"])
                if st_type == "design" or _UI_HINT.search(text):
                    for f in feats:
                        feature_to_design[f].append(st["storyId"])

    auth_any = next(iter(auth_ids), None)
    connect_any = next(iter(connect_ids), None)

    for ep in epics:
        for st in ep["stories"]:
            deps = set(ensure_list(st.get("dependencies")))
            text = f"{st.get('summary','')} {st.get('description','')} {' '.join(st.get('labels',[]))}"
            feats = _extract_features(text)
            st_type = (st.get("type") or "").lower()

            if feats and st_type == "design":
                for f in feats:
                    for eng_id in feature_to_eng.get(f, []):
                        if eng_id != st["storyId"]:
                            deps.add(eng_id)

            # Auth/connection pre-reqs when applicable
            if feats:
                if auth_any and auth_any != st["storyId"]:
                    deps.add(auth_any)
                if connect_any and connect_any != st["storyId"]:
                    deps.add(connect_any)

            st["dependencies"] = list(deps)

def _sanitize_story(story_obj):
    """Final guardrail: ensure strict values for 'type' and 'priority' and ownerRole default."""
    story_obj["type"] = _normalize_type(story_obj.get("type", "")) or story_obj.get("type", "")
    if story_obj["type"] not in ALLOWED_TYPES:
        _infer_missing_type(story_obj)
        if story_obj.get("type") not in ALLOWED_TYPES:
            txt = f'{story_obj.get("summary","")} {story_obj.get("description","")}'
            story_obj["type"] = "Engineering" if _API_HINT.search(txt) else "Design"

    story_obj["priority"] = _normalize_priority(story_obj.get("priority", "")) or story_obj.get("priority", "")
    if story_obj["priority"] not in ALLOWED_PRIORITIES:
        story_obj["priority"] = _infer_missing_priority(story_obj)

    if not story_obj.get("ownerRole"):
        story_obj["ownerRole"] = "UX" if story_obj["type"] == "Design" else "Backend"

# --- Augmenter: ensure UI Engineering implementation stories for user-facing flows ---
def _ensure_ui_impl_stories(epics):
    """
    For each epic:
      - For each Design story (user-facing), ensure there is a matching Engineering UI implementation story.
      - New story depends on: (a) Design story (handoff), (b) related API/auth/connect stories if present.
      - Respect cap of <= 6 stories/epic.
    """
    for ep in epics:
        stories = ep.get("stories", [])
        if len(stories) >= 6:
            continue

        # Index helpers
        eng_ui_candidates = set()
        api_story_ids = []
        for st in stories:
            summary_lower = (st.get("summary","") + " " + st.get("description","")).lower()
            if st.get("type") == "Engineering" and ("ui" in summary_lower or "frontend" in summary_lower or "mobile" in summary_lower or "implement ui" in summary_lower):
                eng_ui_candidates.add(st["storyId"])
            if st.get("type") == "Engineering" and (_API_HINT.search(summary_lower) or "api" in summary_lower):
                api_story_ids.append(st["storyId"])

        # For each design story, consider adding a UI impl story if none exists
        for st in list(stories):
            if len(stories) >= 6:
                break
            if st.get("type") != "Design":
                continue

            base = st.get("summary") or "ui-implementation"
            ui_story_id = f"str-ui-impl-{slugify(base)}"
            # Skip if an Engineering UI story that references this already exists
            exists = any(
                s for s in stories
                if s.get("type") == "Engineering" and (
                    ui_story_id == s.get("storyId") or
                    slugify(base) in slugify(s.get("summary",""))
                )
            )
            if exists:
                continue

            # Construct UI impl story
            labels = list(st.get("labels", []))
            platform_hint = "web"
            txt = (" ".join(labels) + " " + ep.get("description","") + " " + base).lower()
            if any(k in txt for k in ["ios","android","mobile"]):
                platform_hint = "mobile"
            elif "web" in txt or "frontend" in txt or "react" in txt:
                platform_hint = "web"

            deps = set(st.get("dependencies", []))
            # depend on related API stories and the design story itself
            deps.update(api_story_ids)
            deps.add(st.get("storyId"))

            ui_story = {
                "storyId": ui_story_id,
                "type": "Engineering",
                "summary": f"Implement {platform_hint.upper()} UI: {st.get('summary','')}",
                "description": (
                    f"Build the {platform_hint.upper()} UI for '{st.get('summary','')}', "
                    f"wire to back-end APIs, and implement state, validation, and error handling. "
                    f"Includes unit/UI tests and accessibility (WCAG AA)."
                ),
                "acceptanceCriteria": [
                    "UI matches approved design spec and redlines.",
                    "All fields validated; errors surfaced inline and are accessible.",
                    "Integrated with required APIs; success and failure states handled.",
                    "Meets a11y (WCAG AA) for focus order, labels, color contrast.",
                    "Unit/UI tests passing; telemetry events emitted per analytics spec."
                ],
                "estimate": "2-3d",
                "priority": st.get("priority") or "P1",
                "labels": list(set(labels + [platform_hint, "ui", "implementation"])),
                "ownerRole": "Frontend" if platform_hint == "web" else "Mobile",
                "dependencies": list(deps),
                "testCases": [
                    "Given valid inputs, When submitted, Then API call succeeds and success UI shown.",
                    "Given API error, When submitted, Then error toast and recovery UI displayed."
                ]
            }
            _sanitize_story(ui_story)
            _clamp_lengths(ui_story)
            stories.append(ui_story)

        ep["stories"] = stories[:6]  # enforce cap


# ---------------------------
# Validation & normalization
# ---------------------------
def validate_and_normalize(epics_raw):
    warnings = []
    if not isinstance(epics_raw, list):
        raise ValueError("Top-level JSON must be a list of epics.")

    epics = []
    for ei, epic in enumerate(epics_raw[:5]):
        if not isinstance(epic, dict):
            warnings.append(f"Epic {ei} is not an object; skipped.")
            continue

        epic_title = epic.get("epic") or f"Epic {ei+1}"
        epic_id = epic.get("epicId") or f"epc-{slugify(epic_title)}-{ei+1}"
        epic_desc = epic.get("description") or ""
        area = epic.get("area") or "Product"
        if area not in ALLOWED_AREAS:
            warnings.append(f"Epic {epic_title}: invalid area '{area}', defaulting to 'Product'.")
            area = "Product"

        stories_in = epic.get("stories") or []
        if not isinstance(stories_in, list):
            stories_in = []
            warnings.append(f"Epic {epic_title}: 'stories' is not a list; defaulting to empty list.")

        stories = []
        for si, st in enumerate(stories_in[:6]):
            if not isinstance(st, dict):
                warnings.append(f"Story {si+1} in Epic '{epic_title}' not an object; skipped.")
                continue

            summary = st.get("summary") or f"Story {si+1}"
            description = st.get("description") or ""
            story_id = st.get("storyId") or f"str-{slugify(summary)}-{si+1}"

            st_type = _normalize_type(st.get("type") or "")
            priority = _normalize_priority(st.get("priority") or "")

            owner_role = st.get("ownerRole") or ""
            labels = [slugify(x) for x in ensure_list(st.get("labels")) if x]

            estimate = normalize_estimate(st.get("estimate") or "")
            acceptance = [str(x) for x in ensure_list(st.get("acceptanceCriteria")) if str(x).strip()]
            tests = [str(x) for x in ensure_list(st.get("testCases")) if str(x).strip()]
            deps = ensure_list(st.get("dependencies"))

            story_obj = {
                "storyId": story_id,
                "type": st_type,
                "summary": summary,
                "description": description,
                "acceptanceCriteria": acceptance,
                "estimate": estimate,
                "priority": priority,
                "labels": labels,
                "ownerRole": owner_role,
                "dependencies": deps,
                "testCases": tests
            }

            _sanitize_story(story_obj)
            _clamp_lengths(story_obj)
            stories.append(story_obj)

        epics.append({
            "epicId": epic_id,
            "epic": epic_title,
            "description": epic_desc,
            "area": area,
            "stories": stories
        })

    # Remove unknown deps first
    story_ids = {st["storyId"] for ep in epics for st in ep["stories"]}
    for ep in epics:
        for st in ep["stories"]:
            st["dependencies"] = [d for d in st["dependencies"] if d in story_ids]

    _dedupe_story_ids(epics)
    _auto_infer_dependencies(epics)
    _ensure_ui_impl_stories(epics)
    _prune_dependency_cycles(epics)

    return epics, warnings

# ---------------------------
# PRD schema helpers (kept in case you want JSON again later)
# ---------------------------
def validate_prd(prd_raw):
    required_keys = [
        "title", "context", "objectives", "non_goals",
        "personas", "user_journeys",
        "functional_requirements", "non_functional_requirements",
        "ux_deliverables", "analytics", "success_metrics",
        "release_plan", "risks", "open_questions",
        "daci", "raci_view", "dependencies"
    ]
    if not isinstance(prd_raw, dict):
        raise ValueError("PRD must be a JSON object.")
    for k in required_keys:
        prd_raw.setdefault(k, [] if k.endswith("s") and k not in ["context","title"] else "")
    for list_key in [
        "objectives", "non_goals", "personas", "user_journeys",
        "functional_requirements", "non_functional_requirements",
        "ux_deliverables", "analytics", "success_metrics",
        "release_plan", "risks", "open_questions", "dependencies"
    ]:
        prd_raw[list_key] = ensure_list(prd_raw.get(list_key))
    daci = prd_raw.get("daci") or {}
    if not isinstance(daci, dict):
        daci = {}
    for role in ["Driver", "Approver", "Contributors", "Informed"]:
        daci.setdefault(role, [] if role in ["Contributors", "Informed"] else "")
    raci = prd_raw.get("raci_view") or []
    if not isinstance(raci, list):
        raci = []
    prd_raw["raci_view"] = raci[:25]
    prd_raw["daci"] = daci
    return prd_raw

def _md_heading(text, level=2):
    text = str(text or "").strip()
    hashes = "#" * max(1, min(6, level))
    return f"{hashes} {text}\n\n" if text else ""

def _md_list(items):
    out = ""
    for it in ensure_list(items):
        if isinstance(it, dict):
            kv = "; ".join([f"**{k}**: {v}" for k, v in it.items()])
            out += f"- {kv}\n"
        else:
            out += f"- {it}\n"
    return out + ("\n" if out else "")

def prd_to_markdown(prd):
    md = f"# {prd.get('title','PRD')}\n\n"
    if prd.get("context"):
        md += f"{prd['context']}\n\n"
    md += _md_heading("Objectives", 2) + _md_list(prd.get("objectives"))
    md += _md_heading("Non-Goals", 2) + _md_list(prd.get("non_goals"))
    md += _md_heading("Personas", 2)
    for p in ensure_list(prd.get("personas")):
        name = p.get("name","Persona")
        md += _md_heading(name, 3)
        if p.get("summary"): md += f"{p['summary']}\n\n"
        if p.get("primary_jobs_to_be_done"):
            md += _md_heading("Primary JTBD", 4) + _md_list(p["primary_jobs_to_be_done"])
    md += _md_heading("User Journeys", 2)
    for j in ensure_list(prd.get("user_journeys")):
        nm = j.get("name","Journey")
        md += _md_heading(nm, 3)
        if j.get("happy_path_steps"):
            md += _md_heading("Happy Path", 4) + _md_list(j["happy_path_steps"])
        if j.get("edge_cases"):
            md += _md_heading("Edge Cases", 4) + _md_list(j["edge_cases"])
    md += _md_heading("Functional Requirements", 2)
    for fr in ensure_list(prd.get("functional_requirements")):
        title = f"{fr.get('id','FR')}: {fr.get('name','')}".strip()
        md += _md_heading(title, 3)
        md += _md_list(fr.get("details"))
    nfr = prd.get("non_functional_requirements") or {}
    md += _md_heading("Non-Functional Requirements", 2)
    for section, items in nfr.items():
        md += _md_heading(section.replace("_"," ").title(), 3)
        md += _md_list(items)
    if prd.get("ux_deliverables"):
        md += _md_heading("UX Deliverables", 2)
        for d in ensure_list(prd.get("ux_deliverables")):
            flow = d.get("flow","Flow")
            md += _md_heading(flow, 3)
            md += _md_list(d.get("artifacts"))
    if prd.get("analytics"):
        md += _md_heading("Analytics", 2)
        for ev in ensure_list(prd["analytics"]):
            line = f"- **Event**: {ev.get('event','')} — **Props**: {', '.join(ensure_list(ev.get('properties')))} — **Signal**: {ev.get('success_signal','')}\n"
            md += line
        md += "\n"
    if prd.get("success_metrics"):
        md += _md_heading("Success Metrics", 2)
        for m in ensure_list(prd["success_metrics"]):
            line = f"- **{m.get('metric','Metric')}**: {m.get('target','')}\n"
            md += line
        md += "\n"
    if prd.get("release_plan"):
        md += _md_heading("Release Plan", 2)
        for phase in ensure_list(prd["release_plan"]):
            md += _md_heading(phase.get("phase","Phase"), 3)
            if phase.get("scope"): md += _md_heading("Scope", 4) + _md_list(phase["scope"])
            if phase.get("risks"): md += _md_heading("Risks", 4) + _md_list(phase["risks"])
    md += _md_heading("Risks", 2) + _md_list(prd.get("risks"))
    md += _md_heading("Open Questions", 2) + _md_list(prd.get("open_questions"))
    md += _md_heading("Dependencies", 2) + _md_list(prd.get("dependencies"))
    daci = prd.get("daci") or {}
    md += _md_heading("Decision Framework", 2)
    if daci:
        md += _md_heading("DACI", 3)
        md += f"- **Driver**: {daci.get('Driver','')}\n"
        md += f"- **Approver**: {daci.get('Approver','')}\n"
        md += f"- **Contributors**: {', '.join(ensure_list(daci.get('Contributors')))}\n"
        md += f"- **Informed**: {', '.join(ensure_list(daci.get('Informed')))}\n\n"
    raci = ensure_list(prd.get("raci_view"))
    if raci:
        md += _md_heading("RACI", 3)
        for row in raci:
            md += f"- **{row.get('workstream','')}** — R: {row.get('R','')}; A: {row.get('A','')}; C: {', '.join(ensure_list(row.get('C')))}; I: {', '.join(ensure_list(row.get('I')))}\n"
        md += "\n"
    return md.strip() + "\n"

# ---------------------------
# OpenAI helper with retries
# ---------------------------
def openai_chat_with_retries(messages, max_tokens):
    attempts = int(os.getenv("OPENAI_RETRIES", "2"))
    base_delay = float(os.getenv("OPENAI_RETRY_BASE_DELAY", "0.6"))
    last_exc = None
    for i in range(attempts + 1):
        try:
            return client.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=MODEL_TEMPERATURE,
                max_tokens=max_tokens,
                messages=messages,
            )
        except (APITimeoutError, RateLimitError, APIError) as e:
            last_exc = e
            if i < attempts:
                time.sleep(base_delay * (2 ** i))
            else:
                raise
    if last_exc:
        raise last_exc

# ---------------------------
# Routes
# ---------------------------
@app.route('/generate', methods=['POST', 'OPTIONS'])
def generate():
    if request.method == 'OPTIONS':
        return ("", 204)

    try:
        data = request.get_json(silent=True) or {}
        goal = (data.get("goal") or "").strip()[:1200]

        if not goal:
            return jsonify({"epics": [], "status": "error", "error": "No goal provided."}), 400
        if not is_meaningful_goal(goal):
            return jsonify({"epics": [], "status": "error", "error": "Please provide a more descriptive, product-related goal."}), 400
        if not moderate_text(goal):
            return jsonify({"epics": [], "status": "error", "error": "The provided text violates our usage policy. Please rephrase your request."}), 400

        prompt = f"Generate product epics and stories for the following product goal:\n{goal}"
        system_inst = """
You are a Principal PM + Tech Lead + UX Lead generating **Jira-ready planning** for ANY product surface: web apps, mobile apps, back-end services/APIs, data pipelines/ML, integrations, or platform/infra. Your output must be realistic, implementation-aware, and testable.

Return ONLY valid JSON (no markdown, no comments). Output MUST be a JSON array of epics following this schema (fields shown are required). The example values are illustrative only—choose values appropriate to the goal.

[
  {
    "epicId": "epc-example-001",
    "epic": "Epic Title",
    "description": "What this epic covers",
    "area": "Product",
    "stories": [
      {
        "storyId": "str-example-ui-001",
        "type": "Design",
        "summary": "Short summary",
        "description": "As a [user/system], I want [feature], so that [value]",
        "acceptanceCriteria": ["..."],
        "estimate": "2d",
        "priority": "P1",
        "labels": ["web","api"],
        "ownerRole": "UX",
        "dependencies": ["str-example-api-001"],
        "testCases": ["..."]
      }
    ]
  }
]

QUALITY RULES (STRICT):
- Limit to 3–5 epics; each epic 3–6 stories.
- Every story MUST include non-empty "type" (Design|Engineering) AND "priority" (P0|P1|P2).
- For **user-facing goals**, create BOTH:
  (a) Design stories with deliverables (IA/wireframes/visual spec/Figma components/redlines) and a "Figma handoff to engineering" acceptance criterion.
  (b) Engineering stories for **UI implementation** (per primary screen/flow) that wire the UI to back-end APIs, plus Engineering stories for granular APIs.
- Also include Engineering stories for security/compliance/standards (encryption at rest/in transit, OAuth2/OIDC, audit logging, PII handling, SOC2/GDPR/HIPAA where relevant), and ops concerns (observability, rate limiting, retries).
- Dependencies reflect realistic build order and avoid cycles:
  Design → UI Engineering → API Engineering where applicable (and API → auth/infra prerequisites).
- Acceptance criteria are actionable and testable.
- Estimates realistic and normalized (4h|1d|2d|5d or ranges like 1-2d).
- Professional tone. Use kebab-case stable IDs, unique within the response.
"""
        try:
            response = openai_chat_with_retries(
                messages=[
                    {"role": "system", "content": system_inst},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=MAX_TOKENS_GENERATE,
            )
        except APITimeoutError:
            return jsonify({"epics": [], "status": "error", "error": "Upstream model timeout."}), 504
        except RateLimitError:
            return jsonify({"epics": [], "status": "error", "error": "Rate limited by model."}), 429
        except APIError as e:
            return jsonify({"epics": [], "status": "error", "error": f"Upstream API error: {str(e)}"}), 502

        raw_output = strip_code_fences(response.choices[0].message.content or "")

        try:
            epics_raw = json.loads(raw_output)
        except json.JSONDecodeError:
            if ENABLE_REPAIR_PASS:
                repair_prompt = f"""Convert the following content into VALID JSON matching the schema exactly (array of epics with the fields shown). Return ONLY the JSON.

SCHEMA:
[{{"epicId":"...","epic":"...","description":"...","area":"Product|Design|Engineering","stories":[{{"storyId":"...","type":"Design|Engineering","summary":"...","description":"...","acceptanceCriteria":["..."],"estimate":"4h|1d|2d|5d|1-2d","priority":"P0|P1|P2","labels":["..."],"ownerRole":"...","dependencies":["storyId","..."],"testCases":["..."]}}]}}]

CONTENT:
{raw_output}
"""
                try:
                    fix = openai_chat_with_retries(
                        messages=[
                            {"role": "system", "content": "You fix invalid JSON to match the given schema. Output ONLY valid JSON."},
                            {"role": "user", "content": repair_prompt},
                        ],
                        max_tokens=min(1200, MAX_TOKENS_GENERATE),
                    )
                    raw_output = strip_code_fences(fix.choices[0].message.content or "")
                    epics_raw = json.loads(raw_output)
                except Exception:
                    parsed = legacy_parse_response(raw_output)
                    normalized, warnings = validate_and_normalize(parsed)
                    return jsonify({"epics": normalized, "status": "partial", "note": "Fallback parser used; model returned invalid JSON.", "warnings": warnings}), 200
            else:
                parsed = legacy_parse_response(raw_output)
                normalized, warnings = validate_and_normalize(parsed)
                return jsonify({"epics": normalized, "status": "partial", "note": "Fallback parser used; model returned invalid JSON.", "warnings": warnings}), 200

        try:
            normalized, warnings = validate_and_normalize(epics_raw)
        except ValueError as ve:
            parsed = legacy_parse_response(raw_output)
            normalized, warnings = validate_and_normalize(parsed)
            return jsonify({"epics": normalized, "status": "partial", "note": str(ve), "warnings": warnings}), 200

        return jsonify({"epics": normalized, "status": "success", "warnings": warnings}), 200

    except Exception as e:
        return jsonify({
            "epics": [],
            "status": "error",
            "error": "Unexpected server error.",
            "details": str(e)
        }), 500

# ---------------------------
# PRD Generation Route (Markdown-only, zero JSON parsing)
# ---------------------------
@app.route('/generate_prd', methods=['POST', 'OPTIONS'])
def generate_prd():
    if request.method == 'OPTIONS':
        return ("", 204)
    try:
        data = request.get_json(silent=True) or {}
        goal = (data.get("goal") or "").strip()[:1200]

        if not goal:
            return jsonify({"prd_markdown": "", "status": "error", "error": "No goal provided."}), 400
        if not is_meaningful_goal(goal):
            return jsonify({"prd_markdown": "", "status": "error", "error": "Please provide a more descriptive, product-related goal."}), 400
        if not moderate_text(goal):
            return jsonify({"prd_markdown": "", "status": "error", "error": "The provided text violates our usage policy. Please rephrase your request."}), 400

        system_inst = """
You are a Principal PM writing a clear, skimmable Product Requirements Document (PRD) for ANY product surface (web, mobile, back-end/API, data/ML, integrations, platform/infra).

Return ONLY GitHub-flavored Markdown (no code fences, no JSON, no YAML). Be concise, professional, and implementation-aware. Use headings and lists so it reads like a wiki page. Include only sections that make sense for the goal.

REQUIRED SECTIONS (use these exact headings):
# <Title>
A 2–3 sentence overview of scope and value.

## Objectives
- Measurable objectives…

## Non-Goals
- Explicitly out-of-scope items…

## Personas
- **Name** — short summary
  - Primary JTBD: …

## User Journeys
- **Journey name**
  - Happy path: step 1, step 2, …
  - Edge cases: …

## Functional Requirements
- **FR-001 – Name**
  - Details: bullet points of what must be delivered

## Non-Functional Requirements
- **Security & Privacy**: encryption in transit/at rest, OAuth2/OIDC (if applicable), audit logging, PII handling, GDPR/CCPA/HIPAA (as relevant)
- **Reliability & Ops**: SLO/SLA notes, rate limiting, retries/backoff, circuit breaking
- **Performance**: concrete p95/p99 targets if applicable
- **Observability**: logs, metrics, tracing, alerts

## UX Deliverables (if user-facing)
- Information architecture, wireframes, visual spec, Figma components, redlines, handoff checklist

## Analytics
- **Event** — key properties — success signal

## Success Metrics
- **Metric**: target

## Release Plan
- **MVP** — scope bullets; key risks
- **V1** — scope bullets

## Risks
- Bullet list of top risks

## Open Questions
- Bullet list of things to resolve

## Dependencies
- Internal/external dependencies

## Governance
- **DACI** — Driver, Approver, Contributors, Informed
- **RACI** — rows of Workstream — R/A/C/I
"""
        user_msg = f"Write the PRD in Markdown for this product goal:\n\n{goal}"

        try:
            response = openai_chat_with_retries(
                messages=[
                    {"role": "system", "content": system_inst},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=MAX_TOKENS_PRD,
            )
        except APITimeoutError:
            return jsonify({"prd_markdown": "", "status": "error", "error": "Upstream model timeout."}), 504
        except RateLimitError:
            return jsonify({"prd_markdown": "", "status": "error", "error": "Rate limited by model."}), 429
        except APIError as e:
            return jsonify({"prd_markdown": "", "status": "error", "error": f"Upstream API error: {str(e)}"}), 502

        md = strip_code_fences(response.choices[0].message.content or "")
        if not md:
            return jsonify({"prd_markdown": "", "status": "error", "error": "Empty response from model."}), 502

        return jsonify({"prd_markdown": md, "status": "success"}), 200

    except Exception as e:
        return jsonify({
            "prd_markdown": "",
            "status": "error",
            "error": "Unexpected server error.",
            "details": str(e)
        }), 500

# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)

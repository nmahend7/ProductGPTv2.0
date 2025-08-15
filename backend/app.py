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
    if _allowed_origins == "*" or (origin and origin in _allowed_origins):
        resp.headers.setdefault("Access-Control-Allow-Origin", origin if origin else "*")
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

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=OPENAI_TIMEOUT)

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
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:].strip()
    return t

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

# Feature tokens (kept from your earlier logic; harmless for non-vehicle contexts)
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

# Robust normalization for priority & type (handles synonyms/casing & junk like "P?")
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
    return mapping.get(v, "")

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
    """
    Final guardrail: ensure strict values for 'type' and 'priority'.
    Replace unknowns with inferred defaults so FE never sees "P?" or "N/A" unless explicitly empty.
    """
    story_obj["type"] = _normalize_type(story_obj.get("type", "")) or story_obj.get("type", "")
    if story_obj["type"] not in ALLOWED_TYPES:
        _infer_missing_type(story_obj)
        if story_obj.get("type") not in ALLOWED_TYPES:
            # fall back
            txt = f'{story_obj.get("summary","")} {story_obj.get("description","")}'
            story_obj["type"] = "Engineering" if _API_HINT.search(txt) else "Design"

    story_obj["priority"] = _normalize_priority(story_obj.get("priority", "")) or story_obj.get("priority", "")
    if story_obj["priority"] not in ALLOWED_PRIORITIES:
        story_obj["priority"] = _infer_missing_priority(story_obj)

    if not story_obj.get("ownerRole"):
        story_obj["ownerRole"] = "UX" if story_obj["type"] == "Design" else "Backend"

# ---------------------------
# Validation & normalization
# ---------------------------
def validate_and_normalize(epics_raw):
    """
    - Caps epics (<=5) and stories per epic (<=6)
    - Ensures IDs, types, arrays, dependencies, estimate format
    - Ensures unique storyIds
    - Normalizes & guarantees 'type' (Design|Engineering) and 'priority' (P0|P1|P2)
    - Sensible ownerRole defaults
    - Infers obvious dependencies
    - Prunes unknown deps and dependency cycles
    - Clamps long AC/test lists
    Returns (epics, warnings)
    """
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

            # Normalize directly from model output
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

            # Ensure valid & present type/priority and ownerRole defaults
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

    # Ensure unique IDs, then infer deps, then prune cycles
    _dedupe_story_ids(epics)
    _auto_infer_dependencies(epics)
    _prune_dependency_cycles(epics)

    return epics, warnings

# ---------------------------
# PRD schema validation (lightweight)
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
You are a Principal PM + Tech Lead + UX Lead generating **Jira-ready planning** for ANY product surface: web apps, mobile apps, back-end services, APIs, data pipelines/ML, integrations, or platform/infra. Your output must be realistic, implementation-aware, and testable.

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
- For **user-facing goals**, create BOTH Design and Engineering stories for the main flows.
  - Design stories MUST include deliverables (IA/wireframes/visual spec/Figma components/redlines) and a "Figma handoff to engineering" acceptance criterion.
  - Engineering stories MUST include granular back-end/API and UI integration where relevant.
- Also include Engineering stories for security/compliance/standards (encryption in transit/at rest, OAuth2/OIDC, audit logging, PII handling, SOC2/GDPR/HIPAA where relevant), and ops concerns (observability, rate limiting, retries).
- Dependencies must reflect realistic build order; avoid cycles (UI → API; API → auth/infra when applicable).
- Acceptance criteria must be actionable and testable.
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
# PRD Generation Route
# ---------------------------
@app.route('/generate_prd', methods=['POST', 'OPTIONS'])
def generate_prd():
    if request.method == 'OPTIONS':
        return ("", 204)
    try:
        data = request.get_json(silent=True) or {}
        goal = (data.get("goal") or "").strip()[:1200]

        if not goal:
            return jsonify({"prd": {}, "status": "error", "error": "No goal provided."}), 400
        if not is_meaningful_goal(goal):
            return jsonify({"prd": {}, "status": "error", "error": "Please provide a more descriptive, product-related goal."}), 400
        if not moderate_text(goal):
            return jsonify({"prd": {}, "status": "error", "error": "The provided text violates our usage policy. Please rephrase your request."}), 400

        system_inst = """
You are a Principal PM creating a rigorous Product Requirements Document (PRD) for ANY product surface (web, mobile, back-end/API, data/ML, integrations, platform/infra). Use a DACI model for decision-making and also include a simple RACI view per workstream. Be concise, professional, and implementation-aware.

Return ONLY valid JSON (no markdown). Produce ONE JSON object with the EXACT keys below (example values are illustrative):

{
  "title": "string",
  "context": "string",
  "objectives": ["..."],
  "non_goals": ["..."],
  "personas": [
    {"name":"...", "summary":"...", "primary_jobs_to_be_done":["..."]}
  ],
  "user_journeys": [
    {"name":"...", "happy_path_steps":["..."], "edge_cases":["..."]}
  ],
  "functional_requirements": [
    {"id":"FR-001","name":"...","details":["..."]}
  ],
  "non_functional_requirements": {
    "security":["encryption in transit (TLS 1.2+)","encryption at rest","credential storage (OS keystore)","secure key exchange","JWT/OIDC","audit logging"],
    "privacy_compliance":["data minimization","retention policy","DSAR workflow","GDPR/CCPA/HIPAA (if applicable)"],
    "reliability":["SLOs/SLAs","rate limiting","retries/backoff","circuit breaking"],
    "performance":["p95 startup < Xs","p95 API latency < Ys"],
    "observability":["distributed tracing","structured logs","metrics + alerts"]
  },
  "ux_deliverables": [
    {"flow":"User-facing flow name","artifacts":["IA","wireframes","visual spec","Figma components","redlines","handoff checklist"]}
  ],
  "analytics": [
    {"event":"...","properties":["..."],"success_signal":"..."}
  ],
  "success_metrics": [
    {"metric":"...", "target":"..."}
  ],
  "release_plan": [
    {"phase":"MVP","scope":["..."],"risks":["..."]},
    {"phase":"V1","scope":["..."]}
  ],
  "risks": ["..."],
  "open_questions": ["..."],
  "dependencies": ["..."],
  "daci": {
    "Driver":"Product Manager",
    "Approver":"Head of Product",
    "Contributors":["Tech Lead","UX Lead","Security Lead"],
    "Informed":["Support","Sales"]
  },
  "raci_view": [
    {"workstream":"Auth","R":"Tech Lead","A":"Head of Product","C":["Security Lead"],"I":["Support"]},
    {"workstream":"Platform","R":"Platform Lead","A":"Head of Eng","C":["SRE Lead"],"I":["CS"]}
  ]
}

QUALITY RULES:
- Objectives are measurable; non-goals explicitly call out what's excluded.
- Functional requirements cover main features; be specific.
- UX deliverables **only when there is a user-facing surface**, and MUST include a Figma handoff (components & redlines).
- Include security/privacy/compliance and operational concerns under NFRs.
- Release plan has clear phase gates; call out risks and dependencies.
"""
        user_msg = f"Create the PRD for the following product goal:\n{goal}"
        try:
            response = openai_chat_with_retries(
                messages=[
                    {"role": "system", "content": system_inst},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=MAX_TOKENS_PRD,
            )
        except APITimeoutError:
            return jsonify({"prd": {}, "status": "error", "error": "Upstream model timeout."}), 504
        except RateLimitError:
            return jsonify({"prd": {}, "status": "error", "error": "Rate limited by model."}), 429
        except APIError as e:
            return jsonify({"prd": {}, "status": "error", "error": f"Upstream API error: {str(e)}"}), 502

        raw = strip_code_fences(response.choices[0].message.content or "")

        try:
            prd_raw = json.loads(raw)
        except json.JSONDecodeError:
            if ENABLE_REPAIR_PASS:
                repair_prompt = f"""Convert the following into VALID JSON that EXACTLY matches the PRD object schema provided earlier. Return ONLY the JSON object.

CONTENT:
{raw}
"""
                try:
                    fix = openai_chat_with_retries(
                        messages=[
                            {"role": "system", "content": "You fix invalid JSON to match the given schema. Output ONLY valid JSON."},
                            {"role": "user", "content": repair_prompt},
                        ],
                        max_tokens=min(1200, MAX_TOKENS_PRD),
                    )
                    raw = strip_code_fences(fix.choices[0].message.content or "")
                    prd_raw = json.loads(raw)
                except Exception:
                    return jsonify({"prd": {}, "status": "error", "error": "Model returned invalid JSON and repair failed."}), 502
            else:
                return jsonify({"prd": {}, "status": "error", "error": "Model returned invalid JSON."}), 502

        try:
            prd = validate_prd(prd_raw)
        except ValueError as ve:
            return jsonify({"prd": {}, "status": "error", "error": str(ve)}), 400

        return jsonify({"prd": prd, "status": "success"}), 200

    except Exception as e:
        return jsonify({
            "prd": {},
            "status": "error",
            "error": "Unexpected server error.",
            "details": str(e)
        }), 500

# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)

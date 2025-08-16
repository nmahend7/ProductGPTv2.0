// MainApp.jsx
import React, { useMemo, useState } from "react";

/* ----- Animations & background (unchanged) ----- */
const pulseKeyframes = `
@keyframes pulse {
  0%, 100% {
    transform: translate(-50%, -50%) scale(1);
    opacity: 0.4;
  }
  50% {
    transform: translate(-50%, -50%) scale(1.15);
    opacity: 0.8;
  }
}

@keyframes glow {
  0% { text-shadow: 0 0 8px #0ff, 0 0 12px #00f, 0 0 20px #0ff; color: #0ff; }
  25% { text-shadow: 0 0 8px #ff00ff, 0 0 14px #f0f, 0 0 22px #f0f; color: #ff00ff; }
  50% { text-shadow: 0 0 8px #0f0, 0 0 14px #0f0, 0 0 22px #0f0; color: #0f0; }
  75% { text-shadow: 0 0 10px #00f, 0 0 14px #00f, 0 0 22px #00f; color: #00f; }
  100% { text-shadow: 0 0 8px #0ff, 0 0 12px #00f, 0 0 20px #0ff; color: #0ff; }
}
`;

const orbStyle = {
  position: "fixed",
  top: "50%",
  left: "50%",
  width: "400px",
  height: "400px",
  background:
    "radial-gradient(circle, rgba(0,255,255,0.4), rgba(0,0,255,0.2), transparent)",
  borderRadius: "50%",
  filter: "blur(80px)",
  transform: "translate(-50%, -50%)",
  zIndex: -1,
  animation: "pulse 8s ease-in-out infinite",
  pointerEvents: "none",
};

// ✅ Use env var if set; otherwise fall back to your live backend
const API_BASE =
  process.env.REACT_APP_API_BASE || "https://silo-backend.onrender.com";

/* ----- Helpers ----- */
function slugify(s) {
  return String(s || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/(^-|-$)/g, "")
    .slice(0, 40);
}

function normalizeEpics(rawEpics) {
  return (rawEpics || []).map((epic, ei) => {
    const epicId =
      epic.epicId || `epc-${slugify(epic.epic || `epic-${ei + 1}`)}-${ei + 1}`;
    const stories = (epic.stories || []).map((st, si) => {
      const base = st.summary || st.description || `story-${si + 1}`;
      const storyId = st.storyId || `str-${slugify(base)}-${si + 1}`;
      return { ...st, storyId };
    });
    return { ...epic, epicId, stories };
  });
}

function downloadJSON(filename, dataObj) {
  const blob = new Blob([JSON.stringify(dataObj, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

// Topologically sort stories inside an epic (deps first)
function topoSortStories(stories) {
  const byId = {};
  const indeg = {};
  const edges = {};

  stories.forEach((s) => {
    byId[s.storyId] = s;
    indeg[s.storyId] = 0;
    edges[s.storyId] = new Set();
  });

  stories.forEach((s) => {
    const deps = Array.isArray(s.dependencies) ? s.dependencies : [];
    deps.forEach((d) => {
      if (byId[d] && d !== s.storyId) {
        if (!edges[d].has(s.storyId)) {
          edges[d].add(s.storyId);
          indeg[s.storyId] += 1;
        }
      }
    });
  });

  const q = [];
  Object.keys(indeg).forEach((id) => {
    if (indeg[id] === 0) q.push(id);
  });

  const order = [];
  while (q.length) {
    const id = q.shift();
    order.push(id);
    edges[id].forEach((v) => {
      indeg[v] -= 1;
      if (indeg[v] === 0) q.push(v);
    });
  }

  const ordered = order.map((id) => byId[id]).filter(Boolean);
  const leftover = stories.filter((s) => !order.includes(s.storyId));
  return [...ordered, ...leftover];
}

/* ----- Tiny Markdown → HTML (headings, bold, lists, paragraphs) ----- */
function escapeHtml(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}
function mdToHtml(md) {
  if (!md) return "";
  const lines = md.split(/\r?\n/);

  const out = [];
  let inList = false;

  const flushP = (buf) => {
    if (!buf.length) return;
    out.push(`<p>${buf.join(" ").trim()}</p>`);
    buf.length = 0;
  };

  let pbuf = [];
  for (let raw of lines) {
    let line = raw;
    // headings
    if (/^#{1,6}\s+/.test(line)) {
      flushP(pbuf);
      const level = (line.match(/^#+/) || ["#"])[0].length;
      const text = line.replace(/^#{1,6}\s+/, "");
      out.push(`<h${level}>${escapeHtml(text)}</h${level}>`);
      continue;
    }

    // list item
    if (/^\s*-\s+/.test(line)) {
      flushP(pbuf);
      if (!inList) {
        inList = true;
        out.push("<ul>");
      }
      const li = line.replace(/^\s*-\s+/, "");
      out.push(`<li>${inlineMd(escapeHtml(li))}</li>`);
      continue;
    } else if (inList && line.trim() === "") {
      out.push("</ul>");
      inList = false;
      continue;
    }

    // blank line
    if (line.trim() === "") {
      flushP(pbuf);
      continue;
    }

    // paragraph buffer
    pbuf.push(inlineMd(escapeHtml(line)));
  }
  if (inList) out.push("</ul>");
  flushP(pbuf);
  return out.join("\n");
}
function inlineMd(s) {
  // bold **text**
  s = s.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  // simple italics _text_
  s = s.replace(/(^|[^\w])_([^_]+)_/g, "$1<em>$2</em>");
  return s;
}

/* ----- Main App ----- */
function MainApp({ user, onLogout }) {
  const [goal, setGoal] = useState("");
  const [epics, setEpics] = useState([]);
  const [prd, setPrd] = useState(null);
  const [prdMarkdown, setPrdMarkdown] = useState("");
  const [activeTab, setActiveTab] = useState("tickets"); // "tickets" | "prd"
  const [loading, setLoading] = useState(false);
  const [prdLoading, setPrdLoading] = useState(false);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState(() => new Set()); // storyIds
  const [showJsonEditor, setShowJsonEditor] = useState(false);

  const logoStyle = {
    fontSize: "3rem",
    fontWeight: 800,
    marginBottom: "10px",
    textAlign: "center",
    animation: "glow 6s infinite ease-in-out",
    letterSpacing: "6px",
    textTransform: "lowercase",
    fontFamily:
      "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Inter, Helvetica, Arial, sans-serif",
  };

  const logoPill = {
    display: "inline-block",
    padding: "4px 14px",
    borderRadius: 16,
    border: "1px solid #1f2a44",
    background:
      "linear-gradient(180deg, rgba(18,20,38,0.9) 0%, rgba(28,31,58,0.9) 100%)",
    boxShadow: "0 4px 24px rgba(0,255,255,0.15), inset 0 0 0 1px rgba(255,255,255,0.04)",
  };

  const logoText = loading ? "loading" : "silo";

  // Build lookups
  const { storyIndex, dependentsIndex } = useMemo(() => {
    const idx = {};
    const depsIdx = {};
    epics?.forEach((ep) => {
      ep.stories?.forEach((st) => {
        if (st.storyId) {
          idx[st.storyId] = {
            summary: st.summary,
            epicId: ep.epicId,
            epic: ep.epic,
          };
        }
      });
    });
    epics?.forEach((ep) => {
      ep.stories?.forEach((st) => {
        const deps = Array.isArray(st.dependencies) ? st.dependencies : [];
        deps.forEach((d) => {
          if (!depsIdx[d]) depsIdx[d] = [];
          depsIdx[d].push(st.storyId);
        });
      });
    });
    return { storyIndex: idx, dependentsIndex: depsIdx };
  }, [epics]);

  async function fetchTickets() {
    setLoading(true);
    setError("");
    setEpics([]);
    setSelected(new Set());

    try {
      const res = await fetch(`${API_BASE}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ goal }),
      });
      const data = await res.json();

      if (res.ok) {
        const normalized = normalizeEpics(data.epics || []);
        const sorted = normalized.map((ep) => ({
          ...ep,
          stories: topoSortStories(ep.stories || []),
        }));
        setEpics(sorted);
      } else {
        setError(data.error || "Unknown error");
      }
    } catch (err) {
      setError("Failed to connect to backend");
    } finally {
      setLoading(false);
    }
  }

  async function fetchPrd() {
    if (!goal.trim()) return;
    setPrdLoading(true);
    setError("");
    try {
      const res = await fetch(`${API_BASE}/generate_prd`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ goal }),
      });
      const data = await res.json();
      if (res.ok) {
        setPrd(data.prd || {});
        setPrdMarkdown(data.prd_markdown || "");
      } else {
        setError(data.error || "Failed to generate PRD");
      }
    } catch (e) {
      setError("Failed to connect to backend for PRD");
    } finally {
      setPrdLoading(false);
    }
  }

  function handleSubmit(e) {
    e.preventDefault();
    if (!goal.trim()) return;
    if (activeTab === "tickets") {
      fetchTickets();
    } else {
      fetchPrd();
    }
  }

  function toggleSelectStory(storyId) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(storyId)) next.delete(storyId);
      else next.add(storyId);
      return next;
    });
  }

  function addDepsToSelection(depsArr) {
    if (!depsArr || depsArr.length === 0) return;
    setSelected((prev) => {
      const next = new Set(prev);
      depsArr.forEach((d) => next.add(d));
      return next;
    });
  }

  function setEpicSelection(epic, checked) {
    const ids = epic.stories?.map((s) => s.storyId).filter(Boolean) || [];
    setSelected((prev) => {
      const next = new Set(prev);
      ids.forEach((id) => (checked ? next.add(id) : next.delete(id)));
      return next;
    });
  }

  function onUpdateStory(epicId, storyId, updated) {
    setEpics((prev) =>
      prev.map((ep) => {
        if (ep.epicId !== epicId) return ep;
        const stories = ep.stories.map((st) =>
          st.storyId === storyId ? { ...st, ...updated } : st
        );
        return { ...ep, stories };
      })
    );
  }

  function onUpdateEpic(epicId, updatedFields) {
    setEpics((prev) =>
      prev.map((ep) => (ep.epicId === epicId ? { ...ep, ...updatedFields } : ep))
    );
  }

  function exportSelected() {
    const selectedStories = [];
    const includedStoryIds = new Set(selected);

    epics.forEach((ep) => {
      ep.stories.forEach((st) => {
        if (includedStoryIds.has(st.storyId)) {
          const deps = Array.isArray(st.dependencies) ? st.dependencies : [];
          selectedStories.push({
            epicId: ep.epicId,
            epic: ep.epic,
            area: ep.area,
            storyId: st.storyId,
            type: st.type,
            summary: st.summary,
            description: st.description,
            acceptanceCriteria: st.acceptanceCriteria,
            estimate: st.estimate,
            priority: st.priority,
            labels: st.labels,
            ownerRole: st.ownerRole,
            dependencies: deps,
            testCases: st.testCases,
          });
        }
      });
    });

    const payload = {
      goal,
      selectedCount: selectedStories.length,
      stories: selectedStories,
    };

    downloadJSON("silo-export.json", payload);
  }

  const totalSelected = selected.size;

  return (
    <>
      <style>{pulseKeyframes}</style>
      <div style={orbStyle}></div>

      <div
        style={{
          minHeight: "100vh",
          display: "flex",
          flexDirection: "column",
          justifyContent: "flex-start",
          alignItems: "center",
          padding: "20px",
          fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
          backgroundColor: "#0c0c1b",
          color: "#e0e0ff",
          position: "relative",
          zIndex: 1,
        }}
      >
        {/* Top-right user info + logout */}
        <div
          style={{
            position: "absolute",
            top: 16,
            right: 16,
            display: "flex",
            alignItems: "center",
            gap: 10,
          }}
        >
          {user?.photoURL && (
            <img
              src={user.photoURL}
              alt="avatar"
              style={{ width: 32, height: 32, borderRadius: "50%" }}
            />
          )}
          {user?.displayName && (
            <span style={{ fontSize: 14, opacity: 0.85 }}>
              {user.displayName}
            </span>
          )}
          {typeof onLogout === "function" && (
            <button
              onClick={onLogout}
              style={{
                background: "#1e1e2f",
                color: "#fff",
                border: "1px solid #333",
                borderRadius: 8,
                padding: "6px 10px",
                cursor: "pointer",
              }}
            >
              Logout
            </button>
          )}
        </div>

        {/* Header / Logo */}
        <h1 style={logoStyle}>
          <span style={logoPill}>{logoText}</span>
        </h1>

        {/* Tab Bar */}
        <div
          style={{
            display: "flex",
            gap: 8,
            background: "#121428",
            border: "1px solid #1f2540",
            borderRadius: 12,
            padding: 6,
            marginBottom: 14,
          }}
        >
          {["tickets", "prd"].map((t) => {
            const active = activeTab === t;
            return (
              <button
                key={t}
                onClick={() => setActiveTab(t)}
                style={{
                  padding: "8px 14px",
                  borderRadius: 8,
                  border: "1px solid " + (active ? "#00bcd4" : "transparent"),
                  background: active ? "#0e2a33" : "transparent",
                  color: active ? "#bdf2ff" : "#c9d5ff",
                  cursor: "pointer",
                }}
              >
                {t === "tickets" ? "Tickets" : "PRD"}
              </button>
            );
          })}
        </div>

        {/* Query form (buttons are tab-specific) */}
        <form onSubmit={handleSubmit} style={{ width: "100%", maxWidth: 800 }}>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            <input
              type="text"
              placeholder="Enter your product goal..."
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              style={{
                flex: 1,
                minWidth: 260,
                padding: "12px 16px",
                fontSize: "1.1rem",
                borderRadius: "8px",
                border: "1px solid #333",
                backgroundColor: "#1e1e2f",
                color: "#fff",
                boxShadow: "0 2px 5px rgb(0 0 0 / 0.3)",
              }}
            />

            {activeTab === "tickets" ? (
              <button
                type="submit"
                disabled={loading || !goal.trim()}
                style={{
                  padding: "12px 16px",
                  fontSize: "1.05rem",
                  borderRadius: "8px",
                  border: "1px solid " + (loading ? "#2a2f45" : "transparent"),
                  backgroundColor: loading ? "#2a2a42" : "#00bcd4",
                  color: loading ? "#8aa0ac" : "white",
                  cursor: loading ? "not-allowed" : "pointer",
                  minWidth: 160,
                  boxShadow: loading
                    ? "none"
                    : "0 4px 12px rgba(0, 255, 255, 0.4)",
                  transition: "background-color 0.2s",
                }}
                title={loading ? "Generating..." : "Generate tickets"}
              >
                Generate Tickets
              </button>
            ) : (
              <button
                type="submit"
                disabled={prdLoading || !goal.trim()}
                style={{
                  padding: "12px 16px",
                  fontSize: "1.05rem",
                  borderRadius: "8px",
                  border: "1px solid " + (prdLoading ? "#2a2f45" : "transparent"),
                  backgroundColor: prdLoading ? "#2a2a42" : "#00bcd4",
                  color: prdLoading ? "#8aa0ac" : "white",
                  cursor: prdLoading ? "not-allowed" : "pointer",
                  minWidth: 160,
                  boxShadow: prdLoading
                    ? "none"
                    : "0 4px 12px rgba(0, 255, 255, 0.4)",
                  transition: "background-color 0.2s",
                }}
                title={prdLoading ? "Generating PRD..." : "Generate PRD"}
              >
                Generate PRD
              </button>
            )}
          </div>
        </form>

        {/* Error */}
        {error && (
          <div
            style={{
              marginTop: "14px",
              color: "red",
              fontWeight: "600",
              maxWidth: 900,
            }}
          >
            {error}
          </div>
        )}

        {/* CONTENT */}
        {activeTab === "tickets" ? (
          <TicketsView
            epics={epics}
            selected={selected}
            storyIndex={storyIndex}
            dependentsIndex={dependentsIndex}
            totalSelected={totalSelected}
            exportSelected={exportSelected}
            setEpicSelection={setEpicSelection}
            onUpdateStory={onUpdateStory}
            onUpdateEpic={onUpdateEpic}
            toggleSelectStory={toggleSelectStory}
            addDepsToSelection={addDepsToSelection}
          />
        ) : (
          <PrdView
            prd={prd}
            setPrd={setPrd}
            prdMarkdown={prdMarkdown}
            setPrdMarkdown={setPrdMarkdown}
            showJsonEditor={showJsonEditor}
            setShowJsonEditor={setShowJsonEditor}
          />
        )}
      </div>
    </>
  );
}

/* ----- Tickets Tab (Epics & Stories) ----- */
function TicketsView({
  epics,
  selected,
  storyIndex,
  dependentsIndex,
  totalSelected,
  exportSelected,
  setEpicSelection,
  onUpdateStory,
  onUpdateEpic,
  toggleSelectStory,
  addDepsToSelection,
}) {
  return (
    <>
      {/* Export toolbar */}
      {epics.length > 0 && (
        <div
          style={{
            marginTop: 16,
            width: "100%",
            maxWidth: 900,
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            gap: 12,
          }}
        >
          <div style={{ opacity: 0.9 }}>
            Selected: <strong>{totalSelected}</strong>
          </div>
          <button
            disabled={totalSelected === 0}
            onClick={exportSelected}
            style={{
              padding: "10px 16px",
              borderRadius: 8,
              border: "1px solid #00bcd4",
              background: totalSelected === 0 ? "#2a2a42" : "#113a44",
              color: totalSelected === 0 ? "#8aa0ac" : "#bdf2ff",
              cursor: totalSelected === 0 ? "not-allowed" : "pointer",
            }}
          >
            Export Selected (JSON)
          </button>
        </div>
      )}

      {/* Epics & Stories */}
      <div
        style={{
          marginTop: "20px",
          width: "100%",
          maxWidth: 900,
        }}
      >
        {epics.map((epic) => {
          const sortedStories = topoSortStories(epic.stories || []);
          const allIds = sortedStories.map((s) => s.storyId) || [];
          const allSelected =
            allIds.length > 0 && allIds.every((id) => selected.has(id));
          const someSelected = allIds.some((id) => selected.has(id));
          return (
            <EpicCard
              key={epic.epicId}
              epic={{ ...epic, stories: sortedStories }}
              storyIndex={storyIndex}
              dependentsIndex={dependentsIndex}
              selected={selected}
              onToggleEpic={(checked) => setEpicSelection(epic, checked)}
              epicSelectState={{ allSelected, someSelected }}
              onUpdateStory={onUpdateStory}
              onUpdateEpic={onUpdateEpic}
              onToggleStory={toggleSelectStory}
              onAddDeps={addDepsToSelection}
            />
          );
        })}
      </div>
    </>
  );
}

/* ----- PRD Tab (Markdown view + optional JSON editor) ----- */
function PrdView({ prd, setPrd, prdMarkdown, setPrdMarkdown, showJsonEditor, setShowJsonEditor }) {
  const [raw, setRaw] = useState(() => (prd ? JSON.stringify(prd, null, 2) : ""));
  const [parseErr, setParseErr] = useState("");

  React.useEffect(() => {
    if (prd) setRaw(JSON.stringify(prd, null, 2));
  }, [prd]);

  function handleApply() {
    setParseErr("");
    try {
      const obj = JSON.parse(raw);
      setPrd(obj);
    } catch (e) {
      setParseErr(e.message || "Invalid JSON");
    }
  }

  function handleDownloadJSON() {
    try {
      const obj = prd ?? JSON.parse(raw);
      downloadJSON("prd.json", obj);
    } catch {
      const blob = new Blob([raw], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "prd-raw.json";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }
  }

  function handleDownloadMarkdown() {
    const blob = new Blob([prdMarkdown || ""], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "prd.md";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  const html = mdToHtml(prdMarkdown || "");

  return (
    <div style={{ width: "100%", maxWidth: 1000, marginTop: 16 }}>
      {!prdMarkdown && (
        <div style={{ marginBottom: 10, color: "#9fb3c8" }}>
          Tip: enter a goal above and click <strong>Generate PRD</strong>.
        </div>
      )}

      {/* Pretty PRD render */}
      {prdMarkdown && (
        <div
          style={{
            background: "linear-gradient(180deg, #14162c 0%, #0f1122 100%)",
            border: "1px solid #1f2540",
            borderRadius: 14,
            padding: 20,
            boxShadow: "0 6px 28px rgba(0, 255, 255, 0.06)",
          }}
        >
          <div
            style={{
              lineHeight: 1.65,
              color: "#e6f3ff",
            }}
          >
            <style>{`
              .prd h1{font-size:1.9rem;margin:0 0 12px;color:#9be7ff}
              .prd h2{font-size:1.4rem;margin:18px 0 8px;color:#80e8ff}
              .prd h3{font-size:1.15rem;margin:14px 0 6px;color:#bdefff}
              .prd h4{font-size:1rem;margin:10px 0 4px;color:#d8f6ff}
              .prd p{margin:8px 0;color:#e6f3ff}
              .prd ul{margin:6px 0 10px 20px;padding-left:16px}
              .prd li{margin:4px 0}
              .toolbar{display:flex;gap:8px;margin-bottom:10px}
            `}</style>

            <div className="toolbar" style={{ display: "flex", gap: 8, marginBottom: 12 }}>
              <button onClick={handleDownloadMarkdown} style={btnPrimary}>
                Download Markdown
              </button>
              <button onClick={() => setShowJsonEditor((v) => !v)} style={btnSecondary}>
                {showJsonEditor ? "Hide JSON" : "Show JSON"}
              </button>
              <button onClick={handleDownloadJSON} style={btnSecondary}>
                Download JSON
              </button>
            </div>

            <div
              className="prd"
              dangerouslySetInnerHTML={{ __html: html }}
              style={{}}
            />
          </div>
        </div>
      )}

      {/* Optional JSON editor (kept for power users; hidden by default) */}
      {showJsonEditor && (
        <div style={{ marginTop: 14 }}>
          <textarea
            value={raw}
            onChange={(e) => setRaw(e.target.value)}
            placeholder={
              prd
                ? "Edit PRD JSON…"
                : "PRD JSON will appear here once generated. You can also paste your own JSON and click Apply."
            }
            rows={24}
            style={{
              width: "100%",
              padding: "12px 14px",
              borderRadius: 10,
              border: "1px solid #2b3252",
              background: "#121428",
              color: "#e6f3ff",
              outline: "none",
              fontFamily:
                "ui-monospace, SFMono-Regular, Menlo, Monaco, 'Courier New', monospace",
              fontSize: 13,
              lineHeight: 1.5,
              boxShadow: "inset 0 0 5px #00ffff20",
            }}
          />
          {parseErr && (
            <div style={{ color: "#ff8a8a", marginTop: 8 }}>
              Parse error: {parseErr}
            </div>
          )}
          <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
            <button onClick={handleApply} style={btnPrimary}>
              Apply Changes
            </button>
            <button onClick={handleDownloadJSON} style={btnSecondary}>
              Download JSON
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

/* ----- Cards ----- */
function EpicCard({
  epic,
  storyIndex,
  dependentsIndex,
  selected,
  onToggleEpic,
  epicSelectState,
  onUpdateStory,
  onUpdateEpic,
  onToggleStory,
  onAddDeps,
}) {
  const area = epic.area?.trim();
  const { allSelected, someSelected } = epicSelectState;

  const [editingEpic, setEditingEpic] = useState(false);
  const [epicForm, setEpicForm] = useState({
    epic: epic.epic || "",
    description: epic.description || "",
    area: epic.area || "Product",
  });

  function saveEpic() {
    onUpdateEpic(epic.epicId, {
      epic: epicForm.epic,
      description: epicForm.description,
      area: epicForm.area,
    });
    setEditingEpic(false);
  }

  return (
    <div
      style={{
        backgroundColor: "#1a1a2e",
        borderRadius: 12,
        padding: 20,
        marginBottom: 30,
        boxShadow: "0 2px 12px rgba(0, 255, 255, 0.1)",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 12,
          marginBottom: 10,
          justifyContent: "space-between",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          {!editingEpic ? (
            <>
              <h2 style={{ margin: 0, color: "#00ffff" }}>{epic.epic}</h2>
              {area && <Badge label={area} variant="outline" />}
            </>
          ) : (
            <input
              value={epicForm.epic}
              onChange={(e) =>
                setEpicForm((p) => ({ ...p, epic: e.target.value }))
              }
              placeholder="Epic title"
              style={inputMini}
            />
          )}
        </div>

        {/* Epic-level select all */}
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <label
            style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 13 }}
            title="Select all stories in epic"
          >
            <input
              type="checkbox"
              checked={allSelected}
              ref={(el) => {
                if (el) el.indeterminate = !allSelected && someSelected;
              }}
              onChange={(e) => onToggleEpic(e.target.checked)}
            />
            Select all
          </label>
          {!editingEpic ? (
            <button onClick={() => setEditingEpic(true)} style={btnGhost}>
              Edit epic
            </button>
          ) : (
            <>
              <button onClick={saveEpic} style={btnPrimary}>
                Save
              </button>
              <button onClick={() => setEditingEpic(false)} style={btnGhost}>
                Cancel
              </button>
            </>
          )}
        </div>
      </div>

      {!editingEpic ? (
        <p
          style={{
            fontStyle: "italic",
            color: "#bbb",
            marginTop: 6,
            marginBottom: 16,
            whiteSpace: "pre-wrap",
          }}
        >
          {epic.description}
        </p>
      ) : (
        <textarea
          value={epicForm.description}
          onChange={(e) =>
            setEpicForm((p) => ({ ...p, description: e.target.value }))
          }
          placeholder="Epic description"
          rows={2}
          style={textarea}
        />
      )}

      {epic.stories?.map((story) => (
        <StoryCard
          key={story.storyId}
          epic={epic}
          story={story}
          storyIndex={storyIndex}
          dependentsIndex={dependentsIndex}
          selected={selected}
          onToggleStory={onToggleStory}
          onUpdateStory={onUpdateStory}
          onAddDeps={onAddDeps}
        />
      ))}
    </div>
  );
}

function StoryCard({
  epic,
  story,
  storyIndex,
  dependentsIndex,
  selected,
  onToggleStory,
  onUpdateStory,
  onAddDeps,
}) {
  const [editing, setEditing] = useState(false);

  const [form, setForm] = useState(() => ({
    summary: story.summary || "",
    description: story.description || "",
    estimate: story.estimate || "",
    type: story.type || "",
    priority: story.priority || "",
    ownerRole: story.ownerRole || "",
    labels: Array.isArray(story.labels) ? story.labels.join(", ") : "",
    acceptanceCriteria: Array.isArray(story.acceptanceCriteria)
      ? story.acceptanceCriteria.join("\n")
      : "",
    testCases: Array.isArray(story.testCases) ? story.testCases.join("\n") : "",
  }));

  const deps = Array.isArray(story.dependencies) ? story.dependencies : [];
  const dependents = dependentsIndex[story.storyId] || [];
  const isSelected = selected.has(story.storyId);

  function handleChange(field, value) {
    setForm((prev) => ({ ...prev, [field]: value }));
  }

  function handleSave() {
    const updated = {
      ...story,
      summary: form.summary.trim(),
      description: form.description.trim(),
      estimate: form.estimate.trim(),
      type: form.type.trim(),
      priority: form.priority.trim(),
      ownerRole: form.ownerRole.trim(),
      labels: form.labels
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean),
      acceptanceCriteria: form.acceptanceCriteria
        .split("\n")
        .map((s) => s.trim())
        .filter(Boolean),
      testCases: form.testCases
        .split("\n")
        .map((s) => s.trim())
        .filter(Boolean),
    };
    onUpdateStory(epic.epicId, story.storyId, updated);
    setEditing(false);
  }

  const priorityVariant = (story.priority || "").toLowerCase();
  const leftBorderColor = deps.length > 0 ? "#caa84a" : "#1aa3a3";

  return (
    <div
      id={story.storyId || undefined}
      style={{
        backgroundColor: "#26263a",
        borderRadius: 10,
        padding: 16,
        marginBottom: 16,
        boxShadow: "inset 0 0 5px #00ffff40",
        borderLeft: `3px solid ${leftBorderColor}`,
      }}
    >
      {/* Selection + Summary row */}
      <div
        style={{
          display: "flex",
          alignItems: "flex-start",
          gap: 12,
          marginBottom: 8,
          color: "#0ff",
        }}
      >
        <input
          type="checkbox"
          checked={isSelected}
          onChange={() => onToggleStory(story.storyId)}
          style={{ marginTop: 4 }}
          title="Select story"
        />
        <div style={{ flex: 1 }}>
          {!editing ? (
            <>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  gap: 12,
                  marginBottom: 6,
                }}
              >
                <div style={{ fontWeight: 700, fontSize: "1.05rem" }}>
                  {story.summary}
                </div>
                <div
                  style={{
                    display: "flex",
                    gap: 8,
                    alignItems: "center",
                    flexShrink: 0,
                  }}
                >
                  {story.type ? (
                    <Badge
                      label={story.type}
                      variant={
                        (story.type || "").toLowerCase() === "design"
                          ? "design"
                          : "eng"
                      }
                    />
                  ) : (
                    <Badge label="Type: N/A" variant="ghost" />
                  )}

                  {story.priority ? (
                    <Badge label={story.priority} variant={priorityVariant} />
                  ) : (
                    <Badge label="P?" variant="ghost" />
                  )}

                  <Badge
                    label={`Est: ${story.estimate || "N/A"}`}
                    variant="outline"
                  />
                </div>
              </div>
            </>
          ) : (
            <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
              <input
                value={form.summary}
                onChange={(e) => handleChange("summary", e.target.value)}
                placeholder="Summary"
                style={inputMini}
              />
              <input
                value={form.estimate}
                onChange={(e) => handleChange("estimate", e.target.value)}
                placeholder="Estimate (e.g., 2d, 6h)"
                style={inputMini}
              />
            </div>
          )}

          {editing && (
            <div
              style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: 8 }}
            >
              <select
                value={form.type}
                onChange={(e) => handleChange("type", e.target.value)}
                style={selectMini}
              >
                <option value="">Type…</option>
                <option>Design</option>
                <option>Engineering</option>
              </select>
              <select
                value={form.priority}
                onChange={(e) => handleChange("priority", e.target.value)}
                style={selectMini}
              >
                <option value="">Priority…</option>
                <option>P0</option>
                <option>P1</option>
                <option>P2</option>
              </select>
              <input
                value={form.ownerRole}
                onChange={(e) => handleChange("ownerRole", e.target.value)}
                placeholder="Owner role (e.g., UX, Backend)"
                style={inputMini}
              />
              <input
                value={form.labels}
                onChange={(e) => handleChange("labels", e.target.value)}
                placeholder="labels (comma-separated)"
                style={{ ...inputMini, minWidth: 220 }}
              />
            </div>
          )}

          {!editing ? (
            <p style={{ marginBottom: 8, whiteSpace: "pre-wrap", color: "#eee" }}>
              {story.description}
            </p>
          ) : (
            <textarea
              value={form.description}
              onChange={(e) => handleChange("description", e.target.value)}
              placeholder="User story description"
              rows={3}
              style={textarea}
            />
          )}

          {!editing && (
            <>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  marginTop: 6,
                }}
              >
                <div style={{ fontWeight: 600, color: "#89e0ff", minWidth: 90 }}>
                  Blocked by:
                </div>
                <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                  {deps.length > 0 ? (
                    deps.map((id) => {
                      const found = storyIndex[id];
                      const label = found?.summary || id;
                      return (
                        <a
                          key={id}
                          href={`#${id}`}
                          style={depChipStyle}
                          title={`Jump to ${id}`}
                        >
                          → {label}
                        </a>
                      );
                    })
                  ) : (
                    <span style={{ color: "#9fb3c8" }}>None</span>
                  )}
                </div>
                {deps.length > 0 && (
                  <button
                    onClick={() => onAddDeps(deps)}
                    style={{ ...btnGhost, marginLeft: "auto" }}
                    title="Add all prerequisites to selection"
                  >
                    Add deps
                  </button>
                )}
              </div>

              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  marginTop: 6,
                }}
              >
                <div style={{ fontWeight: 600, color: "#c6ffdd", minWidth: 90 }}>
                  Blocks:
                </div>
                <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                  {dependents.length > 0 ? (
                    dependents.map((id) => {
                      const found = storyIndex[id];
                      const label = found?.summary || id;
                      return (
                        <a
                          key={id}
                          href={`#${id}`}
                          style={depChipStyle}
                          title={`Jump to ${id}`}
                        >
                          {label}
                        </a>
                      );
                    })
                  ) : (
                    <span style={{ color: "#9fb3c8" }}>None</span>
                  )}
                </div>
              </div>
            </>
          )}

          {!editing ? (
            <AcceptanceCriteria criteria={story.acceptanceCriteria} />
          ) : (
            <div style={{ marginTop: 8 }}>
              <strong>Acceptance Criteria:</strong>
              <textarea
                value={form.acceptanceCriteria}
                onChange={(e) => handleChange("acceptanceCriteria", e.target.value)}
                placeholder={"One criterion per line"}
                rows={4}
                style={textarea}
              />
            </div>
          )}

          {!editing ? (
            <TestCases testCases={story.testCases} />
          ) : (
            <div style={{ marginTop: 8 }}>
              <strong>Test Cases:</strong>
              <textarea
                value={form.testCases}
                onChange={(e) => handleChange("testCases", e.target.value)}
                placeholder={"One test case per line"}
                rows={4}
                style={textarea}
              />
            </div>
          )}

          <div style={{ marginTop: 10, display: "flex", gap: 8 }}>
            {!editing ? (
              <button onClick={() => setEditing(true)} style={btnSecondary}>
                Edit
              </button>
            ) : (
              <>
                <button onClick={handleSave} style={btnPrimary}>
                  Save
                </button>
                <button onClick={() => setEditing(false)} style={btnGhost}>
                  Cancel
                </button>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ----- Small UI bits ----- */
function Badge({ label, variant = "outline" }) {
  let styles = {
    display: "inline-block",
    padding: "4px 8px",
    borderRadius: 999,
    fontSize: 12,
    border: "1px solid #00ffff55",
    color: "#c7f5ff",
    background: "transparent",
  };

  if (variant === "design") {
    styles = {
      ...styles,
      border: "1px solid #ff6bff55",
      color: "#ffd6ff",
      background: "#ff6bff11",
    };
  } else if (variant === "eng") {
    styles = {
      ...styles,
      border: "1px solid #64ffda55",
      color: "#d6fff3",
      background: "#64ffda11",
    };
  } else if (variant === "p0") {
    styles = {
      ...styles,
      border: "1px solid #ff525255",
      color: "#ffd6d6",
      background: "#ff525211",
    };
  } else if (variant === "p1") {
    styles = {
      ...styles,
      border: "1px solid #ffbf0055",
      color: "#fff1cc",
      background: "#ffbf0011",
    };
  } else if (variant === "p2") {
    styles = {
      ...styles,
      border: "1px solid #7aa2f755",
      color: "#d9e6ff",
      background: "#7aa2f711",
    };
  } else if (variant === "ghost") {
    styles = {
      ...styles,
      border: "1px solid #3a3a52",
      color: "#b8c7ff",
      background: "#2a2a42",
    };
  } else if (variant === "outline") {
    styles = {
      ...styles,
      border: "1px solid #00ffff55",
      color: "#c7f5ff",
      background: "transparent",
    };
  }

  return <span style={styles}>{label}</span>;
}

function AcceptanceCriteria({ criteria }) {
  if (!criteria || criteria.length === 0) return null;
  return (
    <div style={{ marginTop: 8 }}>
      <strong>Acceptance Criteria:</strong>
      <ul style={{ marginTop: 4 }}>
        {criteria.map((c, i) => (
          <li key={i}>{c}</li>
        ))}
      </ul>
    </div>
  );
}

function TestCases({ testCases }) {
  if (!testCases || testCases.length === 0) return null;
  return (
    <div style={{ marginTop: 8 }}>
      <strong>Test Cases:</strong>
      <ul style={{ marginTop: 4 }}>
        {testCases.map((tc, i) => (
          <li key={i}>{tc}</li>
        ))}
      </ul>
    </div>
  );
}

/* ----- Tiny style objects for editors/buttons ----- */
const inputMini = {
  flex: 1,
  minWidth: 160,
  padding: "8px 10px",
  borderRadius: 6,
  border: "1px solid #333",
  background: "#1e1e2f",
  color: "#fff",
  outline: "none",
};

const selectMini = {
  ...inputMini,
  minWidth: 140,
};

const textarea = {
  width: "100%",
  padding: "10px 12px",
  borderRadius: 8,
  border: "1px solid #333",
  background: "#1e1e2f",
  color: "#fff",
  outline: "none",
  resize: "vertical",
};

const btnPrimary = {
  padding: "8px 12px",
  borderRadius: 8,
  border: "1px solid #00bcd4",
  background: "#0d3a44",
  color: "#c7f5ff",
  cursor: "pointer",
};

const btnSecondary = {
  padding: "8px 12px",
  borderRadius: 8,
  border: "1px solid #3a3a52",
  background: "#2a2a42",
  color: "#d9e6ff",
  cursor: "pointer",
};

const btnGhost = {
  padding: "8px 12px",
  borderRadius: 8,
  border: "1px solid transparent",
  background: "transparent",
  color: "#9fb3c8",
  cursor: "pointer",
};

/* Dependency chip style */
const depChipStyle = {
  display: "inline-block",
  padding: "4px 10px",
  borderRadius: "999px",
  fontSize: "0.85rem",
  backgroundColor: "#1e2a3a",
  color: "#cfe8ff",
  border: "1px solid #2b3b52",
  cursor: "pointer",
  textDecoration: "none",
};

export default MainApp;

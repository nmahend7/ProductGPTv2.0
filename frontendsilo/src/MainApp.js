// MainApp.jsx
import React, { useState } from "react";

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
  0% {
    text-shadow: 0 0 8px #0ff, 0 0 12px #00f, 0 0 20px #0ff;
    color: #0ff;
  }
  25% {
    text-shadow: 0 0 8px #ff00ff, 0 0 14px #f0f, 0 0 22px #f0f;
    color: #ff00ff;
  }
  50% {
    text-shadow: 0 0 8px #0f0, 0 0 14px #0f0, 0 0 22px #0f0;
    color: #0f0;
  }
  75% {
    text-shadow: 0 0 10px #00f, 0 0 14px #00f, 0 0 22px #00f;
    color: #00f;
  }
  100% {
    text-shadow: 0 0 8px #0ff, 0 0 12px #00f, 0 0 20px #0ff;
    color: #0ff;
  }
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

function MainApp({ user, onLogout }) {
  const [goal, setGoal] = useState("");
  const [epics, setEpics] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const logoStyle = {
    fontSize: "3rem",
    fontWeight: "bold",
    marginBottom: "20px",
    textAlign: "center",
    animation: "glow 6s infinite ease-in-out",
    letterSpacing: "6px",
  };

  async function handleSubmit(e) {
    e.preventDefault();
    if (!goal.trim()) return;
    setLoading(true);
    setError("");
    setEpics([]);

    try {
      const res = await fetch("http://localhost:5000/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ goal }),
      });
      const data = await res.json();

      if (res.ok) {
        setEpics(data.epics);
      } else {
        setError(data.error || "Unknown error");
      }
    } catch (err) {
      setError("Failed to connect to backend");
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <style>{pulseKeyframes}</style>
      <div style={orbStyle}></div>

      <div
        style={{
          minHeight: "100vh",
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
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

        <h1 style={logoStyle}>SILO</h1>

        <form onSubmit={handleSubmit} style={{ width: "100%", maxWidth: 600 }}>
          <input
            type="text"
            placeholder="Enter your product goal..."
            value={goal}
            onChange={(e) => setGoal(e.target.value)}
            style={{
              width: "100%",
              padding: "12px 16px",
              fontSize: "1.2rem",
              borderRadius: "8px",
              border: "1px solid #333",
              backgroundColor: "#1e1e2f",
              color: "#fff",
              boxShadow: "0 2px 5px rgb(0 0 0 / 0.3)",
            }}
          />
          <button
            type="submit"
            disabled={loading}
            style={{
              marginTop: "12px",
              padding: "12px 20px",
              fontSize: "1.1rem",
              borderRadius: "8px",
              border: "none",
              backgroundColor: "#00bcd4",
              color: "white",
              cursor: "pointer",
              width: "100%",
              boxShadow: "0 4px 12px rgba(0, 255, 255, 0.4)",
              transition: "background-color 0.3s",
            }}
          >
            {loading ? "Generating..." : "Generate Tickets"}
          </button>
        </form>

        {error && (
          <div
            style={{
              marginTop: "20px",
              color: "red",
              fontWeight: "600",
            }}
          >
            {error}
          </div>
        )}

        <div
          style={{
            marginTop: "40px",
            width: "100%",
            maxWidth: 900,
          }}
        >
          {epics.map((epic, i) => (
            <EpicCard key={i} epic={epic} />
          ))}
        </div>
      </div>
    </>
  );
}

function EpicCard({ epic }) {
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
      <h2 style={{ marginBottom: 6, color: "#00ffff" }}>{epic.epic}</h2>
      <p
        style={{
          fontStyle: "italic",
          color: "#bbb",
          marginBottom: 16,
          whiteSpace: "pre-wrap",
        }}
      >
        {epic.description}
      </p>
      {epic.stories.map((story, i) => (
        <StoryCard key={i} story={story} />
      ))}
    </div>
  );
}

function StoryCard({ story }) {
  return (
    <div
      style={{
        backgroundColor: "#26263a",
        borderRadius: 10,
        padding: 16,
        marginBottom: 16,
        boxShadow: "inset 0 0 5px #00ffff40",
      }}
    >
      <div
        style={{
          fontWeight: "700",
          fontSize: "1.1rem",
          marginBottom: 6,
          display: "flex",
          justifyContent: "space-between",
          color: "#0ff",
        }}
      >
        <div>{story.summary}</div>
        <div
          style={{
            fontWeight: "600",
            fontSize: "0.9rem",
            fontStyle: "italic",
            color: "#00bcd4",
          }}
        >
          Estimate: {story.estimate || "N/A"}
        </div>
      </div>
      <p style={{ marginBottom: 8, whiteSpace: "pre-wrap", color: "#eee" }}>
        {story.description}
      </p>

      <AcceptanceCriteria criteria={story.acceptanceCriteria} />
      <TestCases testCases={story.testCases} />
    </div>
  );
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

export default MainApp;

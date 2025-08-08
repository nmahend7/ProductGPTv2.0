// src/Login.js
import React from "react";
import { signInWithPopup } from "firebase/auth";
import { auth, provider } from "./firebase";

function Login({ onLogin }) {
  const handleLogin = async () => {
    try {
      const result = await signInWithPopup(auth, provider);
      onLogin(result.user);
    } catch (err) {
      console.error("Login failed:", err);
    }
  };

  return (
    <div
      style={{
        height: "100vh",
        background: "radial-gradient(ellipse at center, #0f0f2d, #000)",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        color: "white",
        flexDirection: "column",
      }}
    >
      <h1 style={{ fontSize: "3rem", marginBottom: "2rem" }}>Welcome to Silo</h1>
      <button
        onClick={handleLogin}
        style={{
          padding: "12px 24px",
          fontSize: "1.2rem",
          backgroundColor: "#4285F4",
          color: "white",
          border: "none",
          borderRadius: "8px",
          cursor: "pointer",
          boxShadow: "0 4px 10px rgba(0,0,0,0.3)",
        }}
      >
        Sign in with Google
      </button>
    </div>
  );
}

export default Login;

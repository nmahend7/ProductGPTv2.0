// App.js
import React, { useEffect, useState } from "react";
import { onAuthStateChanged, signOut } from "firebase/auth";
import { auth } from "./firebase";
import Login from "./Login";
import MainApp from "./MainApp";

function App() {
  const [user, setUser] = useState(null);
  const [checking, setChecking] = useState(true);

  useEffect(() => {
    const unsub = onAuthStateChanged(auth, (u) => {
      setUser(u);
      setChecking(false);
    });
    return () => unsub();
  }, []);

  if (checking) return null; // or a loading screen

  return user ? (
    <MainApp user={user} onLogout={() => signOut(auth)} />
  ) : (
    <Login onLogin={setUser} />
  );
}

export default App;

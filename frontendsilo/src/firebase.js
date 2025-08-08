// src/firebase.js
import { initializeApp } from "firebase/app";
import { getAuth, GoogleAuthProvider } from "firebase/auth";

const firebaseConfig = {
  apiKey: "AIzaSyC_e0ZC46_PosTIlr2iQE09e-InHygmqZE",
  authDomain: "silo-431c3.firebaseapp.com",
  projectId: "silo-431c3",
  storageBucket: "silo-431c3.firebasestorage.app",
  messagingSenderId: "595483104145",
  appId: "1:595483104145:web:1a86d2cbf345f4dc1a0030",
  measurementId: "G-RFZJHEPXJL"
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const provider = new GoogleAuthProvider();

export { auth, provider };

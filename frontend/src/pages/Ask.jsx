import { useState } from "react";
import { api } from "../api.js";

export default function Ask() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function submit() {
    setLoading(true); setError(""); setAnswer(null);
    try {
      const result = await api.ask(question);
      setAnswer(result);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <h1>Ask the Codebase</h1>
      <p className="subtitle">Retrieval-augmented Q&amp;A over the indexed code.</p>
      <div className="card">
        <label>Your question</label>
        <textarea value={question} onChange={(e) => setQuestion(e.target.value)}
          placeholder="How does the config module work?" />
        <button onClick={submit} disabled={loading || !question.trim()}>
          {loading ? "Asking..." : "Ask"}
        </button>
        {error && <div className="error">{error}</div>}
      </div>
      {answer && (
        <div className="card">
          <h2>Answer <span className="badge">model: {answer.model}</span></h2>
          <pre>{answer.answer}</pre>
          <div style={{ color: "var(--muted)", fontSize: 13, marginTop: 8 }}>
            Sources used: {answer.sources.length}
          </div>
        </div>
      )}
    </div>
  );
}
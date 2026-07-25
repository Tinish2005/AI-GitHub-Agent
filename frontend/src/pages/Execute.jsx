import { useState } from "react";
import { api } from "../api.js";

export default function Execute() {
  const [goal, setGoal] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function submit() {
    setLoading(true); setError(""); setResult(null);
    try {
      setResult(await api.execute(goal));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <h1>Execute</h1>
      <p className="subtitle">Plan a goal and run every step end-to-end.</p>
      <div className="card">
        <label>Goal</label>
        <input value={goal} onChange={(e) => setGoal(e.target.value)}
          placeholder="Review the authentication module" />
        <button onClick={submit} disabled={loading || !goal.trim()}>
          {loading ? "Executing..." : "Plan + Execute"}
        </button>
        {error && <div className="error">{error}</div>}
      </div>
      {result && (
        <div className="card">
          <h2>Result
            <span className="badge">completed: {result.completed}</span>
            <span className="badge">failed: {result.failed}</span>
            <span className="badge">aborted: {String(result.aborted)}</span>
          </h2>
          {result.steps.map((s) => (
            <div className="step" key={s.step_id}>
              <span className="kind">{s.status} - {s.kind}</span>
              <div>{s.output || s.error || "(no output)"}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
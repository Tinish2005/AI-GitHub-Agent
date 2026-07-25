import { useState } from "react";
import { api } from "../api.js";

export default function Plan() {
  const [goal, setGoal] = useState("");
  const [plan, setPlan] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function submit() {
    setLoading(true); setError(""); setPlan(null);
    try {
      setPlan(await api.plan(goal));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <h1>Plan</h1>
      <p className="subtitle">Turn a goal into a structured multi-step plan.</p>
      <div className="card">
        <label>Goal</label>
        <input value={goal} onChange={(e) => setGoal(e.target.value)}
          placeholder="Fix the login bug that crashes on empty password" />
        <button onClick={submit} disabled={loading || !goal.trim()}>
          {loading ? "Planning..." : "Create Plan"}
        </button>
        {error && <div className="error">{error}</div>}
      </div>
      {plan && (
        <div className="card">
          <h2>Plan <span className="badge">{plan.strategy}</span><span className="badge">{plan.steps.length} steps</span></h2>
          {plan.steps.map((s) => (
            <div className="step" key={s.id}>
              <span className="kind">{s.kind}</span>
              <div>{s.id}. {s.description}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
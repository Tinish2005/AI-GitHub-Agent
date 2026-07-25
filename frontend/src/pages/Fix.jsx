import { useState } from "react";
import { api } from "../api.js";

export default function Fix() {
  const [goal, setGoal] = useState("");
  const [proposal, setProposal] = useState(null);
  const [validation, setValidation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function submit() {
    setLoading(true); setError(""); setProposal(null); setValidation(null);
    try {
      const p = await api.proposeFix(goal);
      setProposal(p);
      const v = await api.validateFix(goal);
      setValidation(v);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <h1>Propose Fix</h1>
      <p className="subtitle">Generate a code fix as a diff, then validate it in a sandbox.</p>
      <div className="card">
        <label>Bug / goal</label>
        <textarea value={goal} onChange={(e) => setGoal(e.target.value)}
          placeholder="Fix a bug where the calculator returns the wrong sum" />
        <button onClick={submit} disabled={loading || !goal.trim()}>
          {loading ? "Working..." : "Propose + Validate"}
        </button>
        {error && <div className="error">{error}</div>}
      </div>
      {proposal && (
        <div className="card">
          <h2>Proposal <span className="badge">confidence: {proposal.confidence.toFixed(2)}</span><span className="badge">valid: {String(proposal.is_valid)}</span></h2>
          <p style={{ marginBottom: 12 }}>{proposal.explanation}</p>
          <pre>{proposal.diff}</pre>
        </div>
      )}
      {validation && (
        <div className="card">
          <h2>Validation <span className="badge">passed: {String(validation.passed)}</span><span className="badge">score: {validation.score.toFixed(2)}</span></h2>
          {validation.checks.map((c, i) => (
            <div className="step" key={i}>
              <span className="kind">{c.skipped ? "skipped" : c.passed ? "pass" : "fail"}</span>
              <div>{c.name}: {c.message}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
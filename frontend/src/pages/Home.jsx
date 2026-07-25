export default function Home() {
  return (
    <div>
      <h1>AI GitHub Agent</h1>
      <p className="subtitle">
        An agent that understands GitHub repositories, answers questions,
        plans multi-step work, and proposes fixes as draft pull requests.
      </p>

      <div className="stat-grid">
        <div className="stat"><div className="num">11</div><div className="lbl">REST endpoints</div></div>
        <div className="stat"><div className="num">12</div><div className="lbl">MCP tools</div></div>
        <div className="stat"><div className="num">314</div><div className="lbl">tests passing</div></div>
      </div>

      <div className="card" style={{ marginTop: 24 }}>
        <h2>What it does</h2>
        <div className="step"><span className="kind">RAG</span><div>Ask questions about a codebase, get answers with citations.</div></div>
        <div className="step"><span className="kind">Plan</span><div>Turn a plain-English goal into a structured multi-step plan.</div></div>
        <div className="step"><span className="kind">Fix</span><div>Generate a code fix as a unified diff and validate it.</div></div>
        <div className="step"><span className="kind">Execute</span><div>Run the whole plan end-to-end with per-step reporting.</div></div>
      </div>

      <div className="card">
        <h2>Architecture</h2>
        <p style={{ color: "var(--muted)", fontSize: 14, lineHeight: 1.6 }}>
          Combines RAG (retrieval), MCP (Model Context Protocol), and agent
          reasoning. Two pipelines: an offline indexing pipeline (clone, parse,
          embed, store) and an online runtime pipeline (plan, execute, validate,
          draft PR). Every proposed change is a draft PR reviewed by a human.
        </p>
      </div>
    </div>
  );
}
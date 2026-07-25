import { useState } from "react";
import { api } from "../api.js";

export default function IndexRepo() {
  const [url, setUrl] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function parseUrl(raw) {
    const cleaned = raw.trim().replace(/^https?:\/\/github\.com\//, "").replace(/\.git$/, "");
    const parts = cleaned.split("/").filter(Boolean);
    if (parts.length < 2) return null;
    return { owner: parts[0], repo: parts[1] };
  }

  async function submit() {
    const parsed = parseUrl(url);
    if (!parsed) {
      setError("Enter a valid GitHub URL or owner/repo");
      return;
    }
    setLoading(true); setError(""); setResult(null);
    try {
      const res = await api.indexGithub(parsed.owner, parsed.repo, 10);
      setResult(res);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <h1>Index a GitHub Repo</h1>
      <p className="subtitle">
        Fetches a repo Python files via the GitHub API and indexes them.
        Then use Ask / Plan / Fix on that repo.
      </p>
      <div className="card">
        <label>GitHub URL or owner/repo</label>
        <input value={url} onChange={(e) => setUrl(e.target.value)}
          placeholder="https://github.com/Tinish2005/AI-GitHub-Agent" />
        <button onClick={submit} disabled={loading || !url.trim()}>
          {loading ? "Indexing (takes a minute)..." : "Index Repo"}
        </button>
        {error && <div className="error">{error}</div>}
      </div>
      {result && (
        <div className="card">
          <h2>Indexed
            <span className="badge">{result.owner}/{result.repo}</span>
            <span className="badge">branch: {result.ref}</span>
          </h2>
          <div className="stat-grid">
            <div className="stat"><div className="num">{result.files_indexed}</div><div className="lbl">files</div></div>
            <div className="stat"><div className="num">{result.chunks_indexed}</div><div className="lbl">chunks</div></div>
          </div>
          <p style={{ color: "var(--muted)", fontSize: 13, marginTop: 12 }}>
            Now go to Ask the Codebase and ask about this repo!
          </p>
        </div>
      )}
    </div>
  );
}

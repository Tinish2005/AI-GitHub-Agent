import { useEffect, useState } from 'react';
import { useLocation } from 'react-router-dom';
import { api, decodeFileContent } from '../api';
import { useAppStore } from '../store/appStore';
import { FormattedMarkdown } from '../components/FormattedOutput';
import ErrorNotice from '../components/ErrorNotice';

const normPath = (p) => String(p || '').replace(/\\/g, '/');

// One source of an answer. "Show code" fetches the file from GitHub and cuts out the cited lines.
function SourceRow({ index, source, target }) {
  const [code, setCode] = useState('');
  const [state, setState] = useState('idle'); // idle | loading | error
  const [open, setOpen] = useState(false);

  const toggle = async () => {
    if (open) {
      setOpen(false);
      return;
    }
    setOpen(true);
    if (code || state === 'loading') return;
    setState('loading');
    try {
      const file = await api.githubFile({ owner: target.owner, repo: target.repo, path: normPath(source.file_path), ref: target.ref });
      const lines = decodeFileContent(file).split('\n');
      const start = Math.max(1, source.start_line || 1);
      const end = Math.min(lines.length, source.end_line || start);
      setCode(lines.slice(start - 1, end).map((l, i) => `${String(start + i).padStart(4)}  ${l}`).join('\n'));
      setState('idle');
    } catch (err) {
      setState('error');
      setCode(`Could not load the code: ${err.message}`);
    }
  };

  return (
    <div className="step-card">
      <div className="row-between" style={{ alignItems: 'flex-start' }}>
        <div style={{ minWidth: 0 }}>
          <div style={{ fontSize: 12, fontWeight: 600 }}>
            <span className="muted">[{index + 1}]</span> <span className="mono">{source.qualified_name}</span>
          </div>
          <div className="mono muted" style={{ fontSize: 10.5, marginTop: 2 }}>
            {normPath(source.file_path)}:{source.start_line}-{source.end_line}
          </div>
        </div>
        <div style={{ textAlign: 'right', flexShrink: 0 }}>
          <span className="badge badge-gray" title="How far the code is from your question. Lower is closer. It is not a percentage.">
            distance {Number(source.distance).toFixed(3)}
          </span>
          <div>
            <button type="button" className="link-btn" onClick={toggle} style={{ marginTop: 4 }}>
              {open ? 'Hide code' : 'Show code'}
            </button>
          </div>
        </div>
      </div>
      {open && (state === 'loading' ? <div className="sub" style={{ marginTop: 6 }}>Loading the code from GitHub…</div> : <pre className="code-box">{code}</pre>)}
    </div>
  );
}

export default function Ask() {
  const { target, activeRepo, suggestions, logActivity, refreshStats, markUnreachable, aiLeft } = useAppStore();
  const location = useLocation();
  const [question, setQuestion] = useState('');
  const [topK, setTopK] = useState(5);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null); // { question, repo, answer, sources, model }
  const [error, setError] = useState(null);

  // "Ask about this file" on the dashboard fills the box but does not send it (sending uses an AI call).
  useEffect(() => {
    const prefill = location.state && location.state.prefill;
    if (prefill) setQuestion(prefill);
  }, [location.state]);

  const send = async (text) => {
    const q = (text ?? question).trim();
    if (!q || !target || loading) return;
    setQuestion(q);
    setLoading(true);
    setError(null);
    try {
      const data = await api.ask({ question: q, owner: target.owner, repo: target.repo, topK });
      setResult({ question: q, repo: activeRepo, answer: data.answer, sources: data.sources || [], model: data.model });
      logActivity('Ask', `Asked about ${activeRepo}: "${q.slice(0, 60)}"`);
    } catch (err) {
      setError(err);
      if (err.network) markUnreachable();
    } finally {
      setLoading(false);
      refreshStats();
    }
  };

  if (!target) {
    return <div className="pg"><div className="card muted" style={{ fontSize: 12.5 }}>Select or index a repository first.</div></div>;
  }

  return (
    <div className="pg">
      <div className="card" style={{ borderLeft: '3px solid #6366f1' }}>
        <div className="row-between">
          <div>
            <div className="h-section">Ask the codebase</div>
            <div className="sub">Searches {activeRepo} for the code closest to your question, then the AI writes an answer from it. One AI call per question.</div>
          </div>
          {aiLeft !== null && aiLeft !== undefined && <span className="badge badge-gray">{aiLeft} AI calls left today</span>}
        </div>

        <div className="row" style={{ marginTop: 12 }}>
          {suggestions.ask.map((s) => (
            <button key={s} type="button" className="smart-chip" onClick={() => send(s)} disabled={loading}>
              {s}
            </button>
          ))}
        </div>

        <textarea
          className="inp"
          rows={3}
          style={{ marginTop: 12, resize: 'vertical' }}
          placeholder="Ask a question about this repository…"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
        />
        <div className="row-between" style={{ marginTop: 10 }}>
          <label className="row sub" style={{ marginTop: 0 }}>
            Code snippets to use
            <select className="inp" style={{ width: 70, padding: '3px 6px' }} value={topK} onChange={(e) => setTopK(Number(e.target.value))}>
              {[3, 5, 8, 10].map((n) => <option key={n} value={n}>{n}</option>)}
            </select>
          </label>
          <button type="button" className="btn btn-primary" onClick={() => send()} disabled={loading || !question.trim()}>
            {loading ? <><span className="spinner" /> Searching…</> : 'Query Code'}
          </button>
        </div>
      </div>

      {error && <ErrorNotice error={error} onDismiss={() => setError(null)} />}

      {result && (
        <div className="card">
          <div className="row-between" style={{ marginBottom: 8 }}>
            <span className="badge badge-green">Answer</span>
            {result.model && <span className="mono muted" style={{ fontSize: 10.5 }}>{result.model}</span>}
          </div>
          <div className="sub mono" style={{ marginBottom: 8 }}>{result.question}</div>
          <FormattedMarkdown text={result.answer} />

          <div style={{ borderTop: '1px solid #f1f5f9', marginTop: 14, paddingTop: 12 }}>
            <div className="h-section" style={{ marginBottom: 8 }}>Sources ({result.sources.length})</div>
            {result.sources.length === 0 ? (
              <div className="sub">No code was found for this question.</div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {result.sources.map((s, i) => (
                  <SourceRow key={`${s.chunk_id || s.qualified_name}-${i}`} index={i} source={s} target={target} />
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

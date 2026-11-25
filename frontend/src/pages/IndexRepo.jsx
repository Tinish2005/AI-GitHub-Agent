import { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api, parseGitHubUrl } from '../api';
import { useAppStore } from '../store/appStore';
import ErrorNotice from '../components/ErrorNotice';

const JOB_KEY = 'agent_index_job';
const POLL_MS = 3000;

function progressPercent(job) {
  if (job.status === 'done') return 100;
  if (job.phase === 'fetching') return Math.round(30 * (job.files_total ? job.files_done / job.files_total : 0));
  const est = Math.max(1, job.estimated_seconds || 1);
  return Math.round(30 + 65 * Math.min(1, (job.elapsed_seconds || 0) / est));
}

export default function IndexRepo() {
  const { repos, publicIndexing, refreshStats, setActiveRepo, logActivity, markUnreachable } = useAppStore();
  const navigate = useNavigate();
  const [input, setInput] = useState('');
  const [folder, setFolder] = useState('');
  const [indexing, setIndexing] = useState(''); // folder of the job being followed, for the message
  const [submitting, setSubmitting] = useState(false);
  const [job, setJob] = useState(null);
  const [already, setAlready] = useState('');
  const [error, setError] = useState(null);
  const [lostConnection, setLostConnection] = useState(false);
  const timer = useRef(null);
  const alive = useRef(true);
  const inputRef = useRef(null);

  const poll = useCallback(
    async (jobId) => {
      if (!alive.current) return;
      try {
        const data = await api.indexStatus(jobId);
        if (!alive.current) return;
        setLostConnection(false);
        setJob(data);
        if (data.status === 'queued' || data.status === 'running') {
          timer.current = setTimeout(() => poll(jobId), POLL_MS);
          return;
        }
        sessionStorage.removeItem(JOB_KEY);
        if (data.status === 'done') {
          await refreshStats();
          if (data.repo) setActiveRepo(data.repo);
          logActivity('Index', `Indexed ${data.repo} (${data.chunks_indexed} chunks)`);
        }
      } catch (err) {
        if (!alive.current) return;
        if (err.status === 404) {
          sessionStorage.removeItem(JOB_KEY);
          setJob(null);
          setError('That indexing job is no longer known to the server (it may have restarted). Start it again.');
          return;
        }
        setLostConnection(true); // keep trying; the server may be restarting
        timer.current = setTimeout(() => poll(jobId), POLL_MS);
      }
    },
    [refreshStats, setActiveRepo, logActivity],
  );

  useEffect(() => {
    alive.current = true;
    const saved = sessionStorage.getItem(JOB_KEY);
    if (saved) poll(saved);
    return () => {
      alive.current = false;
      if (timer.current) clearTimeout(timer.current);
    };
  }, [poll]);

  const start = async () => {
    setError(null);
    setAlready('');
    let parsed;
    try {
      parsed = parseGitHubUrl(input);
    } catch (e) {
      setError(e.message);
      return;
    }
    setSubmitting(true);
    setJob(null);
    try {
      const folderValue = folder.trim() || parsed.path || '';
      setIndexing(folderValue);
      const data = await api.indexPublic(parsed.owner, parsed.repo, folderValue);
      if (data.status === 'already_indexed' || !data.job_id) {
        setAlready(`${parsed.owner}/${parsed.repo}`);
        await refreshStats();
      } else {
        sessionStorage.setItem(JOB_KEY, data.job_id);
        setJob(data);
        timer.current = setTimeout(() => poll(data.job_id), POLL_MS);
      }
    } catch (err) {
      setError(err);
      if (err.network) markUnreachable();
    } finally {
      setSubmitting(false);
    }
  };

  const running = job && (job.status === 'queued' || job.status === 'running');
  const open = !publicIndexing || publicIndexing.enabled;
  const pi = publicIndexing;

  // Stop following the old job and unlock the form. The server keeps working on a running job.
  const clearJob = () => {
    if (timer.current) clearTimeout(timer.current);
    sessionStorage.removeItem(JOB_KEY);
    setJob(null);
    setAlready('');
    setError(null);
    setLostConnection(false);
    setTimeout(() => inputRef.current && inputRef.current.focus(), 0);
  };

  // Why the form is locked, if it is. Shown on the page so it is never a mystery.
  const blockReason = !open
    ? 'Visitor indexing is closed on this server, so only the repositories that are already indexed can be used.'
    : running
      ? `Indexing ${job.repo} is still in progress. You can start another repository when it finishes.`
      : '';

  return (
    <div className="pg">
      <div className="card" style={{ borderLeft: '3px solid #6366f1' }}>
        <div className="h-section">Index a repository</div>
        <div className="sub">
          Fetches the repository's Python files from GitHub, splits them into chunks and builds the search index. Public repositories only. The demo repository is already indexed, so you can start with it.
        </div>

        {pi && open && (
          <p className="sub" style={{ marginTop: 8 }}>
            {`${pi.message} Up to ${pi.max_files} files and ${pi.max_chunks} chunks per repository; ${pi.jobs_per_ip_per_hour} repositor${pi.jobs_per_ip_per_hour === 1 ? 'y' : 'ies'} per hour and ${pi.jobs_per_ip_per_day} per day for each visitor.`}
          </p>
        )}

        <div className="lbl" style={{ marginTop: 12 }}>Repository (owner/repo or link)</div>
        {blockReason && (
          <div className="notice notice-amber" style={{ marginBottom: 10 }}>
            <div style={{ flex: 1 }}>
              <div className="notice-text">{blockReason}</div>
            </div>
            {running && (
              <button type="button" className="notice-x" onClick={clearJob}>Stop watching</button>
            )}
          </div>
        )}
        <input
          ref={inputRef}
          className="inp"
          value={input}
          disabled={!open || submitting || !!running}
          placeholder="owner/repo or https://github.com/owner/repo"
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && start()}
        />
        <div className="lbl" style={{ marginTop: 10 }}>Folder (optional)</div>
        <input
          className="inp"
          value={folder}
          disabled={!open || submitting || !!running}
          placeholder="for example src/mypackage"
          onChange={(e) => setFolder(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && start()}
        />
        <p className="sub" style={{ marginTop: 4 }}>
          Large repositories are cut to the first files. If you name the folder that holds the core code, those files are indexed instead and the dashboard shows more import links. A link such as github.com/owner/repo/tree/main/src/pkg works too.
        </p>
        <div className="row" style={{ marginTop: 10, justifyContent: 'flex-end' }}>
          <button type="button" className="btn btn-primary" onClick={start} disabled={!open || submitting || !!running || !input.trim()}>
            {submitting ? <><span className="spinner" /> Starting…</> : 'Start Indexing'}
          </button>
        </div>
      </div>

      {error && <ErrorNotice error={error} onDismiss={() => setError(null)} />}

      {already && (
        <div className="notice notice-green">
          <div style={{ flex: 1 }}>
            <div className="notice-title">Already indexed: {already}</div>
            <div className="notice-text">It is ready to use.</div>
            <div className="row" style={{ marginTop: 8 }}>
              <button type="button" className="btn btn-sec" onClick={() => { setActiveRepo(already); navigate('/'); }}>View dashboard</button>
              <button type="button" className="btn btn-primary" onClick={() => { setActiveRepo(already); navigate('/ask'); }}>Ask a question</button>
              <button type="button" className="btn btn-sec" onClick={clearJob}>Index another repository</button>
            </div>
          </div>
        </div>
      )}

      {job && running && (
        <div className="card">
          <div style={{ fontSize: 14, fontWeight: 700 }}>Indexing {job.repo}{indexing ? ` (folder ${indexing})` : ''}</div>
          <p className="sub">{job.message} It keeps running on the server if you leave this page.</p>
          <div className="progress" style={{ margin: '12px 0 8px' }}><i style={{ width: `${progressPercent(job)}%` }} /></div>
          <div className="sub mono">
            {job.phase === 'fetching'
              ? `Fetching files from GitHub${job.files_total ? `: ${job.files_done} of ${job.files_total}` : '…'}`
              : `Embedding ${job.chunks_total} chunks · about ${Math.max(1, Math.ceil((job.estimated_seconds || 60) / 60))} min · ${Math.round(job.elapsed_seconds || 0)} s so far`}
          </div>
          {lostConnection && <p className="sub" style={{ color: '#b45309' }}>Lost the connection to the server. Trying again…</p>}
        </div>
      )}

      {job && job.status === 'failed' && (
        <div className="notice notice-red">
          <div>
            <div className="notice-title">Indexing failed</div>
            <div className="notice-text">{job.error || job.message}</div>
          </div>
          <button type="button" className="notice-x" onClick={clearJob}>Index another repository</button>
        </div>
      )}

      {job && job.status === 'done' && (
        <div className="notice notice-green">
          <div style={{ flex: 1 }}>
            <div className="notice-title">Indexed {job.repo}{indexing ? ` (folder ${indexing})` : ''}</div>
            <div className="notice-text">
              {job.files_indexed} files and {job.chunks_indexed} chunks are in the search index.
              {job.truncated ? ` Only part of the repository was indexed, because of the per-repository limit${pi ? ` of ${pi.max_files} files and ${pi.max_chunks} chunks` : ''}.` : ''}
            </div>
            <div className="row" style={{ marginTop: 8 }}>
              <button type="button" className="btn btn-sec" onClick={() => navigate('/')}>View dashboard</button>
              <button type="button" className="btn btn-primary" onClick={() => navigate('/ask')}>Ask a question</button>
              <button type="button" className="btn btn-sec" onClick={clearJob}>Index another repository</button>
            </div>
          </div>
        </div>
      )}

      <div className="card">
        <div className="h-section">Indexed repositories</div>
        <div className="sub">Visitor repositories can be removed to make room, or when the server restarts. The demo always stays.</div>
        <table className="tbl" style={{ marginTop: 8 }}>
          <thead>
            <tr><th>Repository</th><th>Files</th><th>Chunks</th><th /></tr>
          </thead>
          <tbody>
            {repos.map((r) => (
              <tr key={r.repo}>
                <td className="mono">{r.repo} {r.is_demo && <span className="badge badge-purple">Demo</span>}</td>
                <td>{r.files}</td>
                <td>{r.chunks.toLocaleString()}</td>
                <td style={{ textAlign: 'right' }}>
                  <button type="button" className="link-btn" onClick={() => { setActiveRepo(r.repo); navigate('/'); }}>Use this repository</button>
                </td>
              </tr>
            ))}
            {!repos.length && <tr><td colSpan={4} className="muted">Nothing is indexed yet.</td></tr>}
          </tbody>
        </table>
      </div>
    </div>
  );
}

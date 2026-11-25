import { useNavigate } from 'react-router-dom';
import { useAppStore } from '../store/appStore';

const fmtDate = (iso) => {
  if (!iso) return '-';
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? '-' : d.toLocaleString([], { month: 'short', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit' });
};

export default function CodeMemory() {
  const { repos, stats, setActiveRepo, activeRepo } = useAppStore();
  const navigate = useNavigate();
  const emb = stats && stats.embedding_model ? stats : null;

  return (
    <div className="pg">
      <div className="card">
        <div className="h-section">Code memory</div>
        <div className="sub">
          The repositories in the search index right now. Visitor repositories can be removed to make room for others, or when the server restarts. The demo always stays.
        </div>
        {emb && (
          <p className="sub mono" style={{ marginTop: 8 }}>
            Embedding model: {stats.embedding_model}{stats.embedding_dimension ? ` · ${stats.embedding_dimension} numbers per chunk` : ''}
          </p>
        )}
        <table className="tbl" style={{ marginTop: 10 }}>
          <thead>
            <tr><th>Repository</th><th>Commit</th><th>Files</th><th>Chunks</th><th>Types</th><th>Last synced</th><th /></tr>
          </thead>
          <tbody>
            {repos.map((r) => (
              <tr key={r.repo}>
                <td className="mono">
                  {r.repo} {r.is_demo && <span className="badge badge-purple">Demo</span>} {r.repo === activeRepo && <span className="badge badge-green">Selected</span>}
                </td>
                <td className="mono">{r.commit ? r.commit.slice(0, 7) : r.ref || '-'}</td>
                <td>{r.files}</td>
                <td>{r.chunks.toLocaleString()}</td>
                <td className="muted" style={{ fontSize: 11 }}>
                  {Object.entries(r.chunks_by_type || {}).map(([k, v]) => `${k} ${v}`).join(' · ') || '-'}
                </td>
                <td className="muted">{fmtDate(r.last_synced)}</td>
                <td style={{ textAlign: 'right' }}>
                  <button type="button" className="link-btn" onClick={() => { setActiveRepo(r.repo); navigate('/'); }}>Open</button>
                </td>
              </tr>
            ))}
            {!repos.length && <tr><td colSpan={7} className="muted">Nothing is indexed yet.</td></tr>}
          </tbody>
        </table>
      </div>
    </div>
  );
}

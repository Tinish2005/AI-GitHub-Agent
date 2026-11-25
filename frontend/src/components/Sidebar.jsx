import { NavLink } from 'react-router-dom';
import { useAppStore } from '../store/appStore';

export default function Sidebar() {
  const { repos, activeRepo, setActiveRepo, totalChunks, server, aiLeft, aiPerDay } = useAppStore();

  const linkClass = ({ isActive }) => (isActive ? 'nav-item active' : 'nav-item');
  const ready = server === 'ready';
  const status = ready
    ? { text: 'Online', color: '#10b981' }
    : server === 'down'
      ? { text: 'Not responding', color: '#ef4444' }
      : { text: 'Waking up', color: '#f59e0b' };

  return (
    <div className="sidebar">
      <NavLink to="/" className="brand">
        <div className="brand-icon">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 2L2 7l10 5 10-5-10-5z" />
            <path d="M2 17l10 5 10-5" />
            <path d="M2 12l10 5 10-5" />
          </svg>
        </div>
        <div>
          <div className="brand-name">AI GitHub Agent</div>
        </div>
      </NavLink>

      <div className="sidebar-target-wrap">
        <label className="sidebar-target-lbl" htmlFor="sidebar-repo">Target Repository</label>
        <select
          id="sidebar-repo"
          className="sidebar-target-select"
          value={activeRepo}
          disabled={!repos.length}
          onChange={(e) => setActiveRepo(e.target.value)}
        >
          {!repos.length && <option value="">{ready ? 'No repository indexed' : 'Loading…'}</option>}
          {repos.map((r) => (
            <option key={r.repo} value={r.repo}>
              {r.repo}
            </option>
          ))}
        </select>
      </div>

      <div className="nav-section">Workspace</div>
      <NavLink to="/" end className={linkClass}>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" /><polyline points="9 22 9 12 15 12 15 22" /></svg>
        Dashboard
      </NavLink>
      <NavLink to="/index-repo" className={linkClass}>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="7 10 12 15 17 10" /><line x1="12" y1="15" x2="12" y2="3" /></svg>
        Index a Repo
        <span className="nav-badge">{repos.length}</span>
      </NavLink>
      <NavLink to="/ask" className={linkClass}>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z" /></svg>
        Ask Codebase
      </NavLink>

      <div className="nav-section">Workflow</div>
      <NavLink to="/plan" className={linkClass}>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><polyline points="14 2 14 8 20 8" /><line x1="16" y1="13" x2="8" y2="13" /><line x1="16" y1="17" x2="8" y2="17" /></svg>
        Plan
      </NavLink>
      <NavLink to="/fix" className={linkClass}>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M11 4H4a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" /><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" /></svg>
        Fix a Bug
      </NavLink>
      <NavLink to="/execute" className={linkClass}>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="5 3 19 12 5 21 5 3" /></svg>
        Execute
      </NavLink>

      <div className="nav-section">Insights</div>
      <NavLink to="/code-memory" className={linkClass}>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><ellipse cx="12" cy="5" rx="9" ry="3" /><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" /><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" /></svg>
        Code Memory
      </NavLink>
      <NavLink to="/activity" className={linkClass}>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12" /></svg>
        Activity
      </NavLink>

      <div className="sidebar-footer">
        <div className="sf-row">
          <span className="sf-label">Repositories</span>
          <span className="sf-val">{repos.length}</span>
        </div>
        <div className="sf-row">
          <span className="sf-label">Code Chunks</span>
          <span className="sf-val">{totalChunks.toLocaleString()}</span>
        </div>
        <div className="sf-row">
          <span className="sf-label">AI calls left today</span>
          <span className="sf-val" id="side-ai">{aiLeft === null || aiLeft === undefined ? '-' : `${aiLeft} of ${aiPerDay}`}</span>
        </div>
        <div className="sf-row" style={{ marginBottom: 0 }}>
          <span className="sf-label">Backend</span>
          <span style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: '11px', fontWeight: 600, color: status.color }}>
            <span style={{ width: 6, height: 6, borderRadius: '50%', background: status.color }}></span>
            {status.text}
          </span>
        </div>
      </div>
    </div>
  );
}

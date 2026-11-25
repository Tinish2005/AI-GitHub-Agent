import { useMemo, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAppStore } from '../store/appStore';

const baseName = (p) => String(p).replace(/\\/g, '/').split('/').pop();
const fmtDate = (iso) => {
  if (!iso) return '-';
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? '-' : d.toLocaleDateString([], { month: 'short', day: 'numeric', year: 'numeric' });
};

// Arc diagram of the import links between the largest files. All numbers come from GET /stats.
function DependencyMap({ files, edges, selected, onSelect }) {
  const W = 1000;
  const baseY = 200;
  const maxChunks = Math.max(...files.map((f) => f.chunks), 1);
  const pad = 70;
  const xs = files.map((_, i) => (files.length === 1 ? W / 2 : pad + (i * (W - 2 * pad)) / (files.length - 1)));
  const pos = Object.fromEntries(files.map((f, i) => [f.file_path, { x: xs[i], r: 7 + 15 * Math.sqrt(f.chunks / maxChunks) }]));

  const shown = edges.filter((e) => pos[e.source] && pos[e.target]);
  return (
    <svg viewBox={`0 0 ${W} 300`} style={{ width: '100%', height: 'auto', display: 'block' }} role="img" aria-label="Import links between files">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="context-stroke" />
        </marker>
      </defs>
      {shown.map((e, i) => {
        const a = pos[e.source].x;
        const b = pos[e.target].x;
        const rx = Math.abs(b - a) / 2;
        const ry = Math.min(150, rx * 0.7);
        const out = e.source === selected;
        const inc = e.target === selected;
        const color = out ? '#6366f1' : inc ? '#10b981' : '#cbd5e1';
        const sweep = a < b ? 1 : 0;
        return (
          <path
            key={i}
            d={`M ${a},${baseY} A ${rx},${ry} 0 0 ${sweep} ${b},${baseY}`}
            fill="none"
            stroke={color}
            strokeWidth={out || inc ? 1.8 : 1}
            opacity={out || inc ? 0.95 : 0.55}
            markerEnd="url(#arrow)"
          />
        );
      })}
      {files.map((f) => {
        const { x, r } = pos[f.file_path];
        const sel = f.file_path === selected;
        const label = baseName(f.file_path);
        return (
          <g key={f.file_path} style={{ cursor: 'pointer' }} onClick={() => onSelect(f.file_path)}>
            <title>{`${f.file_path} (${f.chunks} chunks)`}</title>
            <circle cx={x} cy={baseY} r={r} fill={sel ? '#4f46e5' : '#a5b4fc'} stroke={sel ? '#312e81' : '#6366f1'} strokeWidth={sel ? 2 : 1} />
            <text x={x} y={baseY + r + 16} textAnchor="middle" fontSize="11" fontFamily="ui-monospace, Menlo, Consolas, monospace" fill={sel ? '#0f172a' : '#475569'} fontWeight={sel ? 700 : 400}>
              {label.length > 16 ? `${label.slice(0, 15)}…` : label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export default function Home() {
  const { currentRepo, repos, server, stats, statsError, refreshStats, publicIndexing } = useAppStore();
  const navigate = useNavigate();
  const [picked, setPicked] = useState('');

  const files = useMemo(() => (currentRepo ? currentRepo.chunks_per_file || [] : []), [currentRepo]);
  const edges = useMemo(() => (currentRepo ? currentRepo.import_edges || [] : []), [currentRepo]);
  const external = currentRepo ? currentRepo.top_external_imports || [] : [];

  const sorted = useMemo(() => [...files].sort((a, b) => b.chunks - a.chunks), [files]);
  const mapFiles = sorted.slice(0, 8);
  const barFiles = sorted.slice(0, 6);
  const total = currentRepo ? currentRepo.chunks : 0;
  const restChunks = total - barFiles.reduce((s, f) => s + f.chunks, 0);
  const restFiles = currentRepo ? currentRepo.files - barFiles.length : 0;

  const selected = files.some((f) => f.file_path === picked) ? picked : mapFiles[0] ? mapFiles[0].file_path : '';
  const selFile = files.find((f) => f.file_path === selected);
  const imports = edges.filter((e) => e.source === selected).map((e) => e.target);
  const importedBy = edges.filter((e) => e.target === selected).map((e) => e.source);
  const list = (arr) => (arr.length ? arr.map(baseName).join(', ') : 'No other indexed file');

  if (!stats && statsError) {
    return (
      <div className="pg">
        <div className="notice notice-red">
          <div style={{ flex: 1 }}>
            <div className="notice-title">Could not load the repository data</div>
            <div className="notice-text">{statsError}</div>
          </div>
          <button type="button" className="notice-x" onClick={refreshStats}>Try again</button>
        </div>
      </div>
    );
  }

  if (!stats) {
    return <div className="pg"><div className="card muted" style={{ fontSize: 12.5 }}>Loading repository data…</div></div>;
  }

  if (!currentRepo) {
    return (
      <div className="pg">
        <div className="card" style={{ textAlign: 'center', padding: '40px 20px' }}>
          <h2 style={{ fontSize: 15, fontWeight: 700, marginBottom: 6 }}>No repository is indexed yet</h2>
          <p className="muted" style={{ fontSize: 12.5, marginBottom: 14 }}>Index a public GitHub repository to see its statistics and ask questions about it.</p>
          <Link to="/index-repo" className="btn btn-primary" style={{ textDecoration: 'none' }}>Index a repository</Link>
        </div>
      </div>
    );
  }

  const askAbout = (path) => {
    navigate('/ask', { state: { prefill: `Explain the role of ${path.replace(/\\/g, '/')} and how it is used` } });
  };

  return (
    <div className="pg">
      <div className="card">
        <div className="row-between">
          <div>
            <div className="row">
              <span className="mono" style={{ fontSize: 15, fontWeight: 700 }}>{currentRepo.repo}</span>
              {currentRepo.ref && <span className="badge badge-gray">{currentRepo.ref}</span>}
              {currentRepo.is_demo && <span className="badge badge-purple">Demo repository</span>}
            </div>
            <div className="sub">
              {currentRepo.is_demo
                ? 'Indexed ahead of time, so you can start asking questions right away.'
                : 'Indexed on this server. Visitor repositories can be removed to make room for others.'}
            </div>
          </div>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10, marginTop: 14 }}>
          <div className="stat-pill"><div className="stat-num">{currentRepo.files}</div><div className="stat-lbl">Files</div></div>
          <div className="stat-pill"><div className="stat-num">{currentRepo.chunks.toLocaleString()}</div><div className="stat-lbl">Code chunks</div></div>
          <div className="stat-pill"><div className="stat-num">{fmtDate(currentRepo.last_synced)}</div><div className="stat-lbl">Last synced</div></div>
          <div className="stat-pill"><div className="stat-num mono">{currentRepo.commit ? currentRepo.commit.slice(0, 7) : '-'}</div><div className="stat-lbl">Commit</div></div>
        </div>
      </div>

      <div className="card">
        <div className="row-between">
          <div>
            <div className="h-section">Import links</div>
            <div className="sub">Arrows point from the importing file to the file it imports. Circle size follows chunk count. The 8 largest files are shown.</div>
          </div>
          <span className="mono muted" style={{ fontSize: 11 }}>{currentRepo.import_edge_count ?? edges.length} import links in total</span>
        </div>
        {mapFiles.length ? (
          <>
            <DependencyMap files={mapFiles} edges={edges} selected={selected} onSelect={setPicked} />
            {!currentRepo.is_demo && (currentRepo.files >= ((publicIndexing && publicIndexing.max_files) || 10) || edges.length === 0) && (
              <div className="notice notice-amber" style={{ marginTop: 8 }}>
                <div className="notice-text">
                  {edges.length === 0
                    ? 'No import links were found between the indexed files. '
                    : 'Only the first files of this repository are indexed, so many imports point to files that are not in the index and show no arrows. '}
                  To see more links, index a single folder that holds the core code (the Folder box on the Index a Repo page).
                </div>
              </div>
            )}
            {external.length > 0 && (
              <p className="sub">Most-used outside packages: {external.map((e) => `${e.name} (${e.files} files)`).join(', ')}</p>
            )}
            {selFile && (
              <div className="step-card" style={{ marginTop: 10 }}>
                <div className="row-between" style={{ alignItems: 'flex-start' }}>
                  <div style={{ minWidth: 0 }}>
                    <div className="mono" style={{ fontSize: 12, fontWeight: 700, marginBottom: 6 }}>{selFile.file_path.replace(/\\/g, '/')}</div>
                    <div className="kv">
                      <span>Chunks</span><span className="mono">{selFile.chunks}</span>
                      <span>Imports</span><span className="mono">{list(imports)}</span>
                      <span>Imported by</span><span className="mono">{list(importedBy)}</span>
                    </div>
                  </div>
                  <button type="button" className="link-btn" onClick={() => askAbout(selFile.file_path)}>Ask about this file →</button>
                </div>
              </div>
            )}
          </>
        ) : (
          <p className="sub" style={{ marginTop: 10 }}>No file data for this repository.</p>
        )}
      </div>

      <div className="card">
        <div className="h-section">Chunks per file</div>
        <div className="sub">Share of indexed chunks in the largest files; the rest are grouped.</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 9, marginTop: 12 }}>
          {barFiles.map((f) => (
            <div key={f.file_path}>
              <div className="row-between" style={{ fontSize: 11.5, marginBottom: 3 }}>
                <span className="mono">{baseName(f.file_path)}</span>
                <span className="muted">{f.chunks} chunks · {Math.round((f.chunks / total) * 100)}%</span>
              </div>
              <div className="bar-track"><div className="bar-fill" style={{ width: `${(f.chunks / total) * 100}%` }} /></div>
            </div>
          ))}
          {restChunks > 0 && restFiles > 0 && (
            <div>
              <div className="row-between" style={{ fontSize: 11.5, marginBottom: 3 }}>
                <span className="mono">{restFiles} other file{restFiles === 1 ? '' : 's'}</span>
                <span className="muted">{restChunks} chunks · {Math.round((restChunks / total) * 100)}%</span>
              </div>
              <div className="bar-track"><div className="bar-fill" style={{ width: `${(restChunks / total) * 100}%`, background: '#cbd5e1' }} /></div>
            </div>
          )}
        </div>
        <div className="sub" style={{ marginTop: 10 }}>{total.toLocaleString()} chunks in total · {repos.length} repositor{repos.length === 1 ? 'y' : 'ies'} indexed</div>
      </div>

      {server === 'ready' && publicIndexing && publicIndexing.enabled && (
        <div className="row-between">
          <span className="sub">Want to try another project?</span>
          <Link to="/index-repo" className="link-btn" style={{ textDecoration: 'none' }}>Index a repository →</Link>
        </div>
      )}
    </div>
  );
}

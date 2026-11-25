import { useAppStore } from '../store/appStore';

export default function Activity() {
  const { activityLog, clearActivity } = useAppStore();
  return (
    <div className="pg">
      <div className="card">
        <div className="row-between">
          <div>
            <div className="h-section">Activity</div>
            <div className="sub">What you did in this browser session. It is not saved on the server.</div>
          </div>
          <button type="button" className="btn btn-sec" onClick={clearActivity} disabled={!activityLog.length}>Clear history</button>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 12 }}>
          {activityLog.map((a) => (
            <div key={a.id} className="step-card row-between">
              <div className="row" style={{ flexWrap: 'nowrap' }}>
                <span className="badge badge-gray">{a.type}</span>
                <span style={{ fontSize: 12.5 }}>{a.desc}</span>
              </div>
              <span className="muted" style={{ fontSize: 11 }}>{a.time}</span>
            </div>
          ))}
          {!activityLog.length && <div className="sub">No activity yet. Ask a question, make a plan or index a repository.</div>}
        </div>
      </div>
    </div>
  );
}

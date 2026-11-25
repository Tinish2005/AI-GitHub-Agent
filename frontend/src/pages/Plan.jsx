import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api';
import { useAppStore } from '../store/appStore';
import ErrorNotice from '../components/ErrorNotice';

export default function Plan() {
  const { suggestions, activeRepo, logActivity, markUnreachable } = useAppStore();
  const navigate = useNavigate();
  const [goal, setGoal] = useState('');
  const [loading, setLoading] = useState(false);
  const [plan, setPlan] = useState(null);
  const [error, setError] = useState(null);

  const run = async (text) => {
    const g = (text ?? goal).trim();
    if (!g || loading) return;
    setGoal(g);
    setLoading(true);
    setError(null);
    try {
      const data = await api.plan(g);
      setPlan(data);
      logActivity('Plan', `Planned: "${g.slice(0, 60)}" (${(data.steps || []).length} steps)`);
    } catch (err) {
      setError(err);
      if (err.network) markUnreachable();
    } finally {
      setLoading(false);
    }
  };

  const steps = plan && Array.isArray(plan.steps) ? plan.steps : [];

  return (
    <div className="pg">
      <div className="card" style={{ borderLeft: '3px solid #6366f1' }}>
        <div className="h-section">Plan a task</div>
        <div className="sub">Turns a goal into the steps the agent would run. Planning does not use AI calls and does not touch your code.</div>

        <div className="row" style={{ marginTop: 12 }}>
          {suggestions.plan.map((s) => (
            <button key={s} type="button" className="smart-chip" onClick={() => run(s)} disabled={loading}>{s}</button>
          ))}
        </div>
        <textarea
          className="inp"
          rows={3}
          style={{ marginTop: 12, resize: 'vertical' }}
          placeholder="Describe what you want to do, for example: fix the bug in issue 42"
          value={goal}
          onChange={(e) => setGoal(e.target.value)}
        />
        <div className="row" style={{ marginTop: 10, justifyContent: 'flex-end' }}>
          <button type="button" className="btn btn-primary" onClick={() => run()} disabled={loading || !goal.trim()}>
            {loading ? <><span className="spinner" /> Planning…</> : 'Generate Plan'}
          </button>
        </div>
      </div>

      {error && <ErrorNotice error={error} onDismiss={() => setError(null)} />}

      {plan && (
        <div className="card">
          <div className="row-between" style={{ marginBottom: 10 }}>
            <div>
              <span className="h-section">Plan</span>
              <span className="mono muted" style={{ fontSize: 11, marginLeft: 8 }}>{steps.length} step{steps.length === 1 ? '' : 's'}</span>
            </div>
            {plan.strategy && <span className="mono muted" style={{ fontSize: 10.5 }}>strategy: {plan.strategy}</span>}
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {steps.map((s, i) => (
              <div key={s.id ?? i} className="step-card row" style={{ alignItems: 'flex-start', flexWrap: 'nowrap' }}>
                <span className="badge badge-gray mono">{String(s.id ?? i + 1).padStart(2, '0')}</span>
                <div>
                  <div className="mono" style={{ fontSize: 12, fontWeight: 700 }}>{s.kind}</div>
                  <div className="sub">{s.description}</div>
                </div>
              </div>
            ))}
          </div>
          <div className="row-between" style={{ marginTop: 12 }}>
            <span className="sub">Opening a pull request is not part of a plan.</span>
            <button
              type="button"
              className="btn btn-sec"
              onClick={() => navigate('/execute', { state: { prefill: goal } })}
              title={`Run this goal on ${activeRepo || 'the selected repository'}`}
            >
              Run this goal in Execute →
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

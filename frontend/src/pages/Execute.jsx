import { useEffect, useState } from 'react';
import { useLocation } from 'react-router-dom';
import { api } from '../api';
import { useAppStore } from '../store/appStore';
import ErrorNotice from '../components/ErrorNotice';
import StepOutput from '../components/StepOutput';
import { stepOutcome, analysisFoundNothing } from '../components/stepOutcome';

export default function Execute() {
  const { target, activeRepo, suggestions, logActivity, refreshStats, markUnreachable, aiLeft } = useAppStore();
  const location = useLocation();
  const [goal, setGoal] = useState('');
  const [abortOnFailure, setAbortOnFailure] = useState(true);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [seconds, setSeconds] = useState(0);
  const [error, setError] = useState(null);

  useEffect(() => {
    const prefill = location.state && location.state.prefill;
    if (prefill) setGoal(prefill);
  }, [location.state]);

  const run = async (text) => {
    const g = (text ?? goal).trim();
    if (!g || !target || loading) return;
    setGoal(g);
    setLoading(true);
    setError(null);
    setResult(null);
    const started = performance.now();
    try {
      const data = await api.execute({ goal: g, owner: target.owner, repo: target.repo, ref: target.ref, abortOnFailure });
      setSeconds((performance.now() - started) / 1000);
      setResult(data);
      logActivity('Execute', `Executed on ${activeRepo}: "${g.slice(0, 50)}"`);
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

  const steps = result && Array.isArray(result.steps) ? result.steps : [];
  // A step that ran but whose own check said no (for example a validation). A step that failed to run is not counted here.
  const checkFailed = steps.filter((s) => s.status === 'completed' && stepOutcome(s).tone === 'bad').length;
  // The app's own AI limit stopped the run: the failed step's text names it.
  const limitStep = steps.find((s) => s.status === 'failed' && /LimitError|AI limit|limit was reached/i.test(`${s.output || ''} ${s.error || ''}`));
  const limitHit = !!limitStep;
  const hourly = limitHit && /this hour/i.test(`${limitStep.output || ''} ${limitStep.error || ''}`);
  // A patch was written even though the analysis found nothing: the patch is a guess.
  const guessed = steps.some(analysisFoundNothing) && steps.some((s) => s.kind === 'generate' && /is_valid=True/.test(s.output || ''));

  return (
    <div className="pg">
      <div className="card" style={{ borderLeft: '3px solid #6366f1' }}>
        <div className="row-between">
          <div>
            <div className="h-section">Execute a workflow</div>
            <div className="sub">Runs the plan for your goal on the server, step by step, and shows each step's result when it finishes. Steps that write or fix use AI calls. Nothing in the repository is changed: a patch is only a proposal.</div>
          </div>
          {aiLeft !== null && aiLeft !== undefined && <span className="badge badge-gray">{aiLeft} AI calls left today</span>}
        </div>
        <div className="row" style={{ marginTop: 12 }}>
          {suggestions.plan.map((s) => (
            <button key={s} type="button" className="smart-chip" onClick={() => run(s)} disabled={loading}>{s}</button>
          ))}
        </div>
        <textarea
          className="inp"
          rows={3}
          style={{ marginTop: 12, resize: 'vertical' }}
          placeholder={`What should the agent do in ${activeRepo}?`}
          value={goal}
          onChange={(e) => setGoal(e.target.value)}
        />
        <div className="row-between" style={{ marginTop: 10 }}>
          <label className="row sub" style={{ marginTop: 0 }}>
            <input type="checkbox" checked={abortOnFailure} onChange={(e) => setAbortOnFailure(e.target.checked)} />
            Stop at the first failed step
          </label>
          <button type="button" className="btn btn-primary" onClick={() => run()} disabled={loading || !goal.trim()}>
            {loading ? <><span className="spinner" /> Running…</> : 'Run Execution'}
          </button>
        </div>
      </div>

      {error && <ErrorNotice error={error} onDismiss={() => setError(null)} />}

      {loading && <div className="card sub">Running the plan on the server. Results appear when it finishes. This can take up to a minute.</div>}

      {result && (
        <div className="card">
          <div className="row-between" style={{ marginBottom: 10 }}>
            <span className="h-section">Execution results</span>
            <span className="mono muted" style={{ fontSize: 10.5 }}>
              {result.completed ?? 0} completed · {result.failed ?? 0} failed · {result.skipped ?? 0} skipped · {seconds.toFixed(1)} s
            </span>
          </div>
          {limitHit && (
            <div className="notice notice-amber" style={{ marginBottom: 10 }}>
              <div className="notice-text">
                {hourly
                  ? 'You have used your AI calls for this hour (each visitor gets only a few per hour), so this run stopped early and the remaining steps were skipped. Try again in a while. The daily total in the sidebar is a separate, shared limit. Nothing is wrong with the repository or the server.'
                  : 'The AI limit was reached, so this run stopped early and the remaining steps were skipped. Nothing is wrong with the repository or the server. Try again later.'}
              </div>
            </div>
          )}
          {guessed && (
            <div className="notice notice-amber" style={{ marginBottom: 10 }}>
              <div className="notice-text">
                The analysis step could not find a cause in the code, so the patch below is a guess. Describe the problem more specifically (the function or file, and what goes wrong) to get a better one.
              </div>
            </div>
          )}
          {checkFailed > 0 && (
            <div className="notice notice-amber" style={{ marginBottom: 10 }}>
              <div className="notice-text">
                {checkFailed} step{checkFailed === 1 ? '' : 's'} ran but did not pass its own check (for example, a validation that said no). The step counts above only say whether a step ran.
              </div>
            </div>
          )}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {steps.map((s, i) => <StepOutput key={s.step_id ?? i} step={s} />)}
          </div>
        </div>
      )}
    </div>
  );
}

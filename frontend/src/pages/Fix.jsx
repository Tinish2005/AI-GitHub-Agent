import { useState } from 'react';
import { api } from '../api';
import { useAppStore } from '../store/appStore';
import { FormattedMarkdown } from '../components/FormattedOutput';
import DiffView from '../components/DiffView';
import { hasRealChanges } from '../components/diffUtil';
import ErrorNotice from '../components/ErrorNotice';

export default function Fix() {
  const { target, activeRepo, suggestions, logActivity, refreshStats, markUnreachable, aiLeft } = useAppStore();
  const [goal, setGoal] = useState('');
  const [busy, setBusy] = useState(''); // '' | 'propose' | 'validate'
  const [proposal, setProposal] = useState(null); // exactly what /fix/propose returned
  const [validation, setValidation] = useState(null);
  const [error, setError] = useState(null);
  const [copied, setCopied] = useState(false);

  const fail = (err) => {
    setError(err);
    if (err.network) markUnreachable();
  };

  const propose = async (text) => {
    const g = (text ?? goal).trim();
    if (!g || !target || busy) return;
    setGoal(g);
    setBusy('propose');
    setError(null);
    setValidation(null);
    setProposal(null);
    try {
      const data = await api.fixPropose({ goal: g, owner: target.owner, repo: target.repo, ref: target.ref });
      setProposal({ ...data, goal: data.goal || g });
      logActivity('Fix', `Proposed a fix for ${activeRepo}: "${g.slice(0, 50)}"`);
    } catch (err) {
      fail(err);
    } finally {
      setBusy('');
      refreshStats();
    }
  };

  const validate = async () => {
    if (!proposal || !target || busy) return;
    setBusy('validate');
    setError(null);
    try {
      const data = await api.fixValidate({ goal: proposal.goal, owner: target.owner, repo: target.repo, ref: target.ref, proposal });
      setValidation(data);
      logActivity('Fix', data.passed ? `Validation passed for ${activeRepo}` : `Validation failed for ${activeRepo}`);
    } catch (err) {
      fail(err);
    } finally {
      setBusy('');
    }
  };

  const copyDiff = async () => {
    try {
      await navigator.clipboard.writeText(proposal.diff);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      setError('Could not copy. Select the diff and copy it by hand.');
    }
  };

  if (!target) {
    return <div className="pg"><div className="card muted" style={{ fontSize: 12.5 }}>Select or index a repository first.</div></div>;
  }

  const noChange = !!proposal && proposal.is_valid && !hasRealChanges(proposal.diff);
  const usable = !!proposal && proposal.is_valid && !noChange;
  const sameDiff = validation && proposal && validation.diff_sha256 === proposal.diff_sha256;

  return (
    <div className="pg">
      <div className="card" style={{ borderLeft: '3px solid #6366f1' }}>
        <div className="row-between">
          <div>
            <div className="h-section">Fix a bug</div>
            <div className="sub">The AI proposes a patch from the real code of {activeRepo}. Nothing is changed in the repository.</div>
          </div>
          {aiLeft !== null && aiLeft !== undefined && <span className="badge badge-gray">{aiLeft} AI calls left today</span>}
        </div>
        <div className="row" style={{ marginTop: 12 }}>
          {suggestions.fix.map((s) => (
            <button key={s} type="button" className="smart-chip" onClick={() => propose(s)} disabled={!!busy}>{s}</button>
          ))}
        </div>
        <textarea
          className="inp"
          rows={3}
          style={{ marginTop: 12, resize: 'vertical' }}
          placeholder="Describe the bug or the change you want…"
          value={goal}
          onChange={(e) => setGoal(e.target.value)}
        />
        <div className="row" style={{ marginTop: 10, justifyContent: 'flex-end' }}>
          <button type="button" className="btn btn-primary" onClick={() => propose()} disabled={!!busy || !goal.trim()}>
            {busy === 'propose' ? <><span className="spinner" /> Proposing…</> : 'Propose Fix'}
          </button>
        </div>
      </div>

      {error && <ErrorNotice error={error} onDismiss={() => setError(null)} />}

      {proposal && (
        <div className="card">
          <div className="row-between" style={{ marginBottom: 8 }}>
            <span className="badge badge-blue">Proposed fix</span>
            <span className="mono muted" style={{ fontSize: 10.5 }}>
              {proposal.model || ''}
              {usable && typeof proposal.confidence === 'number' && (
                <span title="The model's own guess. It is not a check."> · model confidence {Math.round(proposal.confidence * 100)}%</span>
              )}
            </span>
          </div>

          {!usable ? (
            <>
              <div className="notice notice-amber">
                <div>
                  <div className="notice-title">{noChange ? 'The AI did not propose any change' : 'The AI did not return a usable diff'}</div>
                  <div className="notice-text">
                    {noChange
                      ? 'Its explanation below says why. There is nothing to validate. Describe the problem more specifically (the function or file, and what goes wrong) to get a patch.'
                      : `${proposal.validation_error || 'The answer had no valid diff.'} This happens now and then. Click Propose Fix again, or describe the change more specifically.`}
                  </div>
                </div>
              </div>
              {proposal.explanation ? <div style={{ marginTop: 10 }}><FormattedMarkdown text={proposal.explanation} /></div> : null}
            </>
          ) : (
            <>
              {proposal.explanation && <FormattedMarkdown text={proposal.explanation} />}
              <div style={{ marginTop: 10 }}><DiffView diff={proposal.diff} /></div>
              <div className="row-between" style={{ marginTop: 10 }}>
                <span className="mono muted" style={{ fontSize: 10.5 }}>diff fingerprint {String(proposal.diff_sha256 || '').slice(0, 12)}</span>
                <div className="row">
                  <button type="button" className="btn btn-sec" onClick={copyDiff}>{copied ? 'Copied' : 'Copy diff'}</button>
                  <button type="button" className="btn btn-primary" onClick={validate} disabled={!!busy}>
                    {busy === 'validate' ? <><span className="spinner" /> Checking…</> : `Validate against ${target.repo}${target.ref ? ` @ ${target.ref.slice(0, 7)}` : ''}`}
                  </button>
                </div>
              </div>
            </>
          )}

          {validation && (
            <div style={{ marginTop: 14 }} className={`notice ${validation.passed ? 'notice-green' : 'notice-red'}`}>
              <div style={{ flex: 1 }}>
                <div className="notice-title">{validation.passed ? 'Passed' : 'Validation failed'}</div>
                {validation.error ? <div className="notice-text mono" style={{ fontSize: 11 }}>{validation.error}</div> : null}
                {validation.warning ? <div className="notice-text">{validation.warning}</div> : null}
                <div className="notice-text" style={{ marginTop: 6 }}>
                  {(validation.checks || []).map((c) => (
                    <div key={c.name} className="row" style={{ gap: 6 }}>
                      <span style={{ width: 14 }}>{c.skipped ? '–' : c.passed ? '✓' : '✗'}</span>
                      <span className="mono">{c.name}</span>
                      <span style={{ opacity: 0.85 }}>{c.skipped ? 'not run' : c.message}</span>
                    </div>
                  ))}
                </div>
                <div className="notice-text mono" style={{ marginTop: 8, fontSize: 10.5 }}>
                  {validation.base_description || 'Checked against the repository files'}
                  {validation.base_commit ? ` · commit ${validation.base_commit.slice(0, 7)}` : ''}
                  {sameDiff ? ' · same diff as proposed' : ' · the diff fingerprint does not match the proposal'}
                </div>
                <div className="notice-text" style={{ marginTop: 6, opacity: 0.85 }}>
                  Passed means the patch fits the real files, parses, and its imports resolve. It does not mean the fix is correct. Tests and lint are not run.
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

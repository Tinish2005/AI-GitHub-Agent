import DiffView from './DiffView';

import { stepOutcome } from './stepOutcome';

const DOT = { ok: 'dot-ok', bad: 'dot-bad', skip: 'dot-skip', warn: 'dot-warn' };
const BADGE = { ok: 'badge-green', bad: 'badge-amber', skip: 'badge-gray', warn: 'badge-amber' };

// Text output of a step. Fenced "diff" blocks are shown with colours; everything else as plain text.
function Body({ text }) {
  const parts = String(text || '').split(/```/);
  return (
    <div style={{ marginTop: 8 }}>
      {parts.map((part, i) => {
        if (i % 2 === 1) {
          const nl = part.indexOf('\n');
          const lang = nl === -1 ? '' : part.slice(0, nl).trim();
          const body = nl === -1 ? part : part.slice(nl + 1);
          return lang === 'diff' ? <DiffView key={i} diff={body.replace(/\n$/, '')} /> : <pre key={i} className="pre-box">{body}</pre>;
        }
        return part.trim() ? <pre key={i} className="pre-box">{part.trim()}</pre> : null;
      })}
    </div>
  );
}

export default function StepOutput({ step }) {
  const o = stepOutcome(step);
  return (
    <div className="step-card">
      <div className="row">
        <span className={`dot ${DOT[o.tone]}`} />
        <span style={{ fontSize: 12.5, fontWeight: 600 }}>
          Step {step.step_id ?? step.id}: <span className="mono">{step.kind}</span>
        </span>
        <span className={`badge ${BADGE[o.tone]}`}>{o.label}</span>
      </div>
      {step.error ? <pre className="pre-box" style={{ color: '#b91c1c', marginTop: 8 }}>{step.error}</pre> : null}
      {step.output ? <Body text={step.output} /> : null}
    </div>
  );
}

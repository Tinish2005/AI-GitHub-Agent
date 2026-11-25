// True when the analysis step said it could not find anything relevant in the retrieved code.
export function analysisFoundNothing(step) {
  if (step.kind !== 'analyze' || step.status !== 'completed') return false;
  return /does not contain|doesn't contain|do not contain|cannot provide|can't provide|no information|not enough information|unable to/i.test(step.output || '');
}

// How a step should look. The server marks a step "completed" when it ran, even if its validation said no,
// so the validate step's text is read to show the real outcome.
export function stepOutcome(step) {
  const out = step.output || '';
  if (step.status === 'skipped') return { tone: 'skip', label: 'skipped' };
  if (step.status === 'failed') return { tone: 'bad', label: 'failed' };
  if (step.kind === 'validate') {
    const m = out.match(/Validation \(passed=(True|False)/);
    if (m && m[1] === 'False') return { tone: 'bad', label: 'validation failed' };
    if (m && m[1] === 'True') return { tone: 'ok', label: 'validation passed' };
  }
  if (step.kind === 'generate' && /is_valid=False/.test(out)) return { tone: 'warn', label: 'no usable diff' };
  return { tone: 'ok', label: 'completed' };
}


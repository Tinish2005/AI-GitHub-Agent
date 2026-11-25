// True when a unified diff really adds or removes at least one line (file headers do not count).
export function hasRealChanges(diff) {
  return String(diff || '')
    .split('\n')
    .some((l) => (l.startsWith('+') && !l.startsWith('+++')) || (l.startsWith('-') && !l.startsWith('---')));
}

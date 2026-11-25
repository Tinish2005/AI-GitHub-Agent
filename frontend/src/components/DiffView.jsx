// Shows a unified diff with colours. The text itself is never changed.
export default function DiffView({ diff }) {
  const lines = String(diff || '').split('\n');
  return (
    <div className="diff-container">
      {lines.map((line, i) => {
        let cls = '';
        if (line.startsWith('+++') || line.startsWith('---') || line.startsWith('@@')) cls = 'diff-hunk';
        else if (line.startsWith('+')) cls = 'diff-add';
        else if (line.startsWith('-')) cls = 'diff-del';
        return (
          <div key={i} className={cls} style={{ whiteSpace: 'pre' }}>
            {line || ' '}
          </div>
        );
      })}
    </div>
  );
}

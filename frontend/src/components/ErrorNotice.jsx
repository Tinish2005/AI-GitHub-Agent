// Shows an error from the API in plain words. `error` can be an ApiError, an Error or a string.
export default function ErrorNotice({ error, onDismiss }) {
  if (!error) return null;
  const status = typeof error === 'object' ? error.status : 0;
  const network = typeof error === 'object' && error.network;
  const message = typeof error === 'string' ? error : error.message;
  const retryAfter = typeof error === 'object' ? error.retryAfter : null;

  let title = 'Something went wrong';
  let tone = 'notice-red';
  if (network) {
    title = 'Cannot reach the server';
    tone = 'notice-amber';
  } else if (status === 429) {
    title = 'Limit reached';
    tone = 'notice-amber';
  } else if (status === 409) {
    title = 'Not possible';
    tone = 'notice-amber';
  } else if (status === 503) {
    title = 'Service not available';
  } else if (status === 403) {
    title = 'Not allowed';
  }

  const wait =
    retryAfter && retryAfter > 0
      ? retryAfter >= 90
        ? ` Try again in about ${Math.ceil(retryAfter / 60)} minutes.`
        : ` Try again in about ${Math.ceil(retryAfter)} seconds.`
      : '';

  return (
    <div className={`notice ${tone}`} role="alert">
      <div style={{ flex: 1 }}>
        <div className="notice-title">{title}</div>
        <div className="notice-text">
          {message}
          {status === 429 ? wait : ''}
          {network ? ' The app is checking whether the server is awake.' : ''}
        </div>
      </div>
      {onDismiss && (
        <button type="button" className="notice-x" onClick={onDismiss}>
          Dismiss
        </button>
      )}
    </div>
  );
}

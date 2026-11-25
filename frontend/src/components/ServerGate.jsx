import { useAppStore } from '../store/appStore';

// Shown instead of the page while the server is asleep. The free Render server sleeps after about 15 minutes.
export default function ServerGate({ children }) {
  const { server, wakeSeconds, retryServer } = useAppStore();
  if (server === 'ready') return children;

  if (server === 'down') {
    return (
      <div className="gate">
        <div className="gate-card">
          <h2 className="gate-title">The server is not responding</h2>
          <p className="gate-text">
            It is taking longer than usual to start, or the server address is wrong. Please try again in a few minutes.
          </p>
          <button type="button" className="btn btn-primary" onClick={retryServer}>
            Try again
          </button>
        </div>
      </div>
    );
  }

  const waking = server === 'waking';
  return (
    <div className="gate">
      <div className="gate-card">
        <h2 className="gate-title">{waking ? 'Waking up the server' : 'Connecting to the server'}</h2>
        <p className="gate-text">
          {waking
            ? 'The free server sleeps when nobody uses it. Starting it again usually takes under a minute. This page continues on its own.'
            : 'Checking that the server is ready…'}
        </p>
        <div className="progress indet">
          <i />
        </div>
        {waking && <p className="gate-note">Waiting for {wakeSeconds} s</p>}
      </div>
    </div>
  );
}

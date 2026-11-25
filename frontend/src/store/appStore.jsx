import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { api } from '../api';

const AppStoreContext = createContext(null);

const WAKE_EVERY_MS = 3000; // how often to ask /health while the server is waking up
const WAKE_GIVE_UP_MS = 120000; // after this the screen switches to "not responding"

// Suggested prompts. Plain text only; nothing here pretends to be data from the server.
export function suggestionsFor(repoName) {
  const lower = (repoName || '').toLowerCase();
  const short = (repoName || '').split('/')[1] || repoName || 'this repository';
  if (lower === 'psf/requests') {
    return {
      ask: [
        'How does HTTPAdapter handle retries?',
        'Where is SSL verification handled in sessions?',
        'How are cookies extracted and saved?',
        'How are query string parameters encoded into the URL?',
      ],
      plan: [
        'Review authentication flow and session cookies',
        'Suggest how to simplify redirect handling',
        'Audit the retry mechanism and refactor it',
        'Fix an open issue about redirect handling',
      ],
      fix: [
        'In HTTPAdapter.send, raise ValueError when timeout is zero or negative',
        'Handle a missing scheme in URL parsing without raising UnboundLocalError',
        'Raise a clear error when a header value contains a newline',
      ],
    };
  }
  return {
    ask: [
      `How is the main entry point structured in ${short}?`,
      `Where is configuration loaded and validated in ${short}?`,
      'How are errors handled across modules?',
    ],
    plan: [
      `Review code structure in ${short}`,
      `Suggest how to simplify error handling in ${short}`,
      `Audit error handling and refactor retries in ${short}`,
      `Fix an open issue about error handling in ${short}`,
    ],
    fix: [
      'Fix the crash when a configuration value is missing',
      'Resolve an off-by-one error in a loop over the input data',
      'Make sure opened files are always closed',
    ],
  };
}

function readSession(key, fallback) {
  try {
    const raw = sessionStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
}

export function AppStoreProvider({ children }) {
  // ---- server state: checking -> waking -> ready, or down after the give-up time ----
  const [server, setServer] = useState('checking');
  const [wakeSeconds, setWakeSeconds] = useState(0);
  const stopWake = useRef(null);

  // ---- data from GET /stats ----
  const [stats, setStats] = useState(null);
  const [statsError, setStatsError] = useState('');

  const [activeRepo, setActiveRepoState] = useState(() => {
    try {
      return localStorage.getItem('agent_active_repo') || '';
    } catch {
      return '';
    }
  });

  // The activity log lives only in this browser session. It is not saved on the server.
  const [activityLog, setActivityLog] = useState(() => readSession('agent_activity_log', []));

  const refreshStats = useCallback(async () => {
    try {
      const data = await api.stats(50);
      setStats(data);
      setStatsError('');
      return data;
    } catch (err) {
      setStatsError(err.message || 'Could not load the repository statistics.');
      return null;
    }
  }, []);

  const startWake = useCallback(() => {
    if (stopWake.current) stopWake.current();
    let cancelled = false;
    let timer = null;
    const startedAt = Date.now();
    setServer('checking');
    setWakeSeconds(0);

    const tick = async () => {
      if (cancelled) return;
      const ok = await api.healthOk();
      if (cancelled) return;
      if (ok) {
        setServer('ready');
        refreshStats();
        return;
      }
      const waited = Date.now() - startedAt;
      setWakeSeconds(Math.round(waited / 1000));
      if (waited >= WAKE_GIVE_UP_MS) {
        setServer('down');
        return;
      }
      setServer('waking');
      timer = setTimeout(tick, WAKE_EVERY_MS);
    };
    tick();

    stopWake.current = () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [refreshStats]);

  useEffect(() => {
    startWake();
    return () => {
      if (stopWake.current) stopWake.current();
    };
  }, [startWake]);

  // Any page calls this when a request fails with a network error: the server may have gone back to sleep.
  const markUnreachable = useCallback(() => {
    startWake();
  }, [startWake]);

  // ---- repositories (all from the server) ----
  const repos = useMemo(() => (stats && Array.isArray(stats.repos) ? stats.repos : []), [stats]);

  // Keep the selected repo valid: if it is gone (visitor repos can be removed), fall back to the demo or the first repo.
  useEffect(() => {
    if (!repos.length) return;
    if (!repos.some((r) => r.repo === activeRepo)) {
      const next = repos.find((r) => r.is_demo) || repos[0];
      setActiveRepoState(next.repo);
    }
  }, [repos, activeRepo]);

  useEffect(() => {
    try {
      if (activeRepo) localStorage.setItem('agent_active_repo', activeRepo);
    } catch {
      // storage can be blocked; the app still works
    }
  }, [activeRepo]);

  useEffect(() => {
    try {
      sessionStorage.setItem('agent_activity_log', JSON.stringify(activityLog));
    } catch {
      // ignore
    }
  }, [activityLog]);

  const setActiveRepo = useCallback((name) => setActiveRepoState(name), []);

  const logActivity = useCallback((type, desc) => {
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    setActivityLog((prev) => [{ id: Date.now() + Math.random(), type, desc, time }, ...prev.slice(0, 99)]);
  }, []);
  const clearActivity = useCallback(() => setActivityLog([]), []);

  const currentRepo = repos.find((r) => r.repo === activeRepo) || null;
  const [owner, repo] = (currentRepo ? currentRepo.repo : '').split('/');
  const limits = (stats && stats.limits) || null;

  const value = {
    server,
    wakeSeconds,
    retryServer: startWake,
    markUnreachable,

    stats,
    statsError,
    refreshStats,
    repos,
    activeRepo: currentRepo ? currentRepo.repo : '',
    currentRepo,
    setActiveRepo,
    target: currentRepo ? { owner, repo, ref: currentRepo.commit || currentRepo.ref || '' } : null,
    totalChunks: stats ? stats.total_chunks || 0 : 0,

    limits,
    aiLeft: limits && limits.ai ? limits.ai.calls_left_today : null,
    aiPerDay: limits && limits.ai ? limits.ai.calls_per_day : null,
    publicIndexing: limits ? limits.public_indexing : null,

    activityLog,
    logActivity,
    clearActivity,
    suggestions: suggestionsFor(currentRepo ? currentRepo.repo : ''),
  };

  return <AppStoreContext.Provider value={value}>{children}</AppStoreContext.Provider>;
}

export function useAppStore() {
  const ctx = useContext(AppStoreContext);
  if (!ctx) throw new Error('useAppStore must be used within an AppStoreProvider');
  return ctx;
}

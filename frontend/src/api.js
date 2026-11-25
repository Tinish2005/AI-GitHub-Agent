// API client for the FastAPI backend.
// The server address comes from VITE_API_URL (see .env.example). Never put the admin key in the frontend:
// anything in a frontend environment variable is public.
const BASE = (import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '');

export class ApiError extends Error {
  constructor(message, { status = 0, retryAfter = null, network = false } = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status; // HTTP status, 0 when the server could not be reached
    this.retryAfter = retryAfter; // seconds, from retry_after_seconds in a 429 body
    this.network = network; // true for network errors, timeouts and blocked (CORS) requests
  }
}

function detailText(detail) {
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((d) => `${(d.loc || []).filter((p) => p !== 'body').join('.')}: ${d.msg}`)
      .join('; ');
  }
  return '';
}

async function request(method, path, { body, timeout = 120000 } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  let res;
  try {
    res = await fetch(`${BASE}${path}`, {
      method,
      headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: controller.signal,
    });
  } catch (err) {
    const timedOut = err && err.name === 'AbortError';
    throw new ApiError(
      timedOut ? 'The server took too long to answer.' : 'Cannot reach the server.',
      { network: true },
    );
  } finally {
    clearTimeout(timer);
  }

  const raw = await res.text();
  let data = null;
  try {
    data = raw ? JSON.parse(raw) : null;
  } catch {
    data = null; // not JSON (for example a proxy error page)
  }

  if (!res.ok) {
    const message = detailText(data && data.detail) || (typeof raw === 'string' && raw.length < 200 ? raw : '') || `Request failed (HTTP ${res.status}).`;
    const retry = data && typeof data.retry_after_seconds === 'number' ? data.retry_after_seconds : null;
    throw new ApiError(message, { status: res.status, retryAfter: retry, network: res.status >= 502 && res.status <= 504 && !data });
  }
  if (data === null) {
    throw new ApiError('The server sent an answer that is not JSON.', { status: res.status, network: true });
  }
  return data;
}

// "owner/repo", "https://github.com/owner/repo" or "github.com/owner/repo/"  ->  { owner, repo }
export function parseGitHubUrl(input) {
  const clean = (input || '').trim();
  if (!clean) throw new Error('Enter a GitHub repository as owner/repo or a github.com link.');
  const stripped = clean
    .replace(/^https?:\/\/(www\.)?github\.com\//i, '')
    .replace(/^github\.com\//i, '')
    .replace(/\.git$/i, '')
    .replace(/^\/+|\/+$/g, '');
  const parts = stripped.split('/');
  if (parts.length >= 2 && /^[\w.-]+$/.test(parts[0]) && /^[\w.-]+$/.test(parts[1])) {
    // A link to a folder, like github.com/owner/repo/tree/main/src/pkg, also names the folder to index.
    const path = parts[2] === 'tree' && parts.length >= 5 ? parts.slice(4).join('/') : '';
    return { owner: parts[0], repo: parts[1], path };
  }
  throw new Error('Enter a valid GitHub repository, for example psf/requests or https://github.com/psf/requests');
}

export const api = {
  // GET /health. True only when the server says status "ok" (a 200 from anything else means "not ready").
  async healthOk() {
    try {
      const data = await request('GET', '/health', { timeout: 5000 });
      return data && data.status === 'ok';
    } catch {
      return false;
    }
  },

  // GET /stats?top=50  (repos, files, chunks, import links, limits)
  stats: (top = 50) => request('GET', `/stats?top=${top}`, { timeout: 20000 }),

  // POST /qa
  ask: ({ question, owner, repo, ref, topK = 5 }) =>
    request('POST', '/qa', { body: { question, owner, repo, ref: ref || null, top_k: topK } }),

  // POST /plan  (takes only the goal; planning does not use AI calls)
  plan: (goal) => request('POST', '/plan', { body: { goal, strategy: 'rule_based' }, timeout: 30000 }),

  // POST /execute
  execute: ({ goal, owner, repo, ref, abortOnFailure = true }) =>
    request('POST', '/execute', {
      body: { goal, owner, repo, ref: ref || null, strategy: 'rule_based', abort_on_failure: abortOnFailure },
      timeout: 240000,
    }),

  // POST /fix/propose
  fixPropose: ({ goal, owner, repo, ref }) =>
    request('POST', '/fix/propose', { body: { goal, owner, repo, ref: ref || null } }),

  // POST /fix/validate. The proposal goes back exactly as it came from /fix/propose (never edit the diff).
  fixValidate: ({ goal, owner, repo, ref, proposal }) =>
    request('POST', '/fix/validate', {
      body: {
        goal,
        owner,
        repo,
        ref: ref || null,
        proposal: {
          goal: proposal.goal,
          explanation: proposal.explanation,
          diff: proposal.diff,
          confidence: proposal.confidence,
          model: proposal.model,
          diff_sha256: proposal.diff_sha256,
        },
      },
      timeout: 60000,
    }),

  // POST /index/public  -> 202 {job_id, ...} or 200 {status: "already_indexed", ...}
  // The optional folder limits indexing to one part of the repository (sent only when given).
  indexPublic: (owner, repo, path = '') =>
    request('POST', '/index/public', { body: path ? { owner, repo, path } : { owner, repo }, timeout: 30000 }),

  // GET /index/public/{job_id}
  indexStatus: (jobId) => request('GET', `/index/public/${encodeURIComponent(jobId)}`, { timeout: 15000 }),

  // GET /github/file  (used to show the code behind an answer's sources)
  githubFile: ({ owner, repo, path, ref }) => {
    const q = new URLSearchParams({ owner, repo, path });
    if (ref) q.set('ref', ref);
    return request('GET', `/github/file?${q.toString()}`, { timeout: 30000 });
  },
};

// Decode the base64 content that GET /github/file returns.
export function decodeFileContent(file) {
  if (!file || typeof file.content !== 'string') return '';
  if (file.encoding && file.encoding !== 'base64') return file.content;
  try {
    const bin = atob(file.content.replace(/\s/g, ''));
    const bytes = Uint8Array.from(bin, (c) => c.charCodeAt(0));
    return new TextDecoder('utf-8').decode(bytes);
  } catch {
    return '';
  }
}

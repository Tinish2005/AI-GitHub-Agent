async function post(path, body) {
  const res = await fetch(`/api${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) {
    const detail = data && data.detail ? JSON.stringify(data.detail) : res.statusText;
    throw new Error(detail);
  }
  return data;
}

async function get(path) {
  const res = await fetch(`/api${path}`);
  const data = await res.json();
  if (!res.ok) throw new Error(res.statusText);
  return data;
}

export const api = {
  health: () => get("/health"),
  ask: (question, top_k = 5) => post("/qa", { question, top_k }),
  plan: (goal, strategy = "rule_based") => post("/plan", { goal, strategy }),
  execute: (goal, strategy = "rule_based") => post("/execute", { goal, strategy }),
  proposeFix: (goal, context = "") => post("/fix/propose", { goal, context }),
  validateFix: (goal, context = "") => post("/fix/validate", { goal, context }),
  indexGithub: (owner, repo, max_files = 10) => post("/index/github", { owner, repo, max_files }),
};

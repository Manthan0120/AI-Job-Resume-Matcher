// Central API client. In dev, Vite proxies /api -> http://localhost:8000
// (see vite.config.js), so relative paths work without CORS headaches.

const BASE = "/api";

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: options.body instanceof FormData ? {} : { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `Request failed: ${res.status}`);
  }
  return res.json();
}

export const api = {
  health: () => request("/health"),

  setApiKey: (apiKey) =>
    request("/config/api-key", {
      method: "POST",
      body: JSON.stringify({ api_key: apiKey }),
    }),

  uploadResumes: (files) => {
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    return request("/documents/resumes", { method: "POST", body: form });
  },

  uploadJobs: (file) => {
    const form = new FormData();
    form.append("file", file);
    return request("/documents/jobs", { method: "POST", body: form });
  },

  matchResumeToJobs: (content, topK) =>
    request("/match/resume-to-jobs", {
      method: "POST",
      body: JSON.stringify({ content, top_k: topK }),
    }),

  matchJobToResumes: (content, topK) =>
    request("/match/job-to-resumes", {
      method: "POST",
      body: JSON.stringify({ content, top_k: topK }),
    }),

  agentCareerMatch: (resumeContent, topK) =>
    request("/agent/career-match", {
      method: "POST",
      body: JSON.stringify({ resume_content: resumeContent, top_k: topK }),
    }),

  chat: (question, chatHistory) =>
    request("/chat", {
      method: "POST",
      body: JSON.stringify({ question, chat_history: chatHistory }),
    }),

  analyticsMetrics: () => request("/analytics/metrics"),
};

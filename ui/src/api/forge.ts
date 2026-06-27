/**
 * Forge API Client
 *
 * Communicates with the Forge backend (curriculum builder) via REST API.
 */

const API_BASE = "http://127.0.0.1:8420/api/forge";

// ============================================================
// Type definitions matching the Python backend
// ============================================================

export interface Contributor {
  id: string;
  name: string;
  role: string;
  expertise_areas: string[];
  created_at: string;
}

export interface Discipline {
  id: string;
  name: string;
  description: string;
  status: "draft" | "active" | "archived";
  contributor_ids: string[];
  created_at: string;
  updated_at: string | null;
}

export interface Competency {
  id: string;
  discipline_id: string;
  name: string;
  description: string;
  level: "foundational" | "intermediate" | "advanced" | "expert";
  parent_id: string | null;
  example_count: number;
  target_count: number;
  created_at: string;
}

export interface Example {
  id: string;
  competency_id: string;
  question: string;
  answer: string;
  context: string | null;
  reasoning_pattern: string | null;
  status: "draft" | "approved" | "rejected" | "needs_revision";
  contributor_id: string;
  reviewer_id: string | null;
  review_notes: string | null;
  created_at: string;
  updated_at: string | null;
}

export interface DiscoverySession {
  id: string;
  discipline_id: string;
  phase: string;
  current_question_index: number;
  total_questions: number;
  answers: Record<string, string>;
  completed: boolean;
  created_at: string;
}

export interface ConsistencyIssue {
  id: string;
  type: string;
  severity: "high" | "medium" | "low";
  description: string;
  affected_example_ids: string[];
  suggested_fix: string | null;
}

export interface ConsistencyReport {
  discipline_id: string;
  total_issues: number;
  by_severity: Record<string, number>;
  issues: ConsistencyIssue[];
  checked_at: string;
}

export interface CoverageReport {
  discipline_id: string;
  total_competencies: number;
  covered_competencies: number;
  coverage_pct: number;
  gaps: Array<{
    competency_id: string;
    name: string;
    current: number;
    target: number;
  }>;
}

export interface CurriculumVersion {
  id: string;
  discipline_id: string;
  version: string;
  example_count: number;
  created_at: string;
}

// ============================================================
// API Error
// ============================================================

export class ForgeAPIError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.status = status;
    this.detail = detail;
  }
}

// ============================================================
// Generic fetch wrapper
// ============================================================

async function forgeFetch<T>(
  endpoint: string,
  options: RequestInit = {},
): Promise<T> {
  const url = `${API_BASE}${endpoint}`;
  const response = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
  });

  if (!response.ok) {
    const error = await response
      .json()
      .catch(() => ({ detail: "Unknown error" }));
    throw new ForgeAPIError(
      response.status,
      error.detail || response.statusText,
    );
  }

  return response.json();
}

// ============================================================
// Endpoint groups
// ============================================================

export const forgeHealthAPI = {
  check: () => forgeFetch<{ status: string }>("/health"),
};

export const contributorAPI = {
  list: () => forgeFetch<Contributor[]>("/contributors"),

  get: (id: string) => forgeFetch<Contributor>(`/contributors/${id}`),

  create: (data: { name: string; role: string; expertise_areas: string[] }) =>
    forgeFetch<Contributor>("/contributors", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  update: (id: string, data: Partial<Contributor>) =>
    forgeFetch<Contributor>(`/contributors/${id}`, {
      method: "PUT",
      body: JSON.stringify(data),
    }),

  delete: (id: string) =>
    forgeFetch<{ deleted: boolean }>(`/contributors/${id}`, {
      method: "DELETE",
    }),
};

export const disciplineAPI = {
  list: () => forgeFetch<Discipline[]>("/disciplines"),

  get: (id: string) => forgeFetch<Discipline>(`/disciplines/${id}`),

  create: (data: { name: string; description: string }) =>
    forgeFetch<Discipline>("/disciplines", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  update: (id: string, data: Partial<Discipline>) =>
    forgeFetch<Discipline>(`/disciplines/${id}`, {
      method: "PUT",
      body: JSON.stringify(data),
    }),
};

export const competencyAPI = {
  list: (disciplineId: string) =>
    forgeFetch<Competency[]>(`/disciplines/${disciplineId}/competencies`),

  get: (id: string) => forgeFetch<Competency>(`/competencies/${id}`),

  create: (data: {
    discipline_id: string;
    name: string;
    description: string;
    level: string;
    parent_id?: string | null;
    target_count?: number;
  }) =>
    forgeFetch<Competency>("/competencies", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  update: (id: string, data: Partial<Competency>) =>
    forgeFetch<Competency>(`/competencies/${id}`, {
      method: "PUT",
      body: JSON.stringify(data),
    }),

  delete: (id: string) =>
    forgeFetch<{ deleted: boolean }>(`/competencies/${id}`, {
      method: "DELETE",
    }),
};

export const exampleAPI = {
  list: (competencyId: string) =>
    forgeFetch<Example[]>(`/competencies/${competencyId}/examples`),

  get: (id: string) => forgeFetch<Example>(`/examples/${id}`),

  create: (data: {
    competency_id: string;
    question: string;
    answer: string;
    context?: string | null;
    reasoning_pattern?: string | null;
    contributor_id: string;
  }) =>
    forgeFetch<Example>("/examples", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  update: (id: string, data: Partial<Example>) =>
    forgeFetch<Example>(`/examples/${id}`, {
      method: "PUT",
      body: JSON.stringify(data),
    }),

  delete: (id: string) =>
    forgeFetch<{ deleted: boolean }>(`/examples/${id}`, { method: "DELETE" }),
};

export const discoveryAPI = {
  start: (disciplineId: string) =>
    forgeFetch<DiscoverySession>("/discovery/start", {
      method: "POST",
      body: JSON.stringify({ discipline_id: disciplineId }),
    }),

  answer: (sessionId: string, answer: string) =>
    forgeFetch<DiscoverySession>("/discovery/answer", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, answer }),
    }),

  getProgress: (sessionId: string) =>
    forgeFetch<DiscoverySession>(`/discovery/${sessionId}/progress`),
};

export const consistencyAPI = {
  check: (disciplineId: string) =>
    forgeFetch<ConsistencyReport>(`/consistency/check/${disciplineId}`, {
      method: "POST",
    }),

  getReport: (disciplineId: string) =>
    forgeFetch<ConsistencyReport>(`/consistency/report/${disciplineId}`),
};

export const coverageAPI = {
  get: (disciplineId: string) =>
    forgeFetch<CoverageReport>(`/coverage/${disciplineId}`),
};

export const curriculumAPI = {
  export: (disciplineId: string, format: string = "jsonl") =>
    forgeFetch<{ path: string; example_count: number; exported_at: string }>(
      `/curriculum/export/${disciplineId}`,
      {
        method: "POST",
        body: JSON.stringify({ format }),
      },
    ),
};

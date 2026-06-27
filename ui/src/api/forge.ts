/**
 * Forge API Client
 *
 * Communicates with the Forge backend (curriculum builder) via REST API.
 *
 * The Forge FastAPI router (forge/src/server.py) returns domain `to_dict()`
 * shapes and wraps list results under named keys
 * ({ "disciplines": [...] }, { "competencies": [...] }, { "examples": [...] }).
 * Field names also differ from the UI's normalized shapes, e.g.:
 *   - Competency: backend has `coverage_target` (no `level`/`example_count`)
 *   - Example: backend has `ideal_answer` + `review_status` (UI uses
 *     `answer` + `status`, and the backend's "pending" maps to UI "draft")
 *   - ConsistencyIssue: backend has issue_type/message/example_id/severity
 *     (info|warning|error), mapped to UI type/description/affected_example_ids/
 *     severity (low|medium|high)
 *
 * Each method translates the real backend payload into the interfaces declared
 * in this file, which are the contract the UI components rely on.
 */

const API_BASE = "http://127.0.0.1:8420/api/forge";

/**
 * The backend requires `created_by` on discipline creation, but the UI has no
 * contributor-selection step here yet. A stable default identifies UI-created
 * records until a contributor picker is added (see report notes).
 */
const UI_AUTHOR = "kiln-ui";

// ============================================================
// Type definitions (UI-normalized shapes)
// ============================================================

export interface Contributor {
  id: string;
  name: string;
  email: string;
  created_at: string;
}

export interface Discipline {
  id: string;
  name: string;
  description: string;
  status: "draft" | "active" | "archived";
  created_by: string;
  vocabulary: string[];
  document_types: string[];
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
  status: "draft" | "approved" | "rejected" | "needs_revision";
  contributor_id: string;
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
  total_examples: number;
  total_test_examples: number;
  competency_coverage: Array<{
    competency_id: string;
    competency_name: string;
    example_count: number;
    coverage_target: number;
    met: boolean;
  }>;
  gaps: Array<{
    competency_id: string;
    competency_name: string;
    example_count: number;
    coverage_target: number;
    met: boolean;
  }>;
  coverage_complete: boolean;
}

export interface CurriculumVersion {
  version_id: string;
  discipline_id: string;
  version_number: number;
  example_count: number;
  status: string;
  created_at: string;
}

export interface DiscoveryQuestion {
  question_id: string;
  phase: string;
  text: string;
  hint: string;
  response_type: string;
  required: boolean;
}

export interface DiscoverySession {
  session: Record<string, unknown>;
  current_questions: DiscoveryQuestion[];
}

// ============================================================
// Raw backend payload shapes (forge/src/models.py to_dict)
// ============================================================

interface RawCompetency {
  id: string;
  name: string;
  description: string;
  discipline_id: string;
  parent_id: string | null;
  coverage_target: number;
  created_at: string;
  updated_at: string;
}

interface RawExample {
  id: string;
  question: string;
  ideal_answer: string;
  competency_id: string;
  contributor_id: string;
  discipline_id: string;
  variants: string[];
  context: string;
  review_status: "pending" | "approved" | "rejected" | "needs_revision";
  created_at: string;
}

interface RawConsistencyIssue {
  issue_type: string;
  severity: "info" | "warning" | "error";
  message: string;
  example_id: string | null;
  suggested_fix: string | null;
  details: Record<string, unknown>;
}

// ============================================================
// Transforms
// ============================================================

function toCompetency(raw: RawCompetency): Competency {
  return {
    id: raw.id,
    discipline_id: raw.discipline_id,
    name: raw.name,
    description: raw.description,
    // The backend has no competency "level"; default for display.
    level: "foundational",
    parent_id: raw.parent_id,
    // Per-competency example counts come from the coverage report, not here.
    example_count: 0,
    target_count: raw.coverage_target,
    created_at: raw.created_at,
  };
}

function toExample(raw: RawExample): Example {
  const status =
    raw.review_status === "pending"
      ? "draft"
      : (raw.review_status as Example["status"]);
  return {
    id: raw.id,
    competency_id: raw.competency_id,
    question: raw.question,
    answer: raw.ideal_answer,
    context: raw.context || null,
    status,
    contributor_id: raw.contributor_id,
    created_at: raw.created_at,
  };
}

const SEVERITY_MAP: Record<string, ConsistencyIssue["severity"]> = {
  error: "high",
  warning: "medium",
  info: "low",
};

function toConsistencyIssue(
  raw: RawConsistencyIssue,
  index: number,
): ConsistencyIssue {
  return {
    id: raw.example_id
      ? `${raw.issue_type}:${raw.example_id}`
      : `issue-${index}`,
    type: raw.issue_type,
    severity: SEVERITY_MAP[raw.severity] ?? "low",
    description: raw.message,
    affected_example_ids: raw.example_id ? [raw.example_id] : [],
    suggested_fix: raw.suggested_fix,
  };
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
  list: async (): Promise<Contributor[]> => {
    const data = await forgeFetch<{ contributors: Contributor[] }>(
      "/contributors",
    );
    return data.contributors ?? [];
  },

  get: (id: string) => forgeFetch<Contributor>(`/contributors/${id}`),

  create: (data: { name: string; email?: string }) =>
    forgeFetch<Contributor>("/contributors", {
      method: "POST",
      body: JSON.stringify({ name: data.name, email: data.email ?? "" }),
    }),

  update: (id: string, data: Partial<Contributor>) =>
    forgeFetch<Contributor>(`/contributors/${id}`, {
      method: "PUT",
      body: JSON.stringify(data),
    }),

  delete: (id: string) =>
    forgeFetch<{ deleted: string }>(`/contributors/${id}`, {
      method: "DELETE",
    }),
};

export const disciplineAPI = {
  list: async (): Promise<Discipline[]> => {
    const data = await forgeFetch<{ disciplines: Discipline[] }>(
      "/disciplines",
    );
    return data.disciplines ?? [];
  },

  get: (id: string) => forgeFetch<Discipline>(`/disciplines/${id}`),

  create: (data: { name: string; description: string }) =>
    forgeFetch<Discipline>("/disciplines", {
      method: "POST",
      body: JSON.stringify({
        name: data.name,
        description: data.description,
        created_by: UI_AUTHOR,
      }),
    }),

  update: (id: string, data: Partial<Discipline>) =>
    forgeFetch<Discipline>(`/disciplines/${id}`, {
      method: "PUT",
      body: JSON.stringify(data),
    }),
};

export const competencyAPI = {
  list: async (disciplineId: string): Promise<Competency[]> => {
    const data = await forgeFetch<{ competencies: RawCompetency[] }>(
      `/disciplines/${disciplineId}/competencies`,
    );
    return (data.competencies ?? []).map(toCompetency);
  },

  get: async (id: string): Promise<Competency> => {
    const raw = await forgeFetch<RawCompetency>(`/competencies/${id}`);
    return toCompetency(raw);
  },

  create: async (data: {
    discipline_id: string;
    name: string;
    description: string;
    level?: string;
    parent_id?: string | null;
    target_count?: number;
  }): Promise<Competency> => {
    const raw = await forgeFetch<RawCompetency>("/competencies", {
      method: "POST",
      body: JSON.stringify({
        discipline_id: data.discipline_id,
        name: data.name,
        description: data.description,
        parent_id: data.parent_id ?? null,
        coverage_target: data.target_count ?? 25,
      }),
    });
    return toCompetency(raw);
  },

  update: async (
    id: string,
    data: Partial<{
      name: string;
      description: string;
      target_count: number;
      parent_id: string | null;
    }>,
  ): Promise<Competency> => {
    const body: Record<string, unknown> = {};
    if (data.name !== undefined) body.name = data.name;
    if (data.description !== undefined) body.description = data.description;
    if (data.target_count !== undefined)
      body.coverage_target = data.target_count;
    if (data.parent_id !== undefined) body.parent_id = data.parent_id;
    const raw = await forgeFetch<RawCompetency>(`/competencies/${id}`, {
      method: "PUT",
      body: JSON.stringify(body),
    });
    return toCompetency(raw);
  },

  delete: (id: string) =>
    forgeFetch<{ deleted: string }>(`/competencies/${id}`, {
      method: "DELETE",
    }),
};

export const exampleAPI = {
  list: async (competencyId: string): Promise<Example[]> => {
    const data = await forgeFetch<{ examples: RawExample[] }>(
      `/competencies/${competencyId}/examples`,
    );
    return (data.examples ?? []).map(toExample);
  },

  get: async (id: string): Promise<Example> => {
    const raw = await forgeFetch<RawExample>(`/examples/${id}`);
    return toExample(raw);
  },

  create: async (data: {
    competency_id: string;
    discipline_id: string;
    question: string;
    answer: string;
    context?: string | null;
    contributor_id: string;
  }): Promise<Example> => {
    const raw = await forgeFetch<RawExample>("/examples", {
      method: "POST",
      body: JSON.stringify({
        competency_id: data.competency_id,
        discipline_id: data.discipline_id,
        question: data.question,
        ideal_answer: data.answer,
        context: data.context ?? "",
        contributor_id: data.contributor_id,
      }),
    });
    return toExample(raw);
  },

  update: async (
    id: string,
    data: Partial<{
      question: string;
      answer: string;
      context: string | null;
      status: Example["status"];
    }>,
  ): Promise<Example> => {
    const body: Record<string, unknown> = {};
    if (data.question !== undefined) body.question = data.question;
    if (data.answer !== undefined) body.ideal_answer = data.answer;
    if (data.context !== undefined) body.context = data.context;
    if (data.status !== undefined) body.review_status = data.status;
    const raw = await forgeFetch<RawExample>(`/examples/${id}`, {
      method: "PUT",
      body: JSON.stringify(body),
    });
    return toExample(raw);
  },

  delete: (id: string) =>
    forgeFetch<{ deleted: string }>(`/examples/${id}`, { method: "DELETE" }),
};

export const discoveryAPI = {
  /** Note: not yet wired to a screen; DiscoveryWizard uses local state. */
  start: (disciplineName: string, contributorId: string) =>
    forgeFetch<DiscoverySession>("/discovery/start", {
      method: "POST",
      body: JSON.stringify({
        discipline_name: disciplineName,
        contributor_id: contributorId,
      }),
    }),

  answer: (
    sessionId: string,
    questionId: string,
    answer: { raw_text?: string; items?: string[]; scale_value?: number },
  ) =>
    forgeFetch<DiscoverySession>("/discovery/answer", {
      method: "POST",
      body: JSON.stringify({
        session_id: sessionId,
        question_id: questionId,
        raw_text: answer.raw_text ?? "",
        items: answer.items ?? [],
        scale_value: answer.scale_value ?? null,
      }),
    }),

  getProgress: (sessionId: string) =>
    forgeFetch<{
      session_id: string;
      current_phase: string;
      phases_complete: string[];
      completion_percentage: number;
      unanswered_required: number;
      estimated_minutes_remaining: number;
    }>(`/discovery/${sessionId}/progress`),
};

export const consistencyAPI = {
  check: async (disciplineId: string): Promise<ConsistencyReport> => {
    const data = await forgeFetch<{
      discipline_id: string;
      example_count: number;
      has_errors: boolean;
      has_warnings: boolean;
      issues: RawConsistencyIssue[];
      checked_at: string;
    }>(`/consistency/check/${disciplineId}`, { method: "POST" });
    const issues = (data.issues ?? []).map(toConsistencyIssue);
    const by_severity = issues.reduce<Record<string, number>>((acc, i) => {
      acc[i.severity] = (acc[i.severity] ?? 0) + 1;
      return acc;
    }, {});
    return {
      discipline_id: data.discipline_id,
      total_issues: issues.length,
      by_severity,
      issues,
      checked_at: data.checked_at,
    };
  },

  getReport: async (disciplineId: string): Promise<ConsistencyReport> =>
    consistencyAPI.check(disciplineId),
};

export const coverageAPI = {
  get: (disciplineId: string) =>
    forgeFetch<CoverageReport>(`/coverage/${disciplineId}`),
};

export const curriculumAPI = {
  export: (disciplineId: string, includeTestSet: boolean = false) =>
    forgeFetch<CurriculumVersion>(`/curriculum/export/${disciplineId}`, {
      method: "POST",
      body: JSON.stringify({
        created_by: UI_AUTHOR,
        include_test_set: includeTestSet,
      }),
    }),
};

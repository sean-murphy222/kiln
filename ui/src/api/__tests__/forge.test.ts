import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
import {
  disciplineAPI,
  competencyAPI,
  exampleAPI,
  consistencyAPI,
} from "@/api/forge";

function mockFetchOnce(payload: unknown) {
  const fetchMock = vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    json: async () => payload,
  });
  global.fetch = fetchMock as unknown as typeof fetch;
  return fetchMock;
}

describe("forge client", () => {
  beforeEach(() => vi.clearAllMocks());
  afterEach(() => vi.restoreAllMocks());

  it("GET /disciplines unwraps { disciplines }", async () => {
    mockFetchOnce({
      disciplines: [
        {
          id: "d1",
          name: "Maintenance",
          description: "",
          status: "active",
          created_by: "kiln-ui",
          vocabulary: [],
          document_types: [],
          created_at: "2026-01-01",
          updated_at: null,
        },
      ],
    });
    const ds = await disciplineAPI.list();
    expect(ds).toHaveLength(1);
    expect(ds[0].id).toBe("d1");
  });

  it("POST /disciplines includes created_by", async () => {
    const fetchMock = mockFetchOnce({
      id: "d2",
      name: "Comms",
      description: "",
      status: "draft",
      created_by: "kiln-ui",
      vocabulary: [],
      document_types: [],
      created_at: "2026-01-02",
      updated_at: null,
    });
    await disciplineAPI.create({ name: "Comms", description: "" });
    const body = JSON.parse(
      (fetchMock.mock.calls[0][1] as RequestInit).body as string,
    );
    expect(body.created_by).toBe("kiln-ui");
  });

  it("competency list maps coverage_target -> target_count and defaults level", async () => {
    mockFetchOnce({
      competencies: [
        {
          id: "c1",
          name: "Diagnose faults",
          description: "",
          discipline_id: "d1",
          parent_id: null,
          coverage_target: 30,
          created_at: "2026-01-01",
          updated_at: "2026-01-01",
        },
      ],
    });
    const comps = await competencyAPI.list("d1");
    expect(comps[0].target_count).toBe(30);
    expect(comps[0].level).toBe("foundational");
    expect(comps[0].example_count).toBe(0);
  });

  it("example list maps ideal_answer/review_status and pending -> draft", async () => {
    mockFetchOnce({
      examples: [
        {
          id: "e1",
          question: "Q?",
          ideal_answer: "A.",
          competency_id: "c1",
          contributor_id: "u1",
          discipline_id: "d1",
          variants: [],
          context: "",
          review_status: "pending",
          created_at: "2026-01-01",
        },
      ],
    });
    const examples = await exampleAPI.list("c1");
    expect(examples[0].answer).toBe("A.");
    expect(examples[0].status).toBe("draft");
  });

  it("example update sends review_status (not status)", async () => {
    const fetchMock = mockFetchOnce({
      id: "e1",
      question: "Q?",
      ideal_answer: "A.",
      competency_id: "c1",
      contributor_id: "u1",
      discipline_id: "d1",
      variants: [],
      context: "",
      review_status: "approved",
      created_at: "2026-01-01",
    });
    const updated = await exampleAPI.update("e1", { status: "approved" });
    const body = JSON.parse(
      (fetchMock.mock.calls[0][1] as RequestInit).body as string,
    );
    expect(body.review_status).toBe("approved");
    expect(body.status).toBeUndefined();
    expect(updated.status).toBe("approved");
  });

  it("consistency check maps issue_type/message/severity and unwraps issues", async () => {
    mockFetchOnce({
      discipline_id: "d1",
      example_count: 5,
      has_errors: true,
      has_warnings: false,
      checked_at: "2026-02-01T00:00:00Z",
      issues: [
        {
          issue_type: "Contradiction",
          severity: "error",
          message: "Contradictory training signals detected.",
          example_id: "e1",
          suggested_fix: "Merge the examples.",
          details: {},
        },
      ],
    });
    const report = await consistencyAPI.check("d1");
    expect(report.total_issues).toBe(1);
    expect(report.issues[0].type).toBe("Contradiction");
    expect(report.issues[0].severity).toBe("high");
    expect(report.issues[0].description).toBe(
      "Contradictory training signals detected.",
    );
    expect(report.issues[0].affected_example_ids).toEqual(["e1"]);
  });
});

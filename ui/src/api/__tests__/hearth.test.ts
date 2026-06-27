import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
import { queryAPI, modelAPI, conversationAPI } from "@/api/hearth";

function mockFetchOnce(payload: unknown) {
  const fetchMock = vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    json: async () => payload,
  });
  global.fetch = fetchMock as unknown as typeof fetch;
  return fetchMock;
}

describe("hearth client", () => {
  beforeEach(() => vi.clearAllMocks());
  afterEach(() => vi.restoreAllMocks());

  it("POST /query sends slot_id and maps the QueryResponse + citations", async () => {
    const fetchMock = mockFetchOnce({
      answer: "Perform PMCS per TM-9 [1].",
      citations: [
        {
          document_title: "TM-9-2320-280-10",
          section: "2.3 Engine",
          page: 42,
          relevance_score: 0.91,
          snippet: "Perform PMCS at intervals.",
        },
      ],
      conversation_id: "conv-1",
      model_used: "slot-1",
      latency_ms: 123,
      chunk_count: 3,
    });

    const res = await queryAPI.send({
      query: "How do I PMCS the engine?",
      model_id: "slot-1",
      conversation_id: "conv-1",
    });

    // Correct path + body field name (slot_id, not model_id).
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("http://127.0.0.1:8420/api/hearth/query");
    const body = JSON.parse((init as RequestInit).body as string);
    expect(body.slot_id).toBe("slot-1");
    expect(body.query).toBe("How do I PMCS the engine?");

    // Response is mapped from { answer, citations, conversation_id, latency_ms }.
    expect(res.conversation_id).toBe("conv-1");
    expect(res.processing_time_ms).toBe(123);
    expect(res.message.role).toBe("assistant");
    expect(res.message.content).toBe("Perform PMCS per TM-9 [1].");
    expect(res.message.model_id).toBe("slot-1");
    expect(res.message.citations).toHaveLength(1);
    expect(res.message.citations[0].document_title).toBe("TM-9-2320-280-10");
  });

  it("GET /models unwraps { models } and maps slot fields", async () => {
    mockFetchOnce({
      models: [
        {
          slot_id: "slot-1",
          display_name: "Maintenance v1",
          base_model_family: "llama",
          status: "ready",
          model_path: null,
          discipline_id: null,
          lora_path: "/adapters/maint",
          loaded_at: null,
        },
      ],
    });

    const slots = await modelAPI.list();
    expect(slots).toHaveLength(1);
    expect(slots[0]).toMatchObject({
      id: "slot-1",
      name: "Maintenance v1",
      base_model: "llama",
      adapter_path: "/adapters/maint",
      status: "ready",
    });
  });

  it("GET /conversations unwraps { conversations } and maps ids", async () => {
    mockFetchOnce({
      conversations: [
        {
          conversation_id: "conv-1",
          title: "Engine PMCS",
          created_at: "2026-01-01",
          model_slot_id: "slot-1",
          turn_count: 2,
        },
      ],
    });

    const convos = await conversationAPI.list();
    expect(convos).toHaveLength(1);
    expect(convos[0]).toMatchObject({
      id: "conv-1",
      title: "Engine PMCS",
      model_id: "slot-1",
    });
    expect(convos[0].messages).toEqual([]);
  });
});

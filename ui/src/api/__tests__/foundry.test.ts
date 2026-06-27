import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
import { trainingAPI, regressionAPI, mergingAPI } from "@/api/foundry";

function mockFetchOnce(payload: unknown) {
  const fetchMock = vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    json: async () => payload,
  });
  global.fetch = fetchMock as unknown as typeof fetch;
  return fetchMock;
}

describe("foundry client", () => {
  beforeEach(() => vi.clearAllMocks());
  afterEach(() => vi.restoreAllMocks());

  it("GET /training/runs unwraps { runs } and synthesizes completed status", async () => {
    mockFetchOnce({
      runs: [
        {
          run_id: "run-1",
          config_path: "/c/config.json",
          result_path: "/c/result.json",
          adapter_path: "/adapters/maint-v1",
          discipline_id: "d1",
          created_at: "2026-01-01",
        },
      ],
      total: 1,
    });
    const runs = await trainingAPI.listRuns();
    expect(runs).toHaveLength(1);
    expect(runs[0].id).toBe("run-1");
    expect(runs[0].name).toBe("maint-v1");
    expect(runs[0].status).toBe("completed");
  });

  it("POST /training/configure sends base_model_family + output_dir", async () => {
    const fetchMock = mockFetchOnce({ run_id: "run-1", config: {} });
    await trainingAPI.configure({
      base_model: "meta-llama/Llama-3.1-8B",
      curriculum_path: "curriculum.jsonl",
      adapter_name: "maint-v1",
      lora_rank: 16,
      epochs: 3,
      learning_rate: 0.0002,
    });
    const body = JSON.parse(
      (fetchMock.mock.calls[0][1] as RequestInit).body as string,
    );
    expect(body.base_model_family).toBe("llama");
    expect(body.output_dir).toBe("outputs/maint-v1");
    expect(body.curriculum_path).toBe("curriculum.jsonl");
  });

  it("POST /training/start passes run_id as a query parameter", async () => {
    const fetchMock = mockFetchOnce({
      run_id: "run-1",
      registry_run_id: "reg-1",
      status: "completed",
      adapter_path: null,
    });
    await trainingAPI.start("run-1");
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(
      "http://127.0.0.1:8420/api/foundry/training/start?run_id=run-1",
    );
    expect((init as RequestInit).body).toBeUndefined();
  });

  it("GET /regression/versions unwraps { versions } and maps fields", async () => {
    mockFetchOnce({
      versions: [
        {
          version_id: "v1",
          model_name: "maint-v1",
          discipline_id: "d1",
          training_run_id: "run-1",
          evaluation_run_id: "eval-1",
          adapter_path: "/adapters/maint-v1",
          change_type: "retrain",
          change_description: "",
          created_at: "2026-01-01",
          is_active: true,
        },
      ],
      total: 1,
    });
    const versions = await regressionAPI.listVersions();
    expect(versions[0]).toMatchObject({
      id: "v1",
      name: "maint-v1",
      adapter_path: "/adapters/maint-v1",
    });
  });

  it("GET /merging/registry unwraps { merges } and maps fields", async () => {
    mockFetchOnce({
      merges: [
        {
          merge_id: "m1",
          method: "linear",
          status: "completed",
          adapters: [{ adapter_path: "/a/1" }, { adapter_path: "/a/2" }],
          weights_used: [0.5, 0.5],
          merged_adapter_path: "/a/merged",
          started_at: "2026-01-01",
          completed_at: "2026-01-01",
        },
      ],
      total: 1,
    });
    const merges = await mergingAPI.listRegistry();
    expect(merges[0]).toMatchObject({
      id: "m1",
      output_path: "/a/merged",
      method: "linear",
    });
    expect(merges[0].adapters_merged).toEqual(["/a/1", "/a/2"]);
  });
});

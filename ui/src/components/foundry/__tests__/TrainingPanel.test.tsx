import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, beforeEach, vi } from "vitest";
import { TrainingPanel } from "../TrainingPanel";
import { useFoundryStore } from "@/store/useFoundryStore";
import { trainingAPI } from "@/api/foundry";

vi.mock("@/api/foundry", () => ({
  trainingAPI: {
    listRuns: vi.fn(),
    configure: vi.fn(),
    start: vi.fn(),
    cancel: vi.fn(),
  },
}));

const mockedTrainingAPI = vi.mocked(trainingAPI);

const sampleRun = {
  id: "run-1",
  name: "adapter-x",
  status: "completed" as const,
  progress: 100,
  config: {
    base_model: "meta-llama/Llama-3.1-8B",
    curriculum_path: "curriculum.jsonl",
    adapter_name: "adapter-x",
  },
  metrics: {},
  created_at: "2026-02-01",
  started_at: "2026-02-01",
  completed_at: "2026-02-01",
  error: null,
};

describe("TrainingPanel wiring", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useFoundryStore.setState({
      trainingRuns: [],
      selectedRunId: null,
      error: null,
    });
    mockedTrainingAPI.listRuns.mockResolvedValue([]);
  });

  it("loads training runs on mount", async () => {
    mockedTrainingAPI.listRuns.mockResolvedValue([sampleRun]);
    render(<TrainingPanel />);
    await waitFor(() => expect(mockedTrainingAPI.listRuns).toHaveBeenCalled());
    expect(await screen.findByText("adapter-x")).toBeInTheDocument();
  });

  it("configures and starts a training run via the API", async () => {
    mockedTrainingAPI.listRuns
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce([sampleRun]);
    mockedTrainingAPI.configure.mockResolvedValue({
      run_id: "run-1",
      config: sampleRun.config,
    });
    mockedTrainingAPI.start.mockResolvedValue({
      run_id: "run-1",
      registry_run_id: "reg-1",
      status: "completed",
      adapter_path: null,
    });

    render(<TrainingPanel />);
    await waitFor(() => expect(mockedTrainingAPI.listRuns).toHaveBeenCalled());

    await userEvent.click(
      screen.getByRole("button", { name: /start training/i }),
    );

    await waitFor(() => {
      expect(mockedTrainingAPI.configure).toHaveBeenCalledWith(
        expect.objectContaining({
          base_model: "meta-llama/Llama-3.1-8B",
          curriculum_path: "curriculum.jsonl",
        }),
      );
      expect(mockedTrainingAPI.start).toHaveBeenCalledWith("run-1");
    });
    expect(await screen.findByText("adapter-x")).toBeInTheDocument();
  });
});

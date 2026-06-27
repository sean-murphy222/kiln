import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, beforeEach, vi } from "vitest";
import { EvaluationPanel } from "../EvaluationPanel";
import { useFoundryStore } from "@/store/useFoundryStore";
import { evaluationAPI } from "@/api/foundry";

vi.mock("@/api/foundry", () => ({
  evaluationAPI: {
    run: vi.fn(),
  },
}));

const mockedEvaluationAPI = vi.mocked(evaluationAPI);

describe("EvaluationPanel wiring", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useFoundryStore.setState({
      evaluations: [],
      selectedEvalId: null,
      error: null,
      trainingRuns: [
        {
          id: "run-1",
          name: "adapter-x",
          status: "completed",
          progress: 100,
          base_model: "llama",
          metrics: {},
          created_at: "2026-02-01",
          completed_at: "2026-02-01",
          error: null,
        },
      ],
    });
  });

  it("runs an evaluation via the API and renders the result", async () => {
    mockedEvaluationAPI.run.mockResolvedValue({
      id: "eval-1",
      training_run_id: "run-1",
      model_name: "adapter-x",
      overall_score: 0.8,
      overall_correct: 8,
      overall_total: 10,
      competency_scores: [
        {
          competency_name: "Procedural Comprehension",
          correct: 8,
          total: 10,
          score: 0.8,
          rating: "strong",
        },
      ],
      created_at: "2026-02-02",
    });

    render(<EvaluationPanel />);

    await userEvent.selectOptions(screen.getByRole("combobox"), "run-1");
    await userEvent.click(screen.getByRole("button", { name: /evaluate/i }));

    await waitFor(() => {
      expect(mockedEvaluationAPI.run).toHaveBeenCalledWith({
        training_run_id: "run-1",
      });
    });
    expect(
      await screen.findByText("Procedural Comprehension"),
    ).toBeInTheDocument();
  });
});

import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, beforeEach, vi } from "vitest";
import { DiagnosticsPanel } from "../DiagnosticsPanel";
import { useFoundryStore } from "@/store/useFoundryStore";
import { diagnosticsAPI, regressionAPI, mergingAPI } from "@/api/foundry";

vi.mock("@/api/foundry", () => ({
  diagnosticsAPI: { analyze: vi.fn() },
  regressionAPI: { listVersions: vi.fn() },
  mergingAPI: { listRegistry: vi.fn(), merge: vi.fn() },
}));

const mockedDiagnosticsAPI = vi.mocked(diagnosticsAPI);
const mockedRegressionAPI = vi.mocked(regressionAPI);
const mockedMergingAPI = vi.mocked(mergingAPI);

describe("DiagnosticsPanel wiring", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useFoundryStore.setState({
      diagnosticReport: null,
      modelVersions: [],
      mergeResults: [],
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
    mockedRegressionAPI.listVersions.mockResolvedValue([]);
    mockedMergingAPI.listRegistry.mockResolvedValue([]);
  });

  it("loads versions and merge registry on mount", async () => {
    render(<DiagnosticsPanel />);
    await waitFor(() => {
      expect(mockedRegressionAPI.listVersions).toHaveBeenCalled();
      expect(mockedMergingAPI.listRegistry).toHaveBeenCalled();
    });
  });

  it("analyzes a training run via the API and renders the report", async () => {
    mockedDiagnosticsAPI.analyze.mockResolvedValue({
      run_id: "run-1",
      issues: [
        {
          type: "Learning Rate",
          severity: "medium",
          description: "Loss oscillation detected.",
          recommendation: "Lower the learning rate.",
        },
      ],
      convergence_status: "converging",
      overfit_risk: "low",
      analyzed_at: "2026-02-02",
    });

    render(<DiagnosticsPanel />);

    await userEvent.selectOptions(screen.getByRole("combobox"), "run-1");
    await userEvent.click(screen.getByRole("button", { name: /analyze/i }));

    await waitFor(() => {
      expect(mockedDiagnosticsAPI.analyze).toHaveBeenCalledWith("run-1");
    });
    expect(
      await screen.findByText(/Loss oscillation detected/i),
    ).toBeInTheDocument();
  });
});

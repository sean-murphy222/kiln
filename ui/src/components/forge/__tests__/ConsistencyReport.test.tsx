import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, beforeEach, vi } from "vitest";
import { ConsistencyReport } from "../ConsistencyReport";
import { useForgeStore } from "@/store/useForgeStore";
import { consistencyAPI } from "@/api/forge";

vi.mock("@/api/forge", () => ({
  consistencyAPI: {
    check: vi.fn(),
  },
}));

const mockedConsistencyAPI = vi.mocked(consistencyAPI);

describe("ConsistencyReport wiring", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useForgeStore.setState({
      selectedDisciplineId: "disc-1",
      consistencyIssues: [],
      consistencyCheckedAt: null,
      error: null,
    });
  });

  it("runs a consistency check via the API and renders issues", async () => {
    mockedConsistencyAPI.check.mockResolvedValue({
      discipline_id: "disc-1",
      total_issues: 1,
      by_severity: { high: 1 },
      checked_at: "2026-02-01T00:00:00Z",
      issues: [
        {
          id: "issue-1",
          type: "Duplicate Content",
          severity: "high",
          description: "Contradictory training signals detected.",
          affected_example_ids: ["ex-1"],
          suggested_fix: "Merge the examples.",
        },
      ],
    });

    render(<ConsistencyReport />);
    await userEvent.click(screen.getByRole("button", { name: /run check/i }));

    await waitFor(() => {
      expect(mockedConsistencyAPI.check).toHaveBeenCalledWith("disc-1");
    });
    expect(
      await screen.findByText(/Contradictory training signals detected/i),
    ).toBeInTheDocument();
  });
});

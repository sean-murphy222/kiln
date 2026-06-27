import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, beforeEach, vi } from "vitest";
import { DisciplineList } from "../DisciplineList";
import { useForgeStore } from "@/store/useForgeStore";
import { disciplineAPI } from "@/api/forge";

vi.mock("@/api/forge", () => ({
  disciplineAPI: {
    list: vi.fn(),
    create: vi.fn(),
  },
}));

const mockedDisciplineAPI = vi.mocked(disciplineAPI);

describe("DisciplineList wiring", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useForgeStore.setState({
      disciplines: [],
      selectedDisciplineId: null,
      isLoading: false,
      error: null,
    });
    mockedDisciplineAPI.list.mockResolvedValue([
      {
        id: "disc-1",
        name: "Vehicle Maintenance",
        description: "",
        status: "active",
        created_by: "kiln-ui",
        vocabulary: [],
        document_types: [],
        created_at: "2026-01-01",
        updated_at: null,
      },
    ]);
  });

  it("loads disciplines from the backend on mount", async () => {
    render(<DisciplineList />);
    await waitFor(() => expect(mockedDisciplineAPI.list).toHaveBeenCalled());
    expect(await screen.findByText("Vehicle Maintenance")).toBeInTheDocument();
  });

  it("creates a discipline via the API", async () => {
    mockedDisciplineAPI.create.mockResolvedValue({
      id: "disc-2",
      name: "Comms",
      description: "",
      status: "draft",
      created_by: "kiln-ui",
      vocabulary: [],
      document_types: [],
      created_at: "2026-01-02",
      updated_at: null,
    });

    render(<DisciplineList />);
    await waitFor(() => expect(mockedDisciplineAPI.list).toHaveBeenCalled());

    await userEvent.click(screen.getByTitle("New discipline"));
    await userEvent.type(
      screen.getByPlaceholderText("Discipline name..."),
      "Comms",
    );
    await userEvent.click(screen.getByRole("button", { name: /create/i }));

    await waitFor(() => {
      expect(mockedDisciplineAPI.create).toHaveBeenCalledWith({
        name: "Comms",
        description: "",
      });
    });
    expect(useForgeStore.getState().selectedDisciplineId).toBe("disc-2");
  });
});

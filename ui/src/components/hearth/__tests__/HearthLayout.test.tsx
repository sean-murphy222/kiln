import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, beforeEach, vi } from "vitest";
import { HearthLayout } from "../HearthLayout";
import { useHearthStore } from "@/store/useHearthStore";
import { modelAPI, conversationAPI, queryAPI } from "@/api/hearth";

vi.mock("@/api/hearth", () => ({
  modelAPI: {
    list: vi.fn(),
    load: vi.fn(),
    unload: vi.fn(),
  },
  conversationAPI: {
    list: vi.fn(),
    delete: vi.fn(),
  },
  queryAPI: {
    send: vi.fn(),
  },
}));

const mockedModelAPI = vi.mocked(modelAPI);
const mockedConversationAPI = vi.mocked(conversationAPI);
const mockedQueryAPI = vi.mocked(queryAPI);

function resetStore() {
  useHearthStore.setState({
    conversations: [],
    activeConversationId: null,
    modelSlots: [],
    activeModelId: null,
    isStreaming: false,
    error: null,
  });
}

describe("HearthLayout wiring", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetStore();
    mockedModelAPI.list.mockResolvedValue([
      {
        id: "slot-1",
        name: "Maintenance v1",
        base_model: "llama-3.1-8b",
        adapter_path: "/a/b",
        status: "ready",
        loaded_at: null,
      },
    ]);
    mockedConversationAPI.list.mockResolvedValue([]);
  });

  it("loads models and conversations from the backend on mount", async () => {
    render(<HearthLayout />);

    await waitFor(() => {
      expect(mockedModelAPI.list).toHaveBeenCalledTimes(1);
      expect(mockedConversationAPI.list).toHaveBeenCalledTimes(1);
    });

    // Active model resolves to the ready slot.
    await waitFor(() => {
      expect(screen.getByText("Maintenance v1")).toBeInTheDocument();
    });
  });

  it("sends a query and renders the assistant response with citations", async () => {
    mockedQueryAPI.send.mockResolvedValue({
      conversation_id: "conv-real-1",
      processing_time_ms: 12,
      message: {
        id: "msg-assistant-1",
        role: "assistant",
        content: "Engine PMCS is described in TM-9 [1].",
        citations: [
          {
            document_id: "doc-1",
            document_title: "TM-9-2320-280-10",
            section: "2.3 Engine",
            page: 42,
            relevance_score: 0.91,
            snippet: "Perform PMCS at intervals.",
          },
        ],
        timestamp: new Date().toISOString(),
        model_id: "slot-1",
      },
    });

    render(<HearthLayout />);
    await waitFor(() => expect(mockedModelAPI.list).toHaveBeenCalled());

    const input = screen.getByPlaceholderText(
      /Ask a question about your documents/i,
    );
    await userEvent.type(input, "How do I PMCS the engine?{Enter}");

    await waitFor(() => {
      expect(mockedQueryAPI.send).toHaveBeenCalledWith(
        expect.objectContaining({ query: "How do I PMCS the engine?" }),
      );
    });

    // User message + assistant response are rendered (the user text also
    // appears as the conversation title, hence getAllByText).
    expect(
      screen.getAllByText("How do I PMCS the engine?").length,
    ).toBeGreaterThan(0);
    await waitFor(() => {
      expect(
        screen.getByText(/Engine PMCS is described in TM-9/i),
      ).toBeInTheDocument();
    });
  });
});

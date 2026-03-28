import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

import gomokuScene from "../../test/fixtures/gomoku.visual.json";
import type { VisualScene } from "../../gateway/types";
import type { ArenaSessionStore, ArenaSessionStoreSnapshot } from "../store/arenaSessionStore";
import { SessionPage } from "./SessionPage";

const {
  createArenaGatewayClientMock,
  createArenaSessionStoreMock,
  createArenaMediaResolverMock,
} = vi.hoisted(() => ({
  createArenaGatewayClientMock: vi.fn(),
  createArenaSessionStoreMock: vi.fn(),
  createArenaMediaResolverMock: vi.fn(),
}));

vi.mock("../../gateway/client", () => ({
  createArenaGatewayClient: createArenaGatewayClientMock,
}));

vi.mock("../store/arenaSessionStore", () => ({
  createArenaSessionStore: createArenaSessionStoreMock,
}));

vi.mock("../../gateway/media", () => ({
  createArenaMediaResolver: createArenaMediaResolverMock,
}));

describe("SessionPage", () => {
  beforeEach(() => {
    createArenaGatewayClientMock.mockReset();
    createArenaSessionStoreMock.mockReset();
    createArenaMediaResolverMock.mockReset();
  });

  it("routes host controls side-panel actions and plugin input through the session store", async () => {
    const setObserver = vi.fn().mockResolvedValue(undefined);
    const submitAction = vi.fn().mockResolvedValue({
      intentId: "intent-1",
      state: "accepted",
    });
    const submitControl = vi.fn().mockResolvedValue({
      intentId: "control-1",
      state: "accepted",
    });
    const submitChat = vi.fn().mockResolvedValue({
      intentId: "chat-1",
      state: "accepted",
    });
    const loadSession = vi.fn().mockResolvedValue(undefined);

    const snapshot: ArenaSessionStoreSnapshot = {
      status: "ready",
      sessionRequest: {
        sessionId: "sample-1",
      },
      session: {
        sessionId: "sample-1",
        gameId: "gomoku",
        pluginId: "arena.visualization.gomoku.board_v1",
        lifecycle: "live_running",
        playback: {
          mode: "paused",
          cursorTs: 1005,
          cursorEventSeq: 5,
          speed: 1,
          canSeek: true,
        },
        observer: {
          observerId: "Black",
          observerKind: "player",
        },
        scheduling: {
          family: "turn",
          phase: "waiting_for_intent",
          acceptsHumanIntent: true,
          activeActorId: "White",
        },
        capabilities: {
          observerModes: ["global", "player"],
        },
        summary: {},
        timeline: {},
      },
      sceneStatus: "ready",
      scene: gomokuScene as VisualScene,
      currentSceneSeq: 5,
      timeline: {
        status: "ready",
        events: [],
        nextAfterSeq: null,
        hasMore: false,
        limit: 50,
        filters: {
          eventTypes: [],
          severity: "all",
          humanIntentOnly: false,
        },
      },
      latestActionReceipt: undefined,
      error: undefined,
    };

    const store: ArenaSessionStore = {
      getSnapshot: () => snapshot,
      subscribe: () => () => {},
      loadSession,
      loadMoreTimeline: vi.fn().mockResolvedValue(undefined),
      loadScene: vi.fn().mockResolvedValue(undefined),
      setCurrentSceneSeq: vi.fn(),
      setPlaybackMode: vi.fn(),
      setTimelineFilters: vi.fn(),
      setObserver,
      submitControl,
      submitAction,
      submitChat,
      clearLatestActionReceipt: vi.fn(),
    };

    createArenaGatewayClientMock.mockReturnValue({});
    createArenaSessionStoreMock.mockReturnValue(store);
    createArenaMediaResolverMock.mockReturnValue({
      subscribe: vi.fn(),
    });

    render(
      <MemoryRouter initialEntries={["/sessions/sample-1"]}>
        <Routes>
          <Route path="/sessions/:sessionId" element={<SessionPage />} />
        </Routes>
      </MemoryRouter>,
    );

    await waitFor(() => {
      expect(loadSession).toHaveBeenCalledWith({ sessionId: "sample-1" });
    });

    fireEvent.click(screen.getByRole("button", { name: /live tail/i }));
    fireEvent.click(screen.getByRole("button", { name: /replay/i }));
    fireEvent.click(screen.getByRole("button", { name: "2x" }));
    fireEvent.click(screen.getByRole("button", { name: /step \+1/i }));
    fireEvent.click(screen.getByRole("button", { name: /^end$/i }));
    fireEvent.click(screen.getByRole("button", { name: /back to tail/i }));

    expect(submitControl).toHaveBeenNthCalledWith(1, {
      commandType: "follow_tail",
    });
    expect(submitControl).toHaveBeenNthCalledWith(2, {
      commandType: "replay",
    });
    expect(submitControl).toHaveBeenNthCalledWith(3, {
      commandType: "set_speed",
      speed: 2,
    });
    expect(submitControl).toHaveBeenNthCalledWith(4, {
      commandType: "step",
      stepDelta: 1,
    });
    expect(submitControl).toHaveBeenNthCalledWith(5, {
      commandType: "seek_end",
    });
    expect(submitControl).toHaveBeenNthCalledWith(6, {
      commandType: "back_to_tail",
    });

    fireEvent.change(screen.getByLabelText(/observer view/i), {
      target: { value: "global" },
    });

    await waitFor(() => {
      expect(setObserver).toHaveBeenCalledWith({
        observerId: "",
        observerKind: "global",
      });
    });

    fireEvent.click(screen.getByTestId("board-cell-B1"));

    await waitFor(() => {
      expect(submitAction).toHaveBeenCalledWith({
        playerId: "Black",
        action: { move: "B1" },
      });
    });

    fireEvent.click(screen.getByRole("tab", { name: "Chat" }));
    fireEvent.change(screen.getByLabelText(/chat message/i), {
      target: { value: "hello host" },
    });
    fireEvent.click(screen.getByRole("button", { name: /send chat/i }));

    await waitFor(() => {
      expect(submitChat).toHaveBeenCalledWith({
        playerId: "Black",
        text: "hello host",
      });
    });
  });
});

import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

import doudizhuScene from "../../test/fixtures/doudizhu.visual.json";
import type { VisualScene } from "../../gateway/types";
import type { ArenaSessionStore, ArenaSessionStoreSnapshot } from "../store/arenaSessionStore";
import { SessionPage } from "./SessionPage";

const {
  createArenaGatewayClientMock,
  createArenaSessionStoreMock,
  createArenaMediaResolverMock,
  resolveArenaPluginMock,
} = vi.hoisted(() => ({
  createArenaGatewayClientMock: vi.fn(),
  createArenaSessionStoreMock: vi.fn(),
  createArenaMediaResolverMock: vi.fn(),
  resolveArenaPluginMock: vi.fn(),
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

vi.mock("../../plugins/registry", () => ({
  resolveArenaPlugin: resolveArenaPluginMock,
}));

describe("SessionPage", () => {
  beforeEach(() => {
    createArenaGatewayClientMock.mockReset();
    createArenaSessionStoreMock.mockReset();
    createArenaMediaResolverMock.mockReset();
    resolveArenaPluginMock.mockReset();
  });

  it("routes observer changes and chat submission through the host store", async () => {
    const setObserver = vi.fn().mockResolvedValue(undefined);
    const submitChat = vi.fn().mockResolvedValue({
      intentId: "chat-1",
      state: "accepted",
    });

    const snapshot: ArenaSessionStoreSnapshot = {
      status: "ready",
      sessionRequest: {
        sessionId: "sample-1",
      },
      session: {
        sessionId: "sample-1",
        gameId: "doudizhu",
        pluginId: "arena.visualization.doudizhu.table_v1",
        lifecycle: "live_running",
        playback: {
          mode: "paused",
          cursorTs: 1007,
          cursorEventSeq: 7,
          speed: 1,
          canSeek: true,
        },
        observer: {
          observerId: "player_0",
          observerKind: "player",
        },
        scheduling: {
          family: "turn",
          phase: "waiting_for_intent",
          acceptsHumanIntent: true,
          activeActorId: "player_0",
        },
        capabilities: {
          observerModes: ["global", "player"],
        },
        summary: {},
        timeline: {},
      },
      sceneStatus: "ready",
      scene: doudizhuScene as VisualScene,
      currentSceneSeq: 7,
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
      loadSession: vi.fn().mockResolvedValue(undefined),
      loadMoreTimeline: vi.fn().mockResolvedValue(undefined),
      loadScene: vi.fn().mockResolvedValue(undefined),
      setCurrentSceneSeq: vi.fn(),
      setPlaybackMode: vi.fn(),
      setTimelineFilters: vi.fn(),
      setObserver,
      submitControl: vi.fn().mockResolvedValue({
        intentId: "control-1",
        state: "accepted",
      }),
      submitAction: vi.fn().mockResolvedValue({
        intentId: "intent-1",
        state: "accepted",
      }),
      submitChat,
      clearLatestActionReceipt: vi.fn(),
    };

    createArenaGatewayClientMock.mockReturnValue({});
    createArenaSessionStoreMock.mockReturnValue(store);
    createArenaMediaResolverMock.mockReturnValue({
      subscribe: vi.fn(),
    });
    resolveArenaPluginMock.mockReturnValue(undefined);

    render(
      <MemoryRouter initialEntries={["/sessions/sample-1"]}>
        <Routes>
          <Route path="/sessions/:sessionId" element={<SessionPage />} />
        </Routes>
      </MemoryRouter>,
    );

    fireEvent.change(screen.getByLabelText(/observer view/i), {
      target: { value: "player:player_1" },
    });

    await waitFor(() => {
      expect(setObserver).toHaveBeenCalledWith({
        observerId: "player_1",
        observerKind: "player",
      });
    });

    fireEvent.click(screen.getByRole("tab", { name: "Chat" }));
    fireEvent.change(screen.getByLabelText(/chat message/i), {
      target: { value: "hello host" },
    });
    fireEvent.click(screen.getByRole("button", { name: /send chat/i }));

    await waitFor(() => {
      expect(submitChat).toHaveBeenCalledWith({
        playerId: "player_0",
        text: "hello host",
      });
    });
  });
});

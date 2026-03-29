import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

import gomokuScene from "../test/fixtures/gomoku.visual.json";
import type { VisualScene } from "../gateway/types";
import type {
  ArenaSessionStore,
  ArenaSessionStoreSnapshot,
} from "./store/arenaSessionStore";
import { App } from "./App";

const {
  createArenaGatewayClientMock,
  createArenaSessionStoreMock,
  createArenaMediaResolverMock,
} = vi.hoisted(() => ({
  createArenaGatewayClientMock: vi.fn(),
  createArenaSessionStoreMock: vi.fn(),
  createArenaMediaResolverMock: vi.fn(),
}));

vi.mock("../gateway/client", () => ({
  createArenaGatewayClient: createArenaGatewayClientMock,
}));

vi.mock("./store/arenaSessionStore", () => ({
  createArenaSessionStore: createArenaSessionStoreMock,
}));

vi.mock("../gateway/media", () => ({
  createArenaMediaResolver: createArenaMediaResolverMock,
}));

function buildReadySnapshot(): ArenaSessionStoreSnapshot {
  return {
    status: "ready",
    sessionRequest: {
      sessionId: "demo-session",
    },
    session: {
      sessionId: "demo-session",
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
}

function createStoreMock(
  snapshot: ArenaSessionStoreSnapshot = buildReadySnapshot(),
): ArenaSessionStore {
  return {
    getSnapshot: () => snapshot,
    subscribe: () => () => {},
    loadSession: vi.fn().mockResolvedValue(undefined),
    loadMoreTimeline: vi.fn().mockResolvedValue(undefined),
    loadScene: vi.fn().mockResolvedValue(undefined),
    setCurrentSceneSeq: vi.fn(),
    setPlaybackMode: vi.fn(),
    setTimelineFilters: vi.fn(),
    setObserver: vi.fn().mockResolvedValue(undefined),
    submitControl: vi.fn().mockResolvedValue({
      intentId: "control-1",
      state: "accepted",
    }),
    submitAction: vi.fn().mockResolvedValue({
      intentId: "intent-1",
      state: "accepted",
    }),
    submitChat: vi.fn().mockResolvedValue({
      intentId: "chat-1",
      state: "accepted",
    }),
    clearLatestActionReceipt: vi.fn(),
  };
}

describe("App", () => {
  beforeEach(() => {
    createArenaGatewayClientMock.mockReset();
    createArenaSessionStoreMock.mockReset();
    createArenaMediaResolverMock.mockReset();
    createArenaGatewayClientMock.mockReturnValue({});
    createArenaMediaResolverMock.mockReturnValue({
      subscribe: vi.fn(),
    });
  });

  it("renders the empty host route", () => {
    createArenaSessionStoreMock.mockReturnValue(createStoreMock());

    render(
      <MemoryRouter initialEntries={["/"]}>
        <App />
      </MemoryRouter>,
    );

    expect(
      screen.getByRole("heading", { name: /arena visual host/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/open a live or recorded session to inspect timelines/i),
    ).toBeInTheDocument();
  });

  it("navigates through the host shell into the real session workspace workflow", async () => {
    const store = createStoreMock();
    createArenaSessionStoreMock.mockReturnValue(store);

    render(
      <MemoryRouter initialEntries={["/"]}>
        <App />
      </MemoryRouter>,
    );

    fireEvent.click(screen.getByRole("link", { name: /demo session/i }));

    await waitFor(() => {
      expect(store.loadSession).toHaveBeenCalledWith({ sessionId: "demo-session" });
    });

    expect(document.querySelector(".app-shell")).toHaveClass("app-shell--session");

    expect(screen.getByRole("heading", { name: /demo-session/i })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /step \+1/i }));

    await waitFor(() => {
      expect(store.submitControl).toHaveBeenNthCalledWith(1, {
        commandType: "step",
        stepDelta: 1,
      });
    });

    fireEvent.click(screen.getByRole("button", { name: /live tail/i }));
    fireEvent.change(screen.getByLabelText(/observer view/i), {
      target: { value: "global" },
    });
    fireEvent.click(screen.getByTestId("board-cell-B1"));
    fireEvent.click(screen.getByRole("tab", { name: "Chat" }));
    fireEvent.change(screen.getByLabelText(/chat message/i), {
      target: { value: "hello from app" },
    });
    fireEvent.click(screen.getByRole("button", { name: /send chat/i }));

    await waitFor(() => {
      expect(store.submitControl).toHaveBeenNthCalledWith(2, {
        commandType: "follow_tail",
      });
      expect(store.setObserver).toHaveBeenCalledWith({
        observerId: "",
        observerKind: "global",
      });
      expect(store.submitAction).toHaveBeenCalledWith({
        playerId: "Black",
        action: { move: "B1" },
      });
      expect(store.submitChat).toHaveBeenCalledWith({
        playerId: "Black",
        text: "hello from app",
      });
    });
  });

  it("redirects unknown routes back to the host home", () => {
    createArenaSessionStoreMock.mockReturnValue(createStoreMock());

    render(
      <MemoryRouter initialEntries={["/missing-route"]}>
        <App />
      </MemoryRouter>,
    );

    expect(
      screen.getByRole("heading", { name: /arena visual host/i }),
    ).toBeInTheDocument();
  });
});

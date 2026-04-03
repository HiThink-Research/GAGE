import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import gomokuScene from "../../test/fixtures/gomoku.visual.json";
import doudizhuScene from "../../test/fixtures/doudizhu.visual.json";
import retroScene from "../../test/fixtures/retro-mario.visual.json";
import type { VisualScene } from "../../gateway/types";
import type { ArenaSessionStore, ArenaSessionStoreSnapshot } from "../store/arenaSessionStore";
import { SessionPage } from "./SessionPage";

const FRAME_DATA_URL =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII=";

const {
  createArenaGatewayClientMock,
  createArenaSessionStoreMock,
  createArenaMediaResolverMock,
  createArenaLiveUpdateStreamMock,
} = vi.hoisted(() => ({
  createArenaGatewayClientMock: vi.fn(),
  createArenaSessionStoreMock: vi.fn(),
  createArenaMediaResolverMock: vi.fn(),
  createArenaLiveUpdateStreamMock: vi.fn(),
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

vi.mock("../../gateway/liveUpdateStream", () => ({
  createArenaLiveUpdateStream: createArenaLiveUpdateStreamMock,
}));

describe("SessionPage", () => {
  beforeEach(() => {
    createArenaGatewayClientMock.mockReset();
    createArenaSessionStoreMock.mockReset();
    createArenaMediaResolverMock.mockReset();
    createArenaLiveUpdateStreamMock.mockReset();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  function createMutableStore(initialSnapshot: ArenaSessionStoreSnapshot): {
    setSnapshot: (nextSnapshot: ArenaSessionStoreSnapshot) => void;
    store: ArenaSessionStore;
  } {
    let snapshot = initialSnapshot;
    const listeners = new Set<() => void>();

    return {
      setSnapshot(nextSnapshot) {
        snapshot = nextSnapshot;
        for (const listener of listeners) {
          listener();
        }
      },
      store: {
        getSnapshot: () => snapshot,
        subscribe: (listener: () => void) => {
          listeners.add(listener);
          return () => {
            listeners.delete(listener);
          };
        },
        loadSession: vi.fn().mockResolvedValue(undefined),
        refreshSession: vi.fn().mockResolvedValue(undefined),
        loadMoreTimeline: vi.fn().mockResolvedValue(undefined),
        loadScene: vi.fn().mockResolvedValue(undefined),
        advanceReplayPlayback: vi.fn(),
        setCurrentSceneSeq: vi.fn(),
        setPlaybackMode: vi.fn(),
        setTimelineFilters: vi.fn(),
        setObserver: vi.fn().mockResolvedValue(undefined),
        submitControl: vi.fn().mockResolvedValue(undefined),
        submitAction: vi.fn().mockResolvedValue(undefined),
        submitChat: vi.fn().mockResolvedValue(undefined),
        clearLatestActionReceipt: vi.fn(),
      } as unknown as ArenaSessionStore,
    };
  }

  function expandSessionControls(): void {
    const expandButton = screen.queryByRole("button", { name: /expand session controls/i });
    if (expandButton) {
      fireEvent.click(expandButton);
    }
  }

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
      submitActionLowLatency: vi.fn().mockResolvedValue(undefined),
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

    expect(screen.getByRole("button", { name: /expand session controls/i })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /live tail/i })).not.toBeInTheDocument();

    expandSessionControls();

    expect(screen.getByRole("button", { name: /hide timeline drawer/i })).toBeInTheDocument();
    const utilityRail = screen.getByRole("navigation", { name: /session utility rail/i });
    expect(utilityRail).toBeInTheDocument();
    expect(within(utilityRail).getByRole("button", { name: "Control" })).toBeInTheDocument();
    expect(within(utilityRail).getByRole("button", { name: "Players" })).toBeInTheDocument();
    expect(within(utilityRail).getByRole("button", { name: "Chat" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /live tail/i }));
    await waitFor(() => {
      expect(submitControl).toHaveBeenNthCalledWith(1, {
        commandType: "follow_tail",
      });
    });

    fireEvent.click(screen.getByRole("button", { name: /replay/i }));
    await waitFor(() => {
      expect(submitControl).toHaveBeenNthCalledWith(2, {
        commandType: "replay",
      });
    });

    fireEvent.click(screen.getByRole("button", { name: "2x" }));
    await waitFor(() => {
      expect(submitControl).toHaveBeenNthCalledWith(3, {
        commandType: "set_speed",
        speed: 2,
      });
    });

    fireEvent.click(screen.getByRole("button", { name: /step \+1/i }));
    await waitFor(() => {
      expect(submitControl).toHaveBeenNthCalledWith(4, {
        commandType: "step",
        stepDelta: 1,
      });
    });

    fireEvent.click(screen.getByRole("button", { name: /^end$/i }));
    await waitFor(() => {
      expect(submitControl).toHaveBeenNthCalledWith(5, {
        commandType: "seek_end",
      });
    });

    fireEvent.click(screen.getByRole("button", { name: /back to tail/i }));
    await waitFor(() => {
      expect(submitControl).toHaveBeenNthCalledWith(6, {
        commandType: "back_to_tail",
      });
    });

    fireEvent.click(within(utilityRail).getByRole("button", { name: "Players" }));

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

    fireEvent.click(within(utilityRail).getByRole("button", { name: "Chat" }));
    expect(screen.getByRole("button", { name: /close utility drawer/i })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Chat" })).toHaveAttribute("aria-selected", "true");
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

  it("collapses the session command deck on entry and expands on demand", () => {
    const snapshot: ArenaSessionStoreSnapshot = {
      status: "ready",
      sessionRequest: {
        sessionId: "sample-collapsed",
      },
      session: {
        sessionId: "sample-collapsed",
        gameId: "gomoku",
        pluginId: "arena.visualization.gomoku.board_v1",
        lifecycle: "live_running",
        playback: {
          mode: "live_tail",
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
          activeActorId: "Black",
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

    createArenaGatewayClientMock.mockReturnValue({});
    createArenaSessionStoreMock.mockReturnValue({
      getSnapshot: () => snapshot,
      subscribe: () => () => {},
      loadSession: vi.fn().mockResolvedValue(undefined),
      refreshSession: vi.fn().mockResolvedValue(undefined),
      loadMoreTimeline: vi.fn().mockResolvedValue(undefined),
      loadScene: vi.fn().mockResolvedValue(undefined),
      advanceReplayPlayback: vi.fn(),
      setCurrentSceneSeq: vi.fn(),
      setPlaybackMode: vi.fn(),
      setTimelineFilters: vi.fn(),
      setObserver: vi.fn().mockResolvedValue(undefined),
      submitControl: vi.fn().mockResolvedValue(undefined),
      submitAction: vi.fn().mockResolvedValue(undefined),
      submitChat: vi.fn().mockResolvedValue(undefined),
      clearLatestActionReceipt: vi.fn(),
    } as unknown as ArenaSessionStore);
    createArenaMediaResolverMock.mockReturnValue({
      subscribe: vi.fn(),
    });

    render(
      <MemoryRouter initialEntries={["/sessions/sample-collapsed"]}>
        <Routes>
          <Route path="/sessions/:sessionId" element={<SessionPage />} />
        </Routes>
      </MemoryRouter>,
    );

    expect(screen.getByRole("button", { name: /expand session controls/i })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /live tail/i })).not.toBeInTheDocument();

    expandSessionControls();

    expect(screen.getByRole("button", { name: /collapse session controls/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /live tail/i })).toBeInTheDocument();
  });

  it("shows latest timeline events in the collapsed rail before the drawer is expanded", async () => {
    const snapshot: ArenaSessionStoreSnapshot = {
      status: "ready",
      sessionRequest: {
        sessionId: "sample-live",
      },
      session: {
        sessionId: "sample-live",
        gameId: "gomoku",
        pluginId: "arena.visualization.gomoku.board_v1",
        lifecycle: "live_running",
        playback: {
          mode: "live_tail",
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
        events: [
          {
            seq: 3,
            tsMs: 1003,
            type: "snapshot",
            label: "Scene snapshot committed",
            severity: "info",
          },
          {
            seq: 4,
            tsMs: 1004,
            type: "action_intent",
            label: "Human move window open",
            severity: "warn",
            tags: ["human_intent"],
          },
          {
            seq: 5,
            tsMs: 1005,
            type: "system_marker",
            label: "Latency spike detected",
            severity: "critical",
          },
        ],
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

    createArenaGatewayClientMock.mockReturnValue({});
    createArenaSessionStoreMock.mockReturnValue({
      getSnapshot: () => snapshot,
      subscribe: () => () => {},
      loadSession: vi.fn().mockResolvedValue(undefined),
      refreshSession: vi.fn().mockResolvedValue(undefined),
      loadMoreTimeline: vi.fn().mockResolvedValue(undefined),
      loadScene: vi.fn().mockResolvedValue(undefined),
      advanceReplayPlayback: vi.fn(),
      setCurrentSceneSeq: vi.fn(),
      setPlaybackMode: vi.fn(),
      setTimelineFilters: vi.fn(),
      setObserver: vi.fn().mockResolvedValue(undefined),
      submitControl: vi.fn().mockResolvedValue(undefined),
      submitAction: vi.fn().mockResolvedValue(undefined),
      submitChat: vi.fn().mockResolvedValue(undefined),
      clearLatestActionReceipt: vi.fn(),
    } as unknown as ArenaSessionStore);
    createArenaMediaResolverMock.mockReturnValue({
      subscribe: vi.fn(),
    });

    render(
      <MemoryRouter initialEntries={["/sessions/sample-live"]}>
        <Routes>
          <Route path="/sessions/:sessionId" element={<SessionPage />} />
        </Routes>
      </MemoryRouter>,
    );

    expect(screen.getByText("Human move window open")).toBeInTheDocument();
    expect(screen.getByText("Latency spike detected")).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: /event stream/i })).not.toBeInTheDocument();
  });

  it("surfaces live human-input readiness cues inside the stage hud", async () => {
    const snapshot: ArenaSessionStoreSnapshot = {
      status: "ready",
      sessionRequest: {
        sessionId: "sample-human-live",
      },
      session: {
        sessionId: "sample-human-live",
        gameId: "gomoku",
        pluginId: "arena.visualization.gomoku.board_v1",
        lifecycle: "live_running",
        playback: {
          mode: "live_tail",
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
          supportsLowLatencyRealtimeInput: true,
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

    createArenaGatewayClientMock.mockReturnValue({});
    createArenaSessionStoreMock.mockReturnValue({
      getSnapshot: () => snapshot,
      subscribe: () => () => {},
      loadSession: vi.fn().mockResolvedValue(undefined),
      refreshSession: vi.fn().mockResolvedValue(undefined),
      loadMoreTimeline: vi.fn().mockResolvedValue(undefined),
      loadScene: vi.fn().mockResolvedValue(undefined),
      advanceReplayPlayback: vi.fn(),
      setCurrentSceneSeq: vi.fn(),
      setPlaybackMode: vi.fn(),
      setTimelineFilters: vi.fn(),
      setObserver: vi.fn().mockResolvedValue(undefined),
      submitControl: vi.fn().mockResolvedValue(undefined),
      submitAction: vi.fn().mockResolvedValue(undefined),
      submitActionLowLatency: vi.fn().mockResolvedValue(undefined),
      submitChat: vi.fn().mockResolvedValue(undefined),
      clearLatestActionReceipt: vi.fn(),
    } as unknown as ArenaSessionStore);
    createArenaMediaResolverMock.mockReturnValue({
      subscribe: vi.fn(),
    });

    render(
      <MemoryRouter initialEntries={["/sessions/sample-human-live"]}>
        <Routes>
          <Route path="/sessions/:sessionId" element={<SessionPage />} />
        </Routes>
      </MemoryRouter>,
    );

    expect(screen.getByTestId("session-brand")).toHaveTextContent("GAGE");
    expect(screen.getByTestId("session-brand")).toHaveTextContent("GAME ARENA");
    const utilityRail = screen.getByRole("navigation", { name: /session utility rail/i });
    expect(within(utilityRail).getByRole("button", { name: "Control" })).toBeInTheDocument();
    fireEvent.click(within(utilityRail).getByRole("button", { name: "Control" }));
    expect(screen.getByRole("tab", { name: "Control" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText("Tail locked")).toBeInTheDocument();
    expect(screen.getByText("Human input enabled")).toBeInTheDocument();
    expect(screen.getByText("Low latency input")).toBeInTheDocument();
  });

  it("toggles fullscreen mode for the game stage", async () => {
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
          activeActorId: "Black",
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

    createArenaGatewayClientMock.mockReturnValue({});
    createArenaSessionStoreMock.mockReturnValue({
      getSnapshot: () => snapshot,
      subscribe: () => () => {},
      loadSession: vi.fn().mockResolvedValue(undefined),
      refreshSession: vi.fn().mockResolvedValue(undefined),
      loadMoreTimeline: vi.fn().mockResolvedValue(undefined),
      loadScene: vi.fn().mockResolvedValue(undefined),
      advanceReplayPlayback: vi.fn(),
      setCurrentSceneSeq: vi.fn(),
      setPlaybackMode: vi.fn(),
      setTimelineFilters: vi.fn(),
      setObserver: vi.fn().mockResolvedValue(undefined),
      submitControl: vi.fn().mockResolvedValue(undefined),
      submitAction: vi.fn().mockResolvedValue(undefined),
      submitChat: vi.fn().mockResolvedValue(undefined),
      clearLatestActionReceipt: vi.fn(),
    } as unknown as ArenaSessionStore);
    createArenaMediaResolverMock.mockReturnValue({
      subscribe: vi.fn(),
    });

    let fullscreenElement: Element | null = null;
    const requestFullscreen = vi.fn(async function (this: HTMLElement) {
      fullscreenElement = this;
      document.dispatchEvent(new Event("fullscreenchange"));
    });
    const exitFullscreen = vi.fn(async () => {
      fullscreenElement = null;
      document.dispatchEvent(new Event("fullscreenchange"));
    });
    const originalFullscreenElement = Object.getOwnPropertyDescriptor(
      document,
      "fullscreenElement",
    );
    const originalRequestFullscreen = Object.getOwnPropertyDescriptor(
      HTMLElement.prototype,
      "requestFullscreen",
    );
    const originalExitFullscreen = Object.getOwnPropertyDescriptor(
      document,
      "exitFullscreen",
    );

    Object.defineProperty(document, "fullscreenElement", {
      configurable: true,
      get: () => fullscreenElement,
    });
    Object.defineProperty(HTMLElement.prototype, "requestFullscreen", {
      configurable: true,
      value: requestFullscreen,
    });
    Object.defineProperty(document, "exitFullscreen", {
      configurable: true,
      value: exitFullscreen,
    });

    try {
      render(
        <MemoryRouter initialEntries={["/sessions/sample-1"]}>
          <Routes>
            <Route path="/sessions/:sessionId" element={<SessionPage />} />
          </Routes>
        </MemoryRouter>,
      );

      expect(
        screen.getByRole("button", { name: /^enter fullscreen$/i }),
      ).toBeInTheDocument();

      fireEvent.doubleClick(screen.getByTestId("session-stage"));
      await waitFor(() => {
        expect(requestFullscreen).toHaveBeenCalled();
        expect(screen.getByRole("button", { name: /exit fullscreen/i })).toBeInTheDocument();
      });

      fireEvent.click(screen.getByRole("button", { name: /exit fullscreen/i }));
      await waitFor(() => {
        expect(exitFullscreen).toHaveBeenCalled();
        expect(screen.getByRole("button", { name: /^enter fullscreen$/i })).toBeInTheDocument();
      });
    } finally {
      if (originalFullscreenElement) {
        Object.defineProperty(document, "fullscreenElement", originalFullscreenElement);
      } else {
        delete (document as { fullscreenElement?: unknown }).fullscreenElement;
      }
      if (originalRequestFullscreen) {
        Object.defineProperty(HTMLElement.prototype, "requestFullscreen", originalRequestFullscreen);
      } else {
        delete (HTMLElement.prototype as { requestFullscreen?: unknown }).requestFullscreen;
      }
      if (originalExitFullscreen) {
        Object.defineProperty(document, "exitFullscreen", originalExitFullscreen);
      } else {
        delete (document as { exitFullscreen?: unknown }).exitFullscreen;
      }
    }
  });

  it("hides the stage chrome for immersive Mario once fullscreen is active", async () => {
    const snapshot: ArenaSessionStoreSnapshot = {
      status: "ready",
      sessionRequest: {
        sessionId: "sample-retro",
      },
      session: {
        sessionId: "sample-retro",
        gameId: "retro_platformer",
        pluginId: "arena.visualization.retro_platformer.frame_v1",
        lifecycle: "live_running",
        playback: {
          mode: "live_tail",
          cursorTs: 4023,
          cursorEventSeq: 23,
          speed: 1,
          canSeek: true,
        },
        observer: {
          observerId: "player_0",
          observerKind: "player",
        },
        scheduling: {
          family: "real_time_tick",
          phase: "waiting_for_intent",
          acceptsHumanIntent: true,
          activeActorId: "player_0",
        },
        capabilities: {
          observerModes: ["player", "camera"],
        },
        summary: {},
        timeline: {},
      },
      sceneStatus: "ready",
      scene: retroScene as VisualScene,
      currentSceneSeq: 23,
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

    createArenaGatewayClientMock.mockReturnValue({});
    createArenaSessionStoreMock.mockReturnValue({
      getSnapshot: () => snapshot,
      subscribe: () => () => {},
      loadSession: vi.fn().mockResolvedValue(undefined),
      refreshSession: vi.fn().mockResolvedValue(undefined),
      loadMoreTimeline: vi.fn().mockResolvedValue(undefined),
      loadScene: vi.fn().mockResolvedValue(undefined),
      advanceReplayPlayback: vi.fn(),
      setCurrentSceneSeq: vi.fn(),
      setPlaybackMode: vi.fn(),
      setTimelineFilters: vi.fn(),
      setObserver: vi.fn().mockResolvedValue(undefined),
      submitControl: vi.fn().mockResolvedValue(undefined),
      submitAction: vi.fn().mockResolvedValue(undefined),
      submitChat: vi.fn().mockResolvedValue(undefined),
      clearLatestActionReceipt: vi.fn(),
    } as unknown as ArenaSessionStore);
    createArenaMediaResolverMock.mockReturnValue({
      subscribe: vi.fn(),
    });

    let fullscreenElement: Element | null = null;
    const requestFullscreen = vi.fn(async function (this: HTMLElement) {
      fullscreenElement = this;
      document.dispatchEvent(new Event("fullscreenchange"));
    });
    const originalFullscreenElement = Object.getOwnPropertyDescriptor(
      document,
      "fullscreenElement",
    );
    const originalRequestFullscreen = Object.getOwnPropertyDescriptor(
      HTMLElement.prototype,
      "requestFullscreen",
    );

    Object.defineProperty(document, "fullscreenElement", {
      configurable: true,
      get: () => fullscreenElement,
    });
    Object.defineProperty(HTMLElement.prototype, "requestFullscreen", {
      configurable: true,
      value: requestFullscreen,
    });

    try {
      render(
        <MemoryRouter initialEntries={["/sessions/sample-retro"]}>
          <Routes>
            <Route path="/sessions/:sessionId" element={<SessionPage />} />
          </Routes>
        </MemoryRouter>,
      );

      const utilityRail = screen.getByRole("navigation", { name: /session utility rail/i });
      fireEvent.click(within(utilityRail).getByRole("button", { name: "Control" }));
      expect(
        screen.getByText(/keyboard: arrows\/wasd move, space\/j\/z jump/i),
      ).toBeInTheDocument();

      fireEvent.click(screen.getByTestId("frame-surface-immersive-fullscreen"));
      await waitFor(() => {
        expect(requestFullscreen).toHaveBeenCalled();
      });

      expect(
        screen.getByTestId("session-stage").querySelector(".session-stage__fullscreen-button"),
      ).toBeNull();
      expect(screen.getByTestId("frame-surface-immersive-fullscreen")).toHaveAttribute(
        "aria-label",
        "Exit fullscreen",
      );
    } finally {
      if (originalFullscreenElement) {
        Object.defineProperty(document, "fullscreenElement", originalFullscreenElement);
      } else {
        delete (document as { fullscreenElement?: unknown }).fullscreenElement;
      }
      if (originalRequestFullscreen) {
        Object.defineProperty(HTMLElement.prototype, "requestFullscreen", originalRequestFullscreen);
      } else {
        delete (HTMLElement.prototype as { requestFullscreen?: unknown }).requestFullscreen;
      }
    }
  });

  it("routes pure human realtime frame input through websocket when the capability is enabled", async () => {
    class FakeWebSocket {
      static readonly CONNECTING = 0;
      static readonly OPEN = 1;
      static instances: FakeWebSocket[] = [];

      readonly send = vi.fn();
      readonly url: string;
      readyState = FakeWebSocket.CONNECTING;
      onopen: ((event: Event) => void) | null = null;
      onclose: ((event: Event) => void) | null = null;
      onerror: ((event: Event) => void) | null = null;
      onmessage: ((event: MessageEvent<string>) => void) | null = null;

      constructor(url: string) {
        this.url = url;
        FakeWebSocket.instances.push(this);
      }

      close(): void {
        this.readyState = 3;
        this.onclose?.(new Event("close"));
      }

      open(): void {
        this.readyState = FakeWebSocket.OPEN;
        this.onopen?.(new Event("open"));
      }
    }

    const submitActionLowLatency = vi.fn().mockResolvedValue(undefined);
    const snapshot: ArenaSessionStoreSnapshot = {
      status: "ready",
      sessionRequest: {
        sessionId: "sample-retro-ws",
      },
      session: {
        sessionId: "sample-retro-ws",
        gameId: "retro_platformer",
        pluginId: "arena.visualization.retro_platformer.frame_v1",
        lifecycle: "live_running",
        playback: {
          mode: "live_tail",
          cursorTs: 4023,
          cursorEventSeq: 23,
          speed: 1,
          canSeek: true,
        },
        observer: {
          observerId: "player_0",
          observerKind: "player",
        },
        scheduling: {
          family: "real_time_tick",
          phase: "waiting_for_intent",
          acceptsHumanIntent: true,
          activeActorId: "player_0",
        },
        capabilities: {
          supportsRealtimeInputWebSocket: true,
        },
        summary: {},
        timeline: {},
      },
      sceneStatus: "ready",
      scene: retroScene as VisualScene,
      currentSceneSeq: 23,
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

    createArenaGatewayClientMock.mockReturnValue({
      buildRealtimeActionSocketUrl: vi
        .fn()
        .mockReturnValue("ws://arena.local/arena_visual/sessions/sample-retro-ws/actions/ws"),
    });
    createArenaSessionStoreMock.mockReturnValue({
      getSnapshot: () => snapshot,
      subscribe: () => () => {},
      loadSession: vi.fn().mockResolvedValue(undefined),
      refreshSession: vi.fn().mockResolvedValue(undefined),
      loadMoreTimeline: vi.fn().mockResolvedValue(undefined),
      loadScene: vi.fn().mockResolvedValue(undefined),
      advanceReplayPlayback: vi.fn(),
      setCurrentSceneSeq: vi.fn(),
      setPlaybackMode: vi.fn(),
      setTimelineFilters: vi.fn(),
      setObserver: vi.fn().mockResolvedValue(undefined),
      submitControl: vi.fn().mockResolvedValue(undefined),
      submitAction: vi.fn().mockResolvedValue(undefined),
      submitActionLowLatency,
      submitChat: vi.fn().mockResolvedValue(undefined),
      clearLatestActionReceipt: vi.fn(),
    } as unknown as ArenaSessionStore);
    createArenaMediaResolverMock.mockReturnValue({
      subscribe: vi.fn((request, listener) => {
        listener({
          mediaId: request.mediaId,
          status: "ready",
          src: FRAME_DATA_URL,
        });
        return () => {};
      }),
    });
    vi.stubGlobal("WebSocket", FakeWebSocket as unknown as typeof WebSocket);

    try {
      render(
        <MemoryRouter initialEntries={["/sessions/sample-retro-ws"]}>
          <Routes>
            <Route path="/sessions/:sessionId" element={<SessionPage />} />
          </Routes>
        </MemoryRouter>,
      );

      expect(FakeWebSocket.instances).toHaveLength(1);
      FakeWebSocket.instances[0]?.open();

      await waitFor(() => {
        expect(screen.getByTestId("frame-surface-image")).toHaveAttribute("src", FRAME_DATA_URL);
      });

      fireEvent.keyDown(window, { key: "ArrowRight" });

      await waitFor(() => {
        expect(FakeWebSocket.instances[0]?.send).toHaveBeenCalledWith(
          expect.stringContaining("\"move\":\"right\""),
        );
      });
      expect(submitActionLowLatency).not.toHaveBeenCalled();
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it("switches dense card-table sessions into the wide-stage arena layout", async () => {
    const snapshot: ArenaSessionStoreSnapshot = {
      status: "ready",
      sessionRequest: {
        sessionId: "sample-card",
      },
      session: {
        sessionId: "sample-card",
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
          observerModes: ["player", "global"],
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

    createArenaGatewayClientMock.mockReturnValue({});
    createArenaSessionStoreMock.mockReturnValue({
      getSnapshot: () => snapshot,
      subscribe: () => () => {},
      loadSession: vi.fn().mockResolvedValue(undefined),
      loadMoreTimeline: vi.fn().mockResolvedValue(undefined),
      loadScene: vi.fn().mockResolvedValue(undefined),
      setCurrentSceneSeq: vi.fn(),
      setPlaybackMode: vi.fn(),
      setTimelineFilters: vi.fn(),
      setObserver: vi.fn().mockResolvedValue(undefined),
      submitControl: vi.fn().mockResolvedValue(undefined),
      submitAction: vi.fn().mockResolvedValue(undefined),
      submitChat: vi.fn().mockResolvedValue(undefined),
      clearLatestActionReceipt: vi.fn(),
    } as unknown as ArenaSessionStore);
    createArenaMediaResolverMock.mockReturnValue({
      subscribe: vi.fn(),
    });

    const view = render(
      <MemoryRouter initialEntries={["/sessions/sample-card"]}>
        <Routes>
          <Route path="/sessions/:sessionId" element={<SessionPage />} />
        </Routes>
      </MemoryRouter>,
    );

    expect(view.container.querySelector(".arena-layout")).toHaveClass(
      "arena-layout--wide-stage",
    );
  });

  it("threads run_id from the URL query into the session load request", async () => {
    const loadSession = vi.fn().mockResolvedValue(undefined);
    const snapshot: ArenaSessionStoreSnapshot = {
      status: "idle",
      sceneStatus: "idle",
      timeline: {
        status: "idle",
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
    };

    createArenaGatewayClientMock.mockReturnValue({});
    createArenaSessionStoreMock.mockReturnValue({
      getSnapshot: () => snapshot,
      subscribe: () => () => {},
      loadSession,
      loadMoreTimeline: vi.fn().mockResolvedValue(undefined),
      loadScene: vi.fn().mockResolvedValue(undefined),
      setCurrentSceneSeq: vi.fn(),
      setPlaybackMode: vi.fn(),
      setTimelineFilters: vi.fn(),
      setObserver: vi.fn().mockResolvedValue(undefined),
      submitControl: vi.fn().mockResolvedValue(undefined),
      submitAction: vi.fn().mockResolvedValue(undefined),
      submitChat: vi.fn().mockResolvedValue(undefined),
      clearLatestActionReceipt: vi.fn(),
    } as unknown as ArenaSessionStore);
    createArenaMediaResolverMock.mockReturnValue({
      subscribe: vi.fn(),
    });

    render(
      <MemoryRouter initialEntries={["/sessions/sample-1?run_id=run-live-9"]}>
        <Routes>
          <Route path="/sessions/:sessionId" element={<SessionPage />} />
        </Routes>
      </MemoryRouter>,
    );

    await waitFor(() => {
      expect(loadSession).toHaveBeenCalledWith({
        sessionId: "sample-1",
        runId: "run-live-9",
      });
    });
  });

  it("polls the live timeline while the session stays on live tail", async () => {
    vi.useFakeTimers();
    const loadSession = vi.fn().mockResolvedValue(undefined);
    const loadMoreTimeline = vi.fn().mockResolvedValue(undefined);
    const refreshSession = vi.fn().mockResolvedValue(undefined);
    const snapshot: ArenaSessionStoreSnapshot = {
      status: "ready",
      sessionRequest: {
        sessionId: "sample-1",
        runId: "run-live-9",
      },
      session: {
        sessionId: "sample-1",
        gameId: "pettingzoo",
        pluginId: "arena.visualization.pettingzoo.frame_v1",
        lifecycle: "live_running",
        playback: {
          mode: "live_tail",
          cursorTs: 1005,
          cursorEventSeq: 5,
          speed: 1,
          canSeek: true,
        },
        observer: {
          observerId: "",
          observerKind: "spectator",
        },
        scheduling: {
          family: "real_time_tick",
          phase: "advancing",
          acceptsHumanIntent: false,
        },
        capabilities: {},
        summary: {},
        timeline: {},
      },
      sceneStatus: "idle",
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
    };

    createArenaGatewayClientMock.mockReturnValue({});
    createArenaSessionStoreMock.mockReturnValue({
      getSnapshot: () => snapshot,
      subscribe: () => () => {},
      loadSession,
      refreshSession,
      loadMoreTimeline,
      loadScene: vi.fn().mockResolvedValue(undefined),
      setCurrentSceneSeq: vi.fn(),
      setPlaybackMode: vi.fn(),
      setTimelineFilters: vi.fn(),
      setObserver: vi.fn().mockResolvedValue(undefined),
      submitControl: vi.fn().mockResolvedValue(undefined),
      submitAction: vi.fn().mockResolvedValue(undefined),
      submitChat: vi.fn().mockResolvedValue(undefined),
      clearLatestActionReceipt: vi.fn(),
    } as unknown as ArenaSessionStore);
    createArenaMediaResolverMock.mockReturnValue({
      subscribe: vi.fn(),
    });

    render(
      <MemoryRouter initialEntries={["/sessions/sample-1?run_id=run-live-9"]}>
        <Routes>
          <Route path="/sessions/:sessionId" element={<SessionPage />} />
        </Routes>
      </MemoryRouter>,
    );

    await Promise.resolve();

    expect(loadSession).toHaveBeenCalledWith({
      sessionId: "sample-1",
      runId: "run-live-9",
    });

    await vi.advanceTimersByTimeAsync(350);

    expect(loadMoreTimeline.mock.calls.length).toBeGreaterThanOrEqual(2);
    expect(refreshSession.mock.calls.length).toBeGreaterThanOrEqual(2);

    await vi.advanceTimersByTimeAsync(850);

    expect(loadMoreTimeline).toHaveBeenCalled();
    expect(refreshSession).toHaveBeenCalled();
  });

  it("reuses a low-latency live scene instead of reloading it on every live-tail seq bump", async () => {
    const lowLatencyScene: VisualScene = {
      ...(retroScene as VisualScene),
      phase: "live",
      media: {
        primary: {
          mediaId: "live-channel-main",
          transport: "low_latency_channel",
          mimeType: "multipart/x-mixed-replace",
          url: "/arena_visual/sessions/sample-1/media/live-channel-main/stream",
        },
        auxiliary: [],
      },
    };
    const { store, setSnapshot } = createMutableStore({
      status: "ready",
      sessionRequest: {
        sessionId: "sample-1",
      },
      session: {
        sessionId: "sample-1",
        gameId: "retro_platformer",
        pluginId: "arena.visualization.retro_platformer.frame_v1",
        lifecycle: "live_running",
        playback: {
          mode: "live_tail",
          cursorTs: 1005,
          cursorEventSeq: 23,
          speed: 1,
          canSeek: true,
        },
        observer: {
          observerId: "player_0",
          observerKind: "player",
        },
        scheduling: {
          family: "real_time_tick",
          phase: "advancing",
          acceptsHumanIntent: true,
          activeActorId: "player_0",
        },
        capabilities: {},
        summary: {},
        timeline: {},
      },
      sceneStatus: "ready",
      scene: lowLatencyScene,
      currentSceneSeq: lowLatencyScene.seq,
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
      error: undefined,
    });

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

    await act(async () => {
      setSnapshot({
        ...store.getSnapshot(),
        session: {
          ...store.getSnapshot().session!,
          playback: {
            ...store.getSnapshot().session!.playback,
            cursorEventSeq: 24,
          },
        },
        currentSceneSeq: 24,
      });
      await Promise.resolve();
    });

    expect(store.loadScene).not.toHaveBeenCalled();
  });

  it("throttles timeline polling while a low-latency live scene is streaming", async () => {
    vi.useFakeTimers();
    const loadMoreTimeline = vi.fn().mockResolvedValue(undefined);
    const refreshSession = vi.fn().mockResolvedValue(undefined);
    const lowLatencyScene: VisualScene = {
      ...(retroScene as VisualScene),
      phase: "live",
      media: {
        primary: {
          mediaId: "live-channel-main",
          transport: "low_latency_channel",
          mimeType: "multipart/x-mixed-replace",
          url: "/arena_visual/sessions/sample-1/media/live-channel-main/stream",
        },
        auxiliary: [],
      },
    };
    const snapshot: ArenaSessionStoreSnapshot = {
      status: "ready",
      sessionRequest: {
        sessionId: "sample-1",
      },
      session: {
        sessionId: "sample-1",
        gameId: "retro_platformer",
        pluginId: "arena.visualization.retro_platformer.frame_v1",
        lifecycle: "live_running",
        playback: {
          mode: "live_tail",
          cursorTs: 1005,
          cursorEventSeq: lowLatencyScene.seq,
          speed: 1,
          canSeek: true,
        },
        observer: {
          observerId: "player_0",
          observerKind: "player",
        },
        scheduling: {
          family: "real_time_tick",
          phase: "advancing",
          acceptsHumanIntent: true,
          activeActorId: "player_0",
        },
        capabilities: {},
        summary: {},
        timeline: {},
      },
      sceneStatus: "ready",
      scene: lowLatencyScene,
      currentSceneSeq: lowLatencyScene.seq,
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
      error: undefined,
    };

    createArenaGatewayClientMock.mockReturnValue({});
    createArenaSessionStoreMock.mockReturnValue({
      getSnapshot: () => snapshot,
      subscribe: () => () => {},
      loadSession: vi.fn().mockResolvedValue(undefined),
      refreshSession,
      loadMoreTimeline,
      loadScene: vi.fn().mockResolvedValue(undefined),
      setCurrentSceneSeq: vi.fn(),
      setPlaybackMode: vi.fn(),
      setTimelineFilters: vi.fn(),
      setObserver: vi.fn().mockResolvedValue(undefined),
      submitControl: vi.fn().mockResolvedValue(undefined),
      submitAction: vi.fn().mockResolvedValue(undefined),
      submitChat: vi.fn().mockResolvedValue(undefined),
      clearLatestActionReceipt: vi.fn(),
    } as unknown as ArenaSessionStore);
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

    await Promise.resolve();
    await vi.advanceTimersByTimeAsync(350);

    expect(refreshSession.mock.calls.length).toBeGreaterThanOrEqual(2);
    expect(loadMoreTimeline.mock.calls.length).toBeLessThanOrEqual(1);
  });

  it("prefers the live update stream over interval polling when the session supports it", async () => {
    vi.useFakeTimers();
    const loadSession = vi.fn().mockResolvedValue(undefined);
    const loadMoreTimeline = vi.fn().mockResolvedValue(undefined);
    const refreshSession = vi.fn().mockResolvedValue(undefined);
    const applyLiveSession = vi.fn();
    const applyTimelinePage = vi.fn();
    const close = vi.fn();
    createArenaLiveUpdateStreamMock.mockReturnValue({ close });
    const snapshot: ArenaSessionStoreSnapshot = {
      status: "ready",
      sessionRequest: {
        sessionId: "sample-1",
        runId: "run-live-9",
      },
      session: {
        sessionId: "sample-1",
        gameId: "retro_platformer",
        pluginId: "arena.visualization.retro_platformer.frame_v1",
        lifecycle: "live_running",
        playback: {
          mode: "live_tail",
          cursorTs: 1005,
          cursorEventSeq: 5,
          speed: 1,
          canSeek: true,
        },
        observer: {
          observerId: "player_0",
          observerKind: "player",
        },
        scheduling: {
          family: "real_time_tick",
          phase: "advancing",
          acceptsHumanIntent: true,
          activeActorId: "player_0",
        },
        capabilities: {
          supportsLiveUpdateStream: true,
        },
        summary: {},
        timeline: {},
      },
      sceneStatus: "ready",
      scene: retroScene as VisualScene,
      currentSceneSeq: 5,
      timeline: {
        status: "ready",
        events: [
          {
            seq: 5,
            tsMs: 1005,
            type: "snapshot",
            label: "snapshot-5",
          },
        ],
        nextAfterSeq: 5,
        hasMore: false,
        limit: 50,
        filters: {
          eventTypes: [],
          severity: "all",
          humanIntentOnly: false,
        },
      },
      error: undefined,
    };

    createArenaGatewayClientMock.mockReturnValue({
      buildLiveUpdatesStreamUrl: vi.fn().mockReturnValue(
        "http://arena.local/arena_visual/sessions/sample-1/events?after_seq=5&run_id=run-live-9",
      ),
    });
    createArenaSessionStoreMock.mockReturnValue({
      getSnapshot: () => snapshot,
      subscribe: () => () => {},
      loadSession,
      refreshSession,
      loadMoreTimeline,
      loadScene: vi.fn().mockResolvedValue(undefined),
      setCurrentSceneSeq: vi.fn(),
      setPlaybackMode: vi.fn(),
      setTimelineFilters: vi.fn(),
      setObserver: vi.fn().mockResolvedValue(undefined),
      submitControl: vi.fn().mockResolvedValue(undefined),
      submitAction: vi.fn().mockResolvedValue(undefined),
      submitActionLowLatency: vi.fn().mockResolvedValue(undefined),
      submitChat: vi.fn().mockResolvedValue(undefined),
      clearLatestActionReceipt: vi.fn(),
      applyLiveSession,
      applyTimelinePage,
    } as unknown as ArenaSessionStore);
    createArenaMediaResolverMock.mockReturnValue({
      subscribe: vi.fn(),
    });

    render(
      <MemoryRouter initialEntries={["/sessions/sample-1?run_id=run-live-9"]}>
        <Routes>
          <Route path="/sessions/:sessionId" element={<SessionPage />} />
        </Routes>
      </MemoryRouter>,
    );

    await Promise.resolve();
    await vi.advanceTimersByTimeAsync(500);

    expect(createArenaLiveUpdateStreamMock).toHaveBeenCalledTimes(1);
    expect(refreshSession).not.toHaveBeenCalled();
    expect(loadMoreTimeline).not.toHaveBeenCalled();
  });

  it("shows Stop during live play and routes it through the finish control command", async () => {
    const submitControl = vi.fn().mockResolvedValue({
      intentId: "control-stop-1",
      state: "accepted",
      relatedEventSeq: 12,
      reason: "playback_applied",
    });
    const snapshot: ArenaSessionStoreSnapshot = {
      status: "ready",
      sessionRequest: {
        sessionId: "sample-1",
      },
      session: {
        sessionId: "sample-1",
        gameId: "retro_platformer",
        pluginId: "arena.visualization.retro_platformer.frame_v1",
        lifecycle: "live_running",
        playback: {
          mode: "live_tail",
          cursorTs: 1005,
          cursorEventSeq: 12,
          speed: 1,
          canSeek: true,
        },
        observer: {
          observerId: "player_0",
          observerKind: "player",
        },
        scheduling: {
          family: "real_time_tick",
          phase: "advancing",
          acceptsHumanIntent: true,
          activeActorId: "player_0",
        },
        capabilities: {},
        summary: {},
        timeline: {},
      },
      sceneStatus: "idle",
      currentSceneSeq: 12,
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
      error: undefined,
    };

    createArenaGatewayClientMock.mockReturnValue({});
    createArenaSessionStoreMock.mockReturnValue({
      getSnapshot: () => snapshot,
      subscribe: () => () => {},
      loadSession: vi.fn().mockResolvedValue(undefined),
      loadMoreTimeline: vi.fn().mockResolvedValue(undefined),
      refreshSession: vi.fn().mockResolvedValue(undefined),
      loadScene: vi.fn().mockResolvedValue(undefined),
      setCurrentSceneSeq: vi.fn(),
      setPlaybackMode: vi.fn(),
      setTimelineFilters: vi.fn(),
      setObserver: vi.fn().mockResolvedValue(undefined),
      submitControl,
      submitAction: vi.fn().mockResolvedValue(undefined),
      submitChat: vi.fn().mockResolvedValue(undefined),
      clearLatestActionReceipt: vi.fn(),
    } as unknown as ArenaSessionStore);
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

    expandSessionControls();
    expect(screen.getByRole("button", { name: /^stop$/i })).toBeInTheDocument();

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /^stop$/i }));
      await Promise.resolve();
    });

    expect(submitControl).toHaveBeenCalledWith({
      commandType: "finish",
    });
    expect(screen.getByRole("button", { name: /stopping/i })).toBeDisabled();
    expect(screen.getByText("Stopping session...")).toBeInTheDocument();
  });

  it("shows Restart for restart-capable pure human realtime sessions", async () => {
    const submitControl = vi.fn().mockResolvedValue({
      intentId: "control-restart-1",
      state: "accepted",
      relatedEventSeq: 28,
      reason: "playback_applied",
    });
    const snapshot: ArenaSessionStoreSnapshot = {
      status: "ready",
      sessionRequest: {
        sessionId: "sample-1",
      },
      session: {
        sessionId: "sample-1",
        gameId: "retro_platformer",
        pluginId: "arena.visualization.retro_platformer.frame_v1",
        lifecycle: "live_running",
        playback: {
          mode: "live_tail",
          cursorTs: 1005,
          cursorEventSeq: 23,
          speed: 1,
          canSeek: true,
        },
        observer: {
          observerId: "player_0",
          observerKind: "player",
        },
        scheduling: {
          family: "real_time_tick",
          phase: "advancing",
          acceptsHumanIntent: true,
          activeActorId: "player_0",
        },
        capabilities: {
          supportsRestart: true,
        },
        summary: {},
        timeline: {},
      },
      sceneStatus: "idle",
      currentSceneSeq: 23,
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
      error: undefined,
    };

    createArenaGatewayClientMock.mockReturnValue({});
    createArenaSessionStoreMock.mockReturnValue({
      getSnapshot: () => snapshot,
      subscribe: () => () => {},
      loadSession: vi.fn().mockResolvedValue(undefined),
      loadMoreTimeline: vi.fn().mockResolvedValue(undefined),
      refreshSession: vi.fn().mockResolvedValue(undefined),
      loadScene: vi.fn().mockResolvedValue(undefined),
      setCurrentSceneSeq: vi.fn(),
      setPlaybackMode: vi.fn(),
      setTimelineFilters: vi.fn(),
      setObserver: vi.fn().mockResolvedValue(undefined),
      submitControl,
      submitAction: vi.fn().mockResolvedValue(undefined),
      submitChat: vi.fn().mockResolvedValue(undefined),
      clearLatestActionReceipt: vi.fn(),
    } as unknown as ArenaSessionStore);
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

    expandSessionControls();
    expect(screen.getByRole("button", { name: /^restart$/i })).toBeInTheDocument();

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /^restart$/i }));
      await Promise.resolve();
    });

    expect(submitControl).toHaveBeenCalledWith({
      commandType: "restart",
    });
  });

  it("keeps the live utility rail available during pure human realtime play while leaving the timeline hidden", async () => {
    const snapshot: ArenaSessionStoreSnapshot = {
      status: "ready",
      sessionRequest: {
        sessionId: "sample-1",
      },
      session: {
        sessionId: "sample-1",
        gameId: "retro_platformer",
        pluginId: "arena.visualization.retro_platformer.frame_v1",
        lifecycle: "live_running",
        playback: {
          mode: "live_tail",
          cursorTs: 1005,
          cursorEventSeq: 23,
          speed: 1,
          canSeek: true,
        },
        observer: {
          observerId: "player_0",
          observerKind: "player",
        },
        scheduling: {
          family: "real_time_tick",
          phase: "advancing",
          acceptsHumanIntent: true,
          activeActorId: "player_0",
        },
        capabilities: {
          supportsLiveUpdateStream: true,
          supportsLowLatencyRealtimeInput: true,
        },
        summary: {},
        timeline: {},
      },
      sceneStatus: "ready",
      scene: {
        sceneId: "scene-23",
        gameId: "retro_platformer",
        pluginId: "arena.visualization.retro_platformer.frame_v1",
        kind: "frame",
        tsMs: 1005,
        seq: 23,
        phase: "live",
        activePlayerId: "player_0",
        legalActions: [
          { move: "noop" },
          { move: "left" },
          { move: "right" },
          { move: "jump" },
        ],
        summary: {},
        body: {
          status: {
            tick: 23,
          },
        },
        media: {
          primary: {
            mediaId: "live-channel-main",
            transport: "low_latency_channel",
            url: "/arena_visual/sessions/sample-1/media/live-channel-main/stream",
          },
        },
      },
      currentSceneSeq: 23,
      timeline: {
        status: "ready",
        events: [{ seq: 23, tsMs: 1005, type: "snapshot", label: "snapshot" }],
        nextAfterSeq: 23,
        hasMore: false,
        limit: 50,
        filters: {
          eventTypes: [],
          severity: "all",
          humanIntentOnly: false,
        },
      },
      error: undefined,
    };

    createArenaGatewayClientMock.mockReturnValue({});
    createArenaSessionStoreMock.mockReturnValue({
      getSnapshot: () => snapshot,
      subscribe: () => () => {},
      loadSession: vi.fn().mockResolvedValue(undefined),
      loadMoreTimeline: vi.fn().mockResolvedValue(undefined),
      refreshSession: vi.fn().mockResolvedValue(undefined),
      loadScene: vi.fn().mockResolvedValue(undefined),
      setCurrentSceneSeq: vi.fn(),
      setPlaybackMode: vi.fn(),
      setTimelineFilters: vi.fn(),
      setObserver: vi.fn().mockResolvedValue(undefined),
      submitControl: vi.fn().mockResolvedValue(undefined),
      submitAction: vi.fn().mockResolvedValue(undefined),
      submitChat: vi.fn().mockResolvedValue(undefined),
      clearLatestActionReceipt: vi.fn(),
    } as unknown as ArenaSessionStore);
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

    expect(screen.queryByRole("heading", { name: /event stream/i })).not.toBeInTheDocument();
    const utilityRail = screen.getByRole("navigation", { name: /session utility rail/i });
    expect(utilityRail).toBeInTheDocument();
    expect(within(utilityRail).getByRole("button", { name: "Control" })).toBeInTheDocument();
    expect(within(utilityRail).getByRole("button", { name: "Players" })).toBeInTheDocument();
    expect(within(utilityRail).getByRole("button", { name: "Events" })).toBeInTheDocument();
    expect(within(utilityRail).getByRole("button", { name: "Chat" })).toBeInTheDocument();
    expect(within(utilityRail).getByRole("button", { name: "Trace" })).toBeInTheDocument();

    fireEvent.click(within(utilityRail).getByRole("button", { name: "Players" }));

    expect(screen.getByRole("button", { name: /close utility drawer/i })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Players" })).toHaveAttribute("aria-selected", "true");
  });

  it("ticks replay playback forward while the session is replaying", async () => {
    vi.useFakeTimers();
    const advanceReplayPlayback = vi.fn();
    const snapshot: ArenaSessionStoreSnapshot = {
      status: "ready",
      sessionRequest: {
        sessionId: "sample-1",
        runId: "run-live-9",
      },
      session: {
        sessionId: "sample-1",
        gameId: "pettingzoo",
        pluginId: "arena.visualization.pettingzoo.frame_v1",
        lifecycle: "live_running",
        playback: {
          mode: "replay_playing",
          cursorTs: 1005,
          cursorEventSeq: 1,
          speed: 1,
          canSeek: true,
        },
        observer: {
          observerId: "",
          observerKind: "spectator",
        },
        scheduling: {
          family: "real_time_tick",
          phase: "completed",
          acceptsHumanIntent: false,
        },
        capabilities: {},
        summary: {},
        timeline: {},
      },
      sceneStatus: "ready",
      currentSceneSeq: 1,
      timeline: {
        status: "ready",
        events: [
          { seq: 1, tsMs: 1001, type: "snapshot", label: "snapshot" },
          { seq: 2, tsMs: 1002, type: "snapshot", label: "snapshot" },
        ],
        nextAfterSeq: null,
        hasMore: false,
        limit: 50,
        filters: {
          eventTypes: [],
          severity: "all",
          humanIntentOnly: false,
        },
      },
    };

    createArenaGatewayClientMock.mockReturnValue({});
    createArenaSessionStoreMock.mockReturnValue({
      getSnapshot: () => snapshot,
      subscribe: () => () => {},
      loadSession: vi.fn().mockResolvedValue(undefined),
      loadMoreTimeline: vi.fn().mockResolvedValue(undefined),
      loadScene: vi.fn().mockResolvedValue(undefined),
      advanceReplayPlayback,
      setCurrentSceneSeq: vi.fn(),
      setPlaybackMode: vi.fn(),
      setTimelineFilters: vi.fn(),
      setObserver: vi.fn().mockResolvedValue(undefined),
      submitControl: vi.fn().mockResolvedValue(undefined),
      submitAction: vi.fn().mockResolvedValue(undefined),
      submitChat: vi.fn().mockResolvedValue(undefined),
      clearLatestActionReceipt: vi.fn(),
    } as unknown as ArenaSessionStore);
    createArenaMediaResolverMock.mockReturnValue({
      subscribe: vi.fn(),
    });

    render(
      <MemoryRouter initialEntries={["/sessions/sample-1?run_id=run-live-9"]}>
        <Routes>
          <Route path="/sessions/:sessionId" element={<SessionPage />} />
        </Routes>
      </MemoryRouter>,
    );

    await Promise.resolve();
    await vi.advanceTimersByTimeAsync(900);

    expect(advanceReplayPlayback).toHaveBeenCalled();
  });

  it("shows a hard replay-entry overlay until the first replay scene arrives", async () => {
    const baseScene = gomokuScene as VisualScene;
    const snapshot: ArenaSessionStoreSnapshot = {
      status: "ready",
      sessionRequest: {
        sessionId: "sample-1",
      },
      session: {
        sessionId: "sample-1",
        gameId: "gomoku",
        pluginId: "arena.visualization.gomoku.board_v1",
        lifecycle: "live_ended",
        playback: {
          mode: "live_tail",
          cursorTs: 1005,
          cursorEventSeq: 5,
          speed: 1,
          canSeek: true,
        },
        observer: {
          observerId: "",
          observerKind: "spectator",
        },
        scheduling: {
          family: "turn",
          phase: "completed",
          acceptsHumanIntent: false,
        },
        capabilities: {},
        summary: {},
        timeline: {},
      },
      sceneStatus: "ready",
      scene: baseScene,
      currentSceneSeq: 5,
      timeline: {
        status: "ready",
        events: [
          { seq: 1, tsMs: 1001, type: "snapshot", label: "snapshot" },
          { seq: 5, tsMs: 1005, type: "result", label: "result" },
        ],
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
    const { setSnapshot, store } = createMutableStore(snapshot);

    createArenaGatewayClientMock.mockReturnValue({});
    createArenaSessionStoreMock.mockReturnValue(store);
    createArenaMediaResolverMock.mockReturnValue({
      subscribe: vi.fn(),
    });

    const view = render(
      <MemoryRouter initialEntries={["/sessions/sample-1"]}>
        <Routes>
          <Route path="/sessions/:sessionId" element={<SessionPage />} />
        </Routes>
      </MemoryRouter>,
    );

    expect(screen.queryByText("Loading replay scene...")).not.toBeInTheDocument();

    act(() => {
      setSnapshot({
        ...snapshot,
        session: {
          ...snapshot.session!,
          playback: {
            ...snapshot.session!.playback,
            mode: "replay_playing",
            cursorEventSeq: 1,
          },
        },
        sceneStatus: "loading",
        currentSceneSeq: 1,
      });
    });

    expect(screen.getByText("Loading replay scene...")).toBeInTheDocument();
    expect(view.container.querySelector(".session-stage.is-hard-transition")).not.toBeNull();

    act(() => {
      setSnapshot({
        ...snapshot,
        session: {
          ...snapshot.session!,
          playback: {
            ...snapshot.session!.playback,
            mode: "replay_playing",
            cursorEventSeq: 1,
          },
        },
        sceneStatus: "ready",
        scene: {
          ...baseScene,
          seq: 1,
        },
        currentSceneSeq: 1,
      });
    });

    await waitFor(() => {
      expect(screen.queryByText("Loading replay scene...")).not.toBeInTheDocument();
    });
    expect(view.container.querySelector(".session-stage.is-hard-transition")).toBeNull();
  });

  it("keeps steady replay playback free from transition-overlay flicker", async () => {
    const snapshot: ArenaSessionStoreSnapshot = {
      status: "ready",
      sessionRequest: {
        sessionId: "sample-1",
      },
      session: {
        sessionId: "sample-1",
        gameId: "gomoku",
        pluginId: "arena.visualization.gomoku.board_v1",
        lifecycle: "live_ended",
        playback: {
          mode: "replay_playing",
          cursorTs: 1005,
          cursorEventSeq: 2,
          speed: 1,
          canSeek: true,
        },
        observer: {
          observerId: "",
          observerKind: "spectator",
        },
        scheduling: {
          family: "turn",
          phase: "completed",
          acceptsHumanIntent: false,
        },
        capabilities: {},
        summary: {},
        timeline: {},
      },
      sceneStatus: "loading",
      scene: {
        ...(gomokuScene as VisualScene),
        seq: 1,
      },
      currentSceneSeq: 2,
      timeline: {
        status: "ready",
        events: [
          { seq: 1, tsMs: 1001, type: "snapshot", label: "snapshot" },
          { seq: 2, tsMs: 1002, type: "snapshot", label: "snapshot" },
        ],
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

    createArenaGatewayClientMock.mockReturnValue({});
    createArenaSessionStoreMock.mockReturnValue({
      getSnapshot: () => snapshot,
      subscribe: () => () => {},
      loadSession: vi.fn().mockResolvedValue(undefined),
      loadMoreTimeline: vi.fn().mockResolvedValue(undefined),
      loadScene: vi.fn().mockResolvedValue(undefined),
      setCurrentSceneSeq: vi.fn(),
      setPlaybackMode: vi.fn(),
      setTimelineFilters: vi.fn(),
      setObserver: vi.fn().mockResolvedValue(undefined),
      submitControl: vi.fn().mockResolvedValue(undefined),
      submitAction: vi.fn().mockResolvedValue(undefined),
      submitChat: vi.fn().mockResolvedValue(undefined),
      clearLatestActionReceipt: vi.fn(),
    } as unknown as ArenaSessionStore);
    createArenaMediaResolverMock.mockReturnValue({
      subscribe: vi.fn(),
    });

    const view = render(
      <MemoryRouter initialEntries={["/sessions/sample-1"]}>
        <Routes>
          <Route path="/sessions/:sessionId" element={<SessionPage />} />
        </Routes>
      </MemoryRouter>,
    );

    expect(screen.queryByText("Loading replay scene...")).not.toBeInTheDocument();
    expect(view.container.querySelector(".session-stage.is-transitioning")).toBeNull();
  });

  it("keeps live tail scene refreshes free from stage transition flicker", async () => {
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
          mode: "live_tail",
          cursorTs: 1005,
          cursorEventSeq: 2,
          speed: 1,
          canSeek: true,
        },
        observer: {
          observerId: "",
          observerKind: "spectator",
        },
        scheduling: {
          family: "real_time_tick",
          phase: "advancing",
          acceptsHumanIntent: false,
        },
        capabilities: {},
        summary: {},
        timeline: {},
      },
      sceneStatus: "loading",
      scene: {
        ...(gomokuScene as VisualScene),
        seq: 1,
      },
      currentSceneSeq: 2,
      timeline: {
        status: "ready",
        events: [
          { seq: 1, tsMs: 1001, type: "snapshot", label: "snapshot" },
          { seq: 2, tsMs: 1002, type: "snapshot", label: "snapshot" },
        ],
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

    createArenaGatewayClientMock.mockReturnValue({});
    createArenaSessionStoreMock.mockReturnValue({
      getSnapshot: () => snapshot,
      subscribe: () => () => {},
      loadSession: vi.fn().mockResolvedValue(undefined),
      loadMoreTimeline: vi.fn().mockResolvedValue(undefined),
      loadScene: vi.fn().mockResolvedValue(undefined),
      setCurrentSceneSeq: vi.fn(),
      setPlaybackMode: vi.fn(),
      setTimelineFilters: vi.fn(),
      setObserver: vi.fn().mockResolvedValue(undefined),
      submitControl: vi.fn().mockResolvedValue(undefined),
      submitAction: vi.fn().mockResolvedValue(undefined),
      submitChat: vi.fn().mockResolvedValue(undefined),
      clearLatestActionReceipt: vi.fn(),
    } as unknown as ArenaSessionStore);
    createArenaMediaResolverMock.mockReturnValue({
      subscribe: vi.fn(),
    });

    const view = render(
      <MemoryRouter initialEntries={["/sessions/sample-1"]}>
        <Routes>
          <Route path="/sessions/:sessionId" element={<SessionPage />} />
        </Routes>
      </MemoryRouter>,
    );

    expect(screen.queryByText("Syncing scene...")).not.toBeInTheDocument();
    expect(view.container.querySelector(".session-stage.is-transitioning")).toBeNull();
  });

  it("shows Finish after a post-live replay interaction and routes the finish command through the store", async () => {
    const submitControl = vi.fn().mockResolvedValue({
      intentId: "control-finish-1",
      state: "accepted",
      relatedEventSeq: 1,
      reason: "playback_applied",
    });
    const snapshot: ArenaSessionStoreSnapshot = {
      status: "ready",
      sessionRequest: {
        sessionId: "sample-1",
      },
      session: {
        sessionId: "sample-1",
        gameId: "tictactoe",
        pluginId: "arena.visualization.tictactoe.board_v1",
        lifecycle: "live_ended",
        playback: {
          mode: "live_tail",
          cursorTs: 1005,
          cursorEventSeq: 46,
          speed: 1,
          canSeek: true,
        },
        observer: {
          observerId: "",
          observerKind: "spectator",
        },
        scheduling: {
          family: "turn",
          phase: "completed",
          acceptsHumanIntent: false,
        },
        capabilities: {
          observerModes: ["global"],
        },
        summary: {},
        timeline: {},
      },
      sceneStatus: "idle",
      currentSceneSeq: 46,
      timeline: {
        status: "ready",
        events: [
          {
            seq: 46,
            tsMs: 1006,
            type: "result",
            label: "result",
            payload: {},
          },
        ],
        nextAfterSeq: null,
        hasMore: false,
        limit: 50,
        filters: {
          eventTypes: [],
          severity: "all",
          humanIntentOnly: false,
        },
      },
      error: undefined,
    };

    createArenaGatewayClientMock.mockReturnValue({});
    createArenaSessionStoreMock.mockReturnValue({
      getSnapshot: () => snapshot,
      subscribe: () => () => {},
      loadSession: vi.fn().mockResolvedValue(undefined),
      loadMoreTimeline: vi.fn().mockResolvedValue(undefined),
      loadScene: vi.fn().mockResolvedValue(undefined),
      setCurrentSceneSeq: vi.fn(),
      setPlaybackMode: vi.fn(),
      setTimelineFilters: vi.fn(),
      setObserver: vi.fn().mockResolvedValue(undefined),
      submitControl,
      submitAction: vi.fn().mockResolvedValue(undefined),
      submitChat: vi.fn().mockResolvedValue(undefined),
      clearLatestActionReceipt: vi.fn(),
    } as unknown as ArenaSessionStore);
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

    expandSessionControls();
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /replay/i }));
      await Promise.resolve();
    });

    expect(screen.getByRole("button", { name: /^finish$/i })).toBeInTheDocument();

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /^finish$/i }));
      await Promise.resolve();
    });

    expect(submitControl).toHaveBeenNthCalledWith(1, {
      commandType: "replay",
    });
    expect(submitControl).toHaveBeenNthCalledWith(2, {
      commandType: "finish",
    });

    expect(screen.getByRole("button", { name: /finishing/i })).toBeDisabled();
    expect(screen.getByText("Finishing session...")).toBeInTheDocument();
  });

  it("polls session refresh while finish is pending so the host can observe closure", async () => {
    const { store } = createMutableStore({
      status: "ready",
      sessionRequest: {
        sessionId: "sample-1",
      },
      session: {
        sessionId: "sample-1",
        gameId: "mahjong",
        pluginId: "arena.visualization.mahjong.table_v1",
        lifecycle: "live_ended",
        playback: {
          mode: "paused",
          cursorTs: 1006,
          cursorEventSeq: 46,
          speed: 1,
          canSeek: true,
        },
        observer: {
          observerId: "",
          observerKind: "spectator",
        },
        scheduling: {
          family: "turn",
          phase: "completed",
          acceptsHumanIntent: false,
        },
        capabilities: {},
        summary: {},
        timeline: {},
      },
      sceneStatus: "idle",
      currentSceneSeq: 46,
      timeline: {
        status: "ready",
        events: [
          {
            seq: 46,
            tsMs: 1006,
            type: "result",
            label: "result",
            payload: {},
          },
        ],
        nextAfterSeq: null,
        hasMore: false,
        limit: 50,
        filters: {
          eventTypes: [],
          severity: "all",
          humanIntentOnly: false,
        },
      },
      error: undefined,
    });
    const submitControl = vi
      .fn()
      .mockResolvedValueOnce({
        intentId: "control-replay",
        state: "accepted",
        reason: "playback_applied",
      })
      .mockResolvedValueOnce({
        intentId: "control-finish",
        state: "accepted",
        reason: "playback_applied",
      });
    const refreshSession = vi.fn().mockResolvedValue(undefined);
    store.submitControl = submitControl;
    store.refreshSession = refreshSession;

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

    expandSessionControls();
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /replay/i }));
      await Promise.resolve();
    });

    expect(screen.getByRole("button", { name: /^finish$/i })).toBeInTheDocument();

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /^finish$/i }));
      await Promise.resolve();
    });

    await waitFor(() => {
      expect(submitControl).toHaveBeenNthCalledWith(1, {
        commandType: "replay",
      });
      expect(submitControl).toHaveBeenNthCalledWith(2, {
        commandType: "finish",
      });
    });

    expect(screen.getByText("Finishing session...")).toBeInTheDocument();

    await new Promise((resolve) => {
      window.setTimeout(resolve, 350);
    });

    expect(refreshSession).toHaveBeenCalled();
  });

  it("attempts to close the viewer window after a stop request reaches closed lifecycle", async () => {
    const closeSpy = vi.spyOn(window, "close").mockImplementation(() => {
      return undefined;
    });
    const { store, setSnapshot } = createMutableStore({
      status: "ready",
      sessionRequest: {
        sessionId: "sample-1",
      },
      session: {
        sessionId: "sample-1",
        gameId: "retro_platformer",
        pluginId: "arena.visualization.retro_platformer.frame_v1",
        lifecycle: "live_running",
        playback: {
          mode: "live_tail",
          cursorTs: 1006,
          cursorEventSeq: 46,
          speed: 1,
          canSeek: true,
        },
        observer: {
          observerId: "player_0",
          observerKind: "player",
        },
        scheduling: {
          family: "real_time_tick",
          phase: "advancing",
          acceptsHumanIntent: true,
          activeActorId: "player_0",
        },
        capabilities: {},
        summary: {},
        timeline: {},
      },
      sceneStatus: "idle",
      currentSceneSeq: 46,
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
      error: undefined,
    });
    store.submitControl = vi.fn().mockResolvedValue({
      intentId: "control-stop-close",
      state: "accepted",
      relatedEventSeq: 46,
      reason: "playback_applied",
    });

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

    expandSessionControls();
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /^stop$/i }));
      await Promise.resolve();
    });

    await act(async () => {
      setSnapshot({
        ...store.getSnapshot(),
        session: {
          ...store.getSnapshot().session!,
          lifecycle: "closed",
        },
      });
      await Promise.resolve();
    });

    expect(closeSpy).toHaveBeenCalled();
  });

  it("disables scene-incompatible controls while the session stays on live tail", async () => {
    const snapshot: ArenaSessionStoreSnapshot = {
      status: "ready",
      sessionRequest: {
        sessionId: "sample-1",
      },
      session: {
        sessionId: "sample-1",
        gameId: "tictactoe",
        pluginId: "arena.visualization.tictactoe.board_v1",
        lifecycle: "live_running",
        playback: {
          mode: "live_tail",
          cursorTs: 1005,
          cursorEventSeq: 9,
          speed: 1,
          canSeek: true,
        },
        observer: {
          observerId: "",
          observerKind: "spectator",
        },
        scheduling: {
          family: "turn",
          phase: "advancing",
          acceptsHumanIntent: false,
        },
        capabilities: {
          observerModes: ["global"],
        },
        summary: {},
        timeline: {},
      },
      sceneStatus: "ready",
      scene: gomokuScene as VisualScene,
      currentSceneSeq: 9,
      timeline: {
        status: "ready",
        events: [
          { seq: 1, tsMs: 1001, type: "snapshot", label: "snapshot" },
          { seq: 9, tsMs: 1009, type: "snapshot", label: "snapshot" },
        ],
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

    createArenaGatewayClientMock.mockReturnValue({});
    createArenaSessionStoreMock.mockReturnValue({
      getSnapshot: () => snapshot,
      subscribe: () => () => {},
      loadSession: vi.fn().mockResolvedValue(undefined),
      loadMoreTimeline: vi.fn().mockResolvedValue(undefined),
      loadScene: vi.fn().mockResolvedValue(undefined),
      setCurrentSceneSeq: vi.fn(),
      setPlaybackMode: vi.fn(),
      setTimelineFilters: vi.fn(),
      setObserver: vi.fn().mockResolvedValue(undefined),
      submitControl: vi.fn().mockResolvedValue(undefined),
      submitAction: vi.fn().mockResolvedValue(undefined),
      submitChat: vi.fn().mockResolvedValue(undefined),
      clearLatestActionReceipt: vi.fn(),
    } as unknown as ArenaSessionStore);
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

    expandSessionControls();
    expect(screen.getByRole("button", { name: /live tail/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /pause/i })).toBeEnabled();
    expect(screen.getByRole("button", { name: /replay/i })).toBeEnabled();
    expect(screen.getByRole("button", { name: /step -1/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /step \+1/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /^end$/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /back to tail/i })).toBeDisabled();
  });

  it("keeps the replay control set operational while replaying", async () => {
    const submitControl = vi.fn().mockResolvedValue({
      intentId: "control-replay-ops-1",
      state: "accepted",
      relatedEventSeq: 2,
      reason: "playback_applied",
    });
    const snapshot: ArenaSessionStoreSnapshot = {
      status: "ready",
      sessionRequest: {
        sessionId: "sample-1",
      },
      session: {
        sessionId: "sample-1",
        gameId: "tictactoe",
        pluginId: "arena.visualization.tictactoe.board_v1",
        lifecycle: "live_ended",
        playback: {
          mode: "replay_playing",
          cursorTs: 1005,
          cursorEventSeq: 2,
          speed: 1,
          canSeek: true,
        },
        observer: {
          observerId: "",
          observerKind: "spectator",
        },
        scheduling: {
          family: "turn",
          phase: "completed",
          acceptsHumanIntent: false,
        },
        capabilities: {
          observerModes: ["global"],
        },
        summary: {},
        timeline: {},
      },
      sceneStatus: "ready",
      scene: {
        ...(gomokuScene as VisualScene),
        seq: 2,
      },
      currentSceneSeq: 2,
      timeline: {
        status: "ready",
        events: [
          { seq: 1, tsMs: 1001, type: "snapshot", label: "snapshot" },
          { seq: 2, tsMs: 1002, type: "snapshot", label: "snapshot" },
          { seq: 3, tsMs: 1003, type: "snapshot", label: "snapshot" },
        ],
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

    createArenaGatewayClientMock.mockReturnValue({});
    createArenaSessionStoreMock.mockReturnValue({
      getSnapshot: () => snapshot,
      subscribe: () => () => {},
      loadSession: vi.fn().mockResolvedValue(undefined),
      loadMoreTimeline: vi.fn().mockResolvedValue(undefined),
      loadScene: vi.fn().mockResolvedValue(undefined),
      setCurrentSceneSeq: vi.fn(),
      setPlaybackMode: vi.fn(),
      setTimelineFilters: vi.fn(),
      setObserver: vi.fn().mockResolvedValue(undefined),
      submitControl,
      submitAction: vi.fn().mockResolvedValue(undefined),
      submitChat: vi.fn().mockResolvedValue(undefined),
      clearLatestActionReceipt: vi.fn(),
    } as unknown as ArenaSessionStore);
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

    expandSessionControls();
    expect(screen.getByRole("button", { name: /live tail/i })).toBeEnabled();
    expect(screen.getByRole("button", { name: /pause/i })).toBeEnabled();
    expect(screen.getByRole("button", { name: /replay/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: "2x" })).toBeEnabled();
    expect(screen.getByRole("button", { name: /step -1/i })).toBeEnabled();
    expect(screen.getByRole("button", { name: /step \+1/i })).toBeEnabled();
    expect(screen.getByRole("button", { name: /^end$/i })).toBeEnabled();
    expect(screen.getByRole("button", { name: /back to tail/i })).toBeEnabled();

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /pause/i }));
      await Promise.resolve();
    });
    expect(submitControl).toHaveBeenNthCalledWith(1, {
      commandType: "pause",
    });

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "2x" }));
      await Promise.resolve();
    });
    expect(submitControl).toHaveBeenNthCalledWith(2, {
      commandType: "set_speed",
      speed: 2,
    });

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /step -1/i }));
      await Promise.resolve();
    });
    expect(submitControl).toHaveBeenNthCalledWith(3, {
      commandType: "step",
      stepDelta: -1,
    });

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /step \+1/i }));
      await Promise.resolve();
    });
    expect(submitControl).toHaveBeenNthCalledWith(4, {
      commandType: "step",
      stepDelta: 1,
    });

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /^end$/i }));
      await Promise.resolve();
    });
    expect(submitControl).toHaveBeenNthCalledWith(5, {
      commandType: "seek_end",
    });

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /back to tail/i }));
      await Promise.resolve();
    });
    expect(submitControl).toHaveBeenNthCalledWith(6, {
      commandType: "back_to_tail",
    });

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /live tail/i }));
      await Promise.resolve();
    });
    expect(submitControl).toHaveBeenNthCalledWith(7, {
      commandType: "follow_tail",
    });
  });
});

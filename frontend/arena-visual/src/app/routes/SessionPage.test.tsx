import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

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

    await vi.advanceTimersByTimeAsync(1200);

    expect(loadMoreTimeline).toHaveBeenCalled();
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

    fireEvent.click(screen.getByRole("button", { name: /replay/i }));

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /^finish$/i })).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole("button", { name: /^finish$/i }));

    await waitFor(() => {
      expect(submitControl).toHaveBeenNthCalledWith(1, {
        commandType: "replay",
      });
      expect(submitControl).toHaveBeenNthCalledWith(2, {
        commandType: "finish",
      });
    });

    expect(screen.getByRole("button", { name: /finishing/i })).toBeDisabled();
    expect(screen.getByText("Finishing session...")).toBeInTheDocument();
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

    expect(screen.getByRole("button", { name: /live tail/i })).toBeEnabled();
    expect(screen.getByRole("button", { name: /pause/i })).toBeEnabled();
    expect(screen.getByRole("button", { name: /replay/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: "2x" })).toBeEnabled();
    expect(screen.getByRole("button", { name: /step -1/i })).toBeEnabled();
    expect(screen.getByRole("button", { name: /step \+1/i })).toBeEnabled();
    expect(screen.getByRole("button", { name: /^end$/i })).toBeEnabled();
    expect(screen.getByRole("button", { name: /back to tail/i })).toBeEnabled();

    fireEvent.click(screen.getByRole("button", { name: /pause/i }));
    await waitFor(() => {
      expect(submitControl).toHaveBeenNthCalledWith(1, {
        commandType: "pause",
      });
    });

    fireEvent.click(screen.getByRole("button", { name: "2x" }));
    await waitFor(() => {
      expect(submitControl).toHaveBeenNthCalledWith(2, {
        commandType: "set_speed",
        speed: 2,
      });
    });

    fireEvent.click(screen.getByRole("button", { name: /step -1/i }));
    await waitFor(() => {
      expect(submitControl).toHaveBeenNthCalledWith(3, {
        commandType: "step",
        stepDelta: -1,
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

    fireEvent.click(screen.getByRole("button", { name: /live tail/i }));
    await waitFor(() => {
      expect(submitControl).toHaveBeenNthCalledWith(7, {
        commandType: "follow_tail",
      });
    });
  });
});

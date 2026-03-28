import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { ArenaGatewayClient } from "../../gateway/client";
import type { TimelinePage, VisualScene, VisualSession } from "../../gateway/types";
import { createArenaGatewayClient } from "../../gateway/client";
import { createArenaSessionStore } from "./arenaSessionStore";

function jsonResponse(payload: unknown, init?: ResponseInit): Response {
  return new Response(JSON.stringify(payload), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init,
  });
}

describe("arenaSessionStore", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("loads session and the first timeline page", async () => {
    const fetchMock = vi.mocked(fetch);
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        sessionId: "sample-1",
        gameId: "gomoku",
        pluginId: "arena.visualization.gomoku.board_v1",
        lifecycle: "closed",
        playback: {
          mode: "paused",
          cursorTs: 1005,
          cursorEventSeq: 5,
          speed: 1,
          canSeek: true,
        },
        observer: { observerId: "host", observerKind: "global" },
        scheduling: {
          family: "turn",
          phase: "completed",
          acceptsHumanIntent: false,
        },
        capabilities: { canSubmitAction: true },
        summary: { winner: "player_0" },
        timeline: { eventCount: 5 },
      }),
    );
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        sessionId: "sample-1",
        afterSeq: null,
        nextAfterSeq: 2,
        limit: 50,
        hasMore: true,
        events: [
          {
            seq: 1,
            tsMs: 1001,
            type: "decision_window_open",
            label: "decision_window_open",
          },
          {
            seq: 2,
            tsMs: 1002,
            type: "action_intent",
            label: "action_intent",
          },
        ],
      }),
    );

    const client = createArenaGatewayClient({ baseUrl: "http://arena.local" });
    const store = createArenaSessionStore(client);

    await store.loadSession({ sessionId: "sample-1" });
    const state = store.getSnapshot();

    expect(state.status).toBe("ready");
    expect(state.session?.sessionId).toBe("sample-1");
    expect(state.timeline.events.map((event) => event.seq)).toEqual([1, 2]);
    expect(state.currentSceneSeq).toBe(2);
    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "http://arena.local/arena_visual/sessions/sample-1",
      expect.objectContaining({ method: "GET" }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "http://arena.local/arena_visual/sessions/sample-1/timeline",
      expect.objectContaining({ method: "GET" }),
    );
  });

  it("tracks timeline loading transitions and appends next timeline page", async () => {
    const fetchMock = vi.mocked(fetch);
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        sessionId: "sample-1",
        gameId: "gomoku",
        pluginId: "arena.visualization.gomoku.board_v1",
        lifecycle: "live_running",
        playback: {
          mode: "live_tail",
          cursorTs: 1002,
          cursorEventSeq: 2,
          speed: 1,
          canSeek: true,
        },
        observer: { observerId: "host", observerKind: "global" },
        scheduling: {
          family: "turn",
          phase: "advancing",
          acceptsHumanIntent: true,
        },
        capabilities: {},
        summary: {},
        timeline: { eventCount: 4 },
      }),
    );
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        sessionId: "sample-1",
        afterSeq: null,
        nextAfterSeq: 2,
        limit: 50,
        hasMore: true,
        events: [
          {
            seq: 1,
            tsMs: 1001,
            type: "decision_window_open",
            label: "decision_window_open",
          },
          {
            seq: 2,
            tsMs: 1002,
            type: "action_intent",
            label: "action_intent",
          },
        ],
      }),
    );
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        sessionId: "sample-1",
        afterSeq: 2,
        nextAfterSeq: 4,
        limit: 2,
        hasMore: false,
        events: [
          {
            seq: 3,
            tsMs: 1003,
            type: "action_committed",
            label: "action_committed",
          },
          {
            seq: 4,
            tsMs: 1004,
            type: "result",
            label: "result",
          },
        ],
      }),
    );

    const client = createArenaGatewayClient({ baseUrl: "http://arena.local" });
    const store = createArenaSessionStore(client);
    const timelineStatuses: string[] = [];
    const unsubscribe = store.subscribe(() => {
      timelineStatuses.push(store.getSnapshot().timeline.status);
    });

    await store.loadSession({ sessionId: "sample-1" });
    await store.loadMoreTimeline({ limit: 2 });
    unsubscribe();

    const state = store.getSnapshot();
    expect(state.timeline.events.map((event) => event.seq)).toEqual([1, 2, 3, 4]);
    expect(state.timeline.nextAfterSeq).toBe(4);
    expect(state.timeline.hasMore).toBe(false);
    expect(timelineStatuses).toContain("loading");
    expect(timelineStatuses.at(-1)).toBe("ready");
    expect(fetchMock).toHaveBeenLastCalledWith(
      "http://arena.local/arena_visual/sessions/sample-1/timeline?after_seq=2&limit=2",
      expect.objectContaining({ method: "GET" }),
    );
  });

  it("stores latest action receipt after submitAction", async () => {
    const fetchMock = vi.mocked(fetch);
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        sessionId: "sample-1",
        gameId: "gomoku",
        pluginId: "arena.visualization.gomoku.board_v1",
        lifecycle: "live_running",
        playback: {
          mode: "live_tail",
          cursorTs: 1002,
          cursorEventSeq: 2,
          speed: 1,
          canSeek: true,
        },
        observer: { observerId: "host", observerKind: "global" },
        scheduling: {
          family: "turn",
          phase: "waiting_for_intent",
          acceptsHumanIntent: true,
        },
        capabilities: {},
        summary: {},
        timeline: { eventCount: 2 },
      }),
    );
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        sessionId: "sample-1",
        afterSeq: null,
        nextAfterSeq: 2,
        limit: 50,
        hasMore: false,
        events: [],
      }),
    );
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        intentId: "intent-1",
        state: "accepted",
        relatedEventSeq: 6,
        reason: "queued",
      }),
    );

    const client = createArenaGatewayClient({ baseUrl: "http://arena.local" });
    const store = createArenaSessionStore(client);
    await store.loadSession({ sessionId: "sample-1" });
    await store.submitAction({
      playerId: "player_0",
      action: { move: "fire" },
    });

    const state = store.getSnapshot();
    expect(state.latestActionReceipt).toEqual({
      intentId: "intent-1",
      state: "accepted",
      relatedEventSeq: 6,
      reason: "queued",
    });
    expect(fetchMock).toHaveBeenLastCalledWith(
      "http://arena.local/arena_visual/sessions/sample-1/actions",
      expect.objectContaining({
        method: "POST",
      }),
    );
  });

  it("clears stale session data immediately when a new session starts loading", async () => {
    const secondSession = deferred<VisualSession>();
    const secondTimeline = deferred<TimelinePage>();
    const client = createStubClient({
      loadSession: vi
        .fn()
        .mockResolvedValueOnce(buildSession("sample-1"))
        .mockImplementationOnce(() => secondSession.promise),
      loadTimeline: vi
        .fn()
        .mockResolvedValueOnce(buildTimelinePage("sample-1", [1, 2]))
        .mockImplementationOnce(() => secondTimeline.promise),
    });
    const store = createArenaSessionStore(client);

    await store.loadSession({ sessionId: "sample-1" });
    const reloading = store.loadSession({ sessionId: "sample-2" });

    expect(store.getSnapshot().status).toBe("loading");
    expect(store.getSnapshot().session).toBeUndefined();

    secondSession.resolve(buildSession("sample-2"));
    secondTimeline.resolve(buildTimelinePage("sample-2", [9]));
    await reloading;
  });

  it("ignores stale session loads after a newer request wins", async () => {
    const firstSession = deferred<VisualSession>();
    const firstTimeline = deferred<TimelinePage>();
    const client = createStubClient({
      loadSession: vi
        .fn()
        .mockImplementationOnce(() => firstSession.promise)
        .mockResolvedValueOnce(buildSession("sample-2")),
      loadTimeline: vi
        .fn()
        .mockImplementationOnce(() => firstTimeline.promise)
        .mockResolvedValueOnce(buildTimelinePage("sample-2", [3, 4])),
    });
    const store = createArenaSessionStore(client);

    const firstLoad = store.loadSession({ sessionId: "sample-1" });
    const secondLoad = store.loadSession({ sessionId: "sample-2" });
    await secondLoad;

    firstSession.resolve(buildSession("sample-1"));
    firstTimeline.resolve(buildTimelinePage("sample-1", [1, 2]));
    await firstLoad;

    const state = store.getSnapshot();
    expect(state.sessionRequest?.sessionId).toBe("sample-2");
    expect(state.session?.sessionId).toBe("sample-2");
    expect(state.timeline.events.map((event) => event.seq)).toEqual([3, 4]);
  });

  it("ignores stale scene loads after a newer selection wins", async () => {
    const firstScene = deferred<VisualScene>();
    const client = createStubClient({
      loadScene: vi
        .fn()
        .mockImplementationOnce(() => firstScene.promise)
        .mockResolvedValueOnce(buildScene(9)),
    });
    const store = createArenaSessionStore(client);

    await store.loadSession({ sessionId: "sample-1" });

    const firstLoad = store.loadScene({ seq: 5 });
    const secondLoad = store.loadScene({ seq: 9 });
    await secondLoad;

    firstScene.resolve(buildScene(5));
    await firstLoad;

    const state = store.getSnapshot();
    expect(state.currentSceneSeq).toBe(9);
    expect(state.scene?.seq).toBe(9);
  });

  it("updates the selected seq when switching back to live tail", async () => {
    const store = createArenaSessionStore(createStubClient());

    await store.loadSession({ sessionId: "sample-1" });
    store.setCurrentSceneSeq(1);
    store.setPlaybackMode("live_tail");

    const state = store.getSnapshot();
    expect(state.session?.playback.mode).toBe("live_tail");
    expect(state.currentSceneSeq).toBe(2);
  });

  it("clears the synthetic pending receipt when action submission fails", async () => {
    const client = createStubClient({
      submitAction: vi.fn().mockRejectedValue(new Error("queue_unavailable")),
    });
    const store = createArenaSessionStore(client);

    await store.loadSession({ sessionId: "sample-1" });
    await expect(
      store.submitAction({ playerId: "player_0", action: { move: "fire" } }),
    ).rejects.toThrow("queue_unavailable");

    expect(store.getSnapshot().latestActionReceipt).toEqual({
      intentId: "rejected-intent",
      state: "rejected",
      reason: "queue_unavailable",
    });
  });

  it("reloads session and current scene for an observer override without discarding timeline state", async () => {
    const deferredSession = deferred<VisualSession>();
    const deferredScene = deferred<VisualScene>();
    const client = createStubClient({
      loadSession: vi
        .fn()
        .mockResolvedValueOnce(buildSession("sample-1"))
        .mockImplementationOnce(() => deferredSession.promise),
      loadScene: vi
        .fn()
        .mockResolvedValueOnce(buildScene(2))
        .mockImplementationOnce(() => deferredScene.promise),
    });
    const store = createArenaSessionStore(client);

    await store.loadSession({ sessionId: "sample-1" });
    await store.loadScene({ seq: 2 });

    const observerReload = store.setObserver({
      observerId: "player_0",
      observerKind: "player",
    });

    expect(store.getSnapshot().timeline.events.map((event) => event.seq)).toEqual([1, 2]);
    expect(store.getSnapshot().currentSceneSeq).toBe(2);

    deferredSession.resolve(
      buildSession("sample-1", {
        observer: {
          observerId: "player_0",
          observerKind: "player",
        },
      }),
    );
    deferredScene.resolve(
      buildScene(2, {
        observerPlayerId: "player_0",
      }),
    );
    await observerReload;

    expect(client.loadSession).toHaveBeenLastCalledWith({
      sessionId: "sample-1",
      runId: undefined,
      observer: {
        observerId: "player_0",
        observerKind: "player",
      },
    });
    expect(client.loadScene).toHaveBeenLastCalledWith({
      sessionId: "sample-1",
      runId: undefined,
      seq: 2,
      observer: {
        observerId: "player_0",
        observerKind: "player",
      },
    });

    const state = store.getSnapshot();
    expect(state.timeline.events.map((event) => event.seq)).toEqual([1, 2]);
    expect(state.session?.observer).toEqual({
      observerId: "player_0",
      observerKind: "player",
    });
    expect(state.scene?.body).toEqual({
      seq: 2,
      status: {
        observerPlayerId: "player_0",
      },
    });
  });
});

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

function buildSession(
  sessionId: string,
  overrides: Partial<VisualSession> = {},
): VisualSession {
  return {
    sessionId,
    gameId: "gomoku",
    pluginId: "arena.visualization.gomoku.board_v1",
    lifecycle: "live_running",
    playback: {
      mode: "live_tail",
      cursorTs: 1000,
      cursorEventSeq: 0,
      speed: 1,
      canSeek: true,
    },
    observer: {
      observerId: "host",
      observerKind: "global",
    },
    scheduling: {
      family: "turn",
      phase: "waiting_for_intent",
      acceptsHumanIntent: true,
    },
    capabilities: {},
    summary: {},
    timeline: {},
    ...overrides,
  };
}

function buildTimelinePage(sessionId: string, seqs: number[]): TimelinePage {
  return {
    sessionId,
    afterSeq: null,
    nextAfterSeq: seqs.at(-1) ?? null,
    limit: 50,
    hasMore: false,
    events: seqs.map((seq) => ({
      seq,
      tsMs: 1000 + seq,
      type: "snapshot",
      label: `snapshot-${seq}`,
    })),
  };
}

function buildScene(
  seq: number,
  overrides: {
    observerPlayerId?: string | null;
  } = {},
): VisualScene {
  return {
    sceneId: `scene-${seq}`,
    gameId: "gomoku",
    pluginId: "arena.visualization.gomoku.board_v1",
    kind: "board",
    tsMs: 1000 + seq,
    seq,
    phase: "live",
    activePlayerId: "player_0",
    legalActions: [],
    summary: {},
    body: {
      seq,
      status: {
        observerPlayerId:
          overrides.observerPlayerId === undefined ? null : overrides.observerPlayerId,
      },
    },
  };
}

function createStubClient(
  overrides: Partial<ArenaGatewayClient> = {},
): ArenaGatewayClient {
  return {
    loadSession: vi.fn().mockResolvedValue(buildSession("sample-1")),
    loadTimeline: vi.fn().mockResolvedValue(buildTimelinePage("sample-1", [1, 2])),
    loadScene: vi.fn().mockResolvedValue(buildScene(2)),
    loadMarkers: vi.fn().mockResolvedValue({
      sessionId: "sample-1",
      marker: "snapshot",
      seqs: [2],
    }),
    loadMedia: vi.fn().mockResolvedValue({
      mediaId: "frame-1",
      transport: "artifact_ref",
      url: "/media/frame-1",
    }),
    submitAction: vi.fn().mockResolvedValue({
      intentId: "intent-1",
      state: "accepted",
    }),
    buildMediaUrl: vi.fn().mockReturnValue("/media/frame-1"),
    ...overrides,
  };
}

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
    expect(state.currentSceneSeq).toBe(5);
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
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        sessionId: "sample-1",
        gameId: "gomoku",
        pluginId: "arena.visualization.gomoku.board_v1",
        lifecycle: "live_running",
        playback: {
          mode: "live_tail",
          cursorTs: 1006,
          cursorEventSeq: 6,
          speed: 1,
          canSeek: true,
        },
        observer: { observerId: "host", observerKind: "global" },
        scheduling: {
          family: "turn",
          phase: "advancing",
          acceptsHumanIntent: false,
        },
        capabilities: {},
        summary: {},
        timeline: { eventCount: 3 },
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
    expect(fetchMock).toHaveBeenNthCalledWith(
      3,
      "http://arena.local/arena_visual/sessions/sample-1/actions",
      expect.objectContaining({
        method: "POST",
      }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      4,
      "http://arena.local/arena_visual/sessions/sample-1?observer_kind=global&observer_id=host",
      expect.objectContaining({ method: "GET" }),
    );
    expect(fetchMock).toHaveBeenLastCalledWith(
      "http://arena.local/arena_visual/sessions/sample-1/scene?seq=2&observer_kind=global&observer_id=host",
      expect.objectContaining({ method: "GET" }),
    );
  });

  it("refreshes the live session in place without discarding timeline state", async () => {
    const client = createStubClient({
      loadSession: vi
        .fn()
        .mockResolvedValueOnce(buildSession("sample-1"))
        .mockResolvedValueOnce(
          buildSession("sample-1", {
            lifecycle: "live_ended",
            playback: {
              mode: "live_tail",
              cursorTs: 1005,
              cursorEventSeq: 5,
              speed: 1,
              canSeek: true,
            },
            scheduling: {
              family: "turn",
              phase: "completed",
              acceptsHumanIntent: false,
            },
          }),
        ),
      loadTimeline: vi.fn().mockResolvedValue(buildTimelinePage("sample-1", [1, 2])),
    });
    const store = createArenaSessionStore(client);

    await store.loadSession({ sessionId: "sample-1" });
    await store.refreshSession?.();

    const state = store.getSnapshot();
    expect(state.timeline.events.map((event) => event.seq)).toEqual([1, 2]);
    expect(state.session?.lifecycle).toBe("live_ended");
    expect(state.session?.scheduling.acceptsHumanIntent).toBe(false);
    expect(state.session?.scheduling.phase).toBe("completed");
  });

  it("adopts the server-selected observer for later scene loads", async () => {
    const client = createStubClient({
      loadSession: vi.fn().mockResolvedValue(
        buildSession("sample-1", {
          observer: {
            observerId: "east",
            observerKind: "player",
          },
        }),
      ),
      loadScene: vi.fn().mockResolvedValue(
        buildScene(2, {
          observerPlayerId: "east",
        }),
      ),
    });
    const store = createArenaSessionStore(client);

    await store.loadSession({ sessionId: "sample-1" });
    await store.loadScene({ seq: 2 });

    expect(store.getSnapshot().sessionRequest?.observer).toEqual({
      observerId: "east",
      observerKind: "player",
    });
    expect(client.loadScene).toHaveBeenCalledWith({
      sessionId: "sample-1",
      runId: undefined,
      seq: 2,
      observer: {
        observerId: "east",
        observerKind: "player",
      },
    });
  });

  it("refreshes session scheduling after an accepted action receipt", async () => {
    const client = createStubClient({
      loadSession: vi
        .fn()
        .mockResolvedValueOnce(buildSession("sample-1"))
        .mockResolvedValueOnce(
          buildSession("sample-1", {
            scheduling: {
              family: "turn",
              phase: "advancing",
              acceptsHumanIntent: false,
              activeActorId: "player_1",
            },
          }),
        ),
      submitAction: vi.fn().mockResolvedValue({
        intentId: "intent-2",
        state: "accepted",
        relatedEventSeq: 6,
        reason: "queued",
      }),
    });
    const store = createArenaSessionStore(client);

    await store.loadSession({ sessionId: "sample-1" });
    await store.submitAction({
      playerId: "player_0",
      action: { move: "play_card", card: "3" },
    });

    const state = store.getSnapshot();
    expect(state.latestActionReceipt).toEqual({
      intentId: "intent-2",
      state: "accepted",
      relatedEventSeq: 6,
      reason: "queued",
    });
    expect(state.session?.scheduling.acceptsHumanIntent).toBe(false);
    expect(state.session?.scheduling.phase).toBe("advancing");
    expect(state.session?.scheduling.activeActorId).toBe("player_1");
  });

  it("does not block accepted realtime actions on a low-latency live refresh", async () => {
    const deferredRefresh = deferred<VisualSession>();
    const client = createStubClient({
      loadSession: vi
        .fn()
        .mockResolvedValueOnce(
          buildSession("sample-1", {
            gameId: "retro_platformer",
            pluginId: "arena.visualization.retro_platformer.frame_v1",
            scheduling: {
              family: "real_time_tick",
              phase: "waiting_for_intent",
              acceptsHumanIntent: true,
              activeActorId: "player_0",
            },
          }),
        )
        .mockImplementationOnce(() => deferredRefresh.promise),
      loadTimeline: vi.fn().mockResolvedValue(buildTimelinePage("sample-1", [1, 2])),
      loadScene: vi.fn().mockResolvedValue({
        sceneId: "scene-2",
        gameId: "retro_platformer",
        pluginId: "arena.visualization.retro_platformer.frame_v1",
        kind: "frame",
        tsMs: 1002,
        seq: 2,
        phase: "live",
        activePlayerId: "player_0",
        legalActions: [],
        summary: {},
        body: {
          status: {
            tick: 2,
          },
        },
        media: {
          primary: {
            mediaId: "live-channel-main",
            transport: "low_latency_channel",
            url: "/arena_visual/sessions/sample-1/media/live-channel-main/stream",
          },
        },
      }),
      submitAction: vi.fn().mockResolvedValue({
        intentId: "intent-live",
        state: "accepted",
        relatedEventSeq: 3,
        reason: "queued",
      }),
    });
    const store = createArenaSessionStore(client);

    await store.loadSession({ sessionId: "sample-1" });
    await store.loadScene({ seq: 2 });

    await expect(
      Promise.race([
        store.submitAction({
          playerId: "player_0",
          action: { move: "right" },
        }).then(() => "submitted"),
        deferredRefresh.promise.then(() => "refreshed"),
      ]),
    ).resolves.toBe("submitted");

    expect(store.getSnapshot().latestActionReceipt).toEqual({
      intentId: "intent-live",
      state: "accepted",
      relatedEventSeq: 3,
      reason: "queued",
    });
    expect(client.loadSession).toHaveBeenCalledTimes(2);
  });

  it("submits low-latency realtime actions without mutating latestActionReceipt", async () => {
    const client = createStubClient({
      submitActionLowLatency: vi.fn().mockResolvedValue(undefined),
    });
    const store = createArenaSessionStore(client);

    await store.loadSession({ sessionId: "sample-1", runId: "run-1" });

    await store.submitActionLowLatency({
      playerId: "player_0",
      action: {
        move: "right",
        metadata: {
          input_seq: 7,
        },
      },
    });

    expect(client.submitActionLowLatency).toHaveBeenCalledWith({
      sessionId: "sample-1",
      runId: "run-1",
      payload: {
        playerId: "player_0",
        action: {
          move: "right",
          metadata: {
            input_seq: 7,
          },
        },
      },
    });
    expect(store.getSnapshot().latestActionReceipt).toBeUndefined();
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

  it("keeps the previous scene mounted while a newer scene is loading", async () => {
    const nextScene = deferred<VisualScene>();
    const client = createStubClient({
      loadScene: vi
        .fn()
        .mockResolvedValueOnce(buildScene(2))
        .mockImplementationOnce(() => nextScene.promise),
    });
    const store = createArenaSessionStore(client);

    await store.loadSession({ sessionId: "sample-1" });
    await store.loadScene({ seq: 2 });

    const loading = store.loadScene({ seq: 9 });

    expect(store.getSnapshot().sceneStatus).toBe("loading");
    expect(store.getSnapshot().scene?.seq).toBe(2);

    nextScene.resolve(buildScene(9));
    await loading;

    expect(store.getSnapshot().sceneStatus).toBe("ready");
    expect(store.getSnapshot().scene?.seq).toBe(9);
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

  it("uses the playback cursor when loading a non-live-tail session", async () => {
    const client = createStubClient({
      loadSession: vi.fn().mockResolvedValue(
        buildSession("sample-1", {
          lifecycle: "live_ended",
          playback: {
            mode: "replay_playing",
            cursorTs: 1000,
            cursorEventSeq: 10,
            speed: 1,
            canSeek: true,
          },
        }),
      ),
      loadTimeline: vi.fn().mockResolvedValue(buildTimelinePage("sample-1", [5, 10, 15, 20])),
    });
    const store = createArenaSessionStore(client);

    await store.loadSession({ sessionId: "sample-1" });

    const state = store.getSnapshot();
    expect(state.session?.playback.mode).toBe("replay_playing");
    expect(state.currentSceneSeq).toBe(10);
  });

  it("advances replay playback locally until the tail and then pauses", async () => {
    const store = createArenaSessionStore(createStubClient());

    await store.loadSession({ sessionId: "sample-1" });
    await store.submitControl({
      commandType: "replay",
    });

    store.advanceReplayPlayback?.();
    let state = store.getSnapshot();
    expect(state.session?.playback.mode).toBe("replay_playing");
    expect(state.currentSceneSeq).toBe(2);

    store.advanceReplayPlayback?.();
    state = store.getSnapshot();
    expect(state.session?.playback.mode).toBe("paused");
    expect(state.currentSceneSeq).toBe(2);
  });

  it("advances replay playback across stable turn checkpoints instead of every raw event", async () => {
    const client = createStubClient({
      loadTimeline: vi.fn().mockResolvedValue(
        buildTypedTimelinePage("sample-1", [
          [1, "decision_window_open"],
          [2, "action_intent"],
          [3, "action_committed"],
          [4, "decision_window_close"],
          [5, "snapshot"],
          [6, "decision_window_open"],
          [7, "action_intent"],
          [8, "action_committed"],
          [9, "decision_window_close"],
          [10, "snapshot"],
          [11, "result"],
        ]),
      ),
      submitControl: vi.fn().mockResolvedValue({
        intentId: "control-1",
        state: "accepted",
        relatedEventSeq: 1,
      }),
    });
    const store = createArenaSessionStore(client);

    await store.loadSession({ sessionId: "sample-1" });
    await store.submitControl({
      commandType: "replay",
    });

    expect(store.getSnapshot().currentSceneSeq).toBe(1);

    store.advanceReplayPlayback?.();
    let state = store.getSnapshot();
    expect(state.session?.playback.mode).toBe("replay_playing");
    expect(state.currentSceneSeq).toBe(6);

    store.advanceReplayPlayback?.();
    state = store.getSnapshot();
    expect(state.session?.playback.mode).toBe("replay_playing");
    expect(state.currentSceneSeq).toBe(11);

    store.advanceReplayPlayback?.();
    state = store.getSnapshot();
    expect(state.session?.playback.mode).toBe("paused");
    expect(state.currentSceneSeq).toBe(11);
  });

  it("stores timeline filters and applies an accepted seek receipt to the current scene cursor", async () => {
    const client = createStubClient({
      submitControl: vi.fn().mockResolvedValue({
        intentId: "control-1",
        state: "accepted",
        relatedEventSeq: 9,
        reason: "seek_applied",
      }),
      loadTimeline: vi.fn().mockResolvedValue(buildTimelinePage("sample-1", [3, 5, 9])),
    });
    const store = createArenaSessionStore(client);

    await store.loadSession({ sessionId: "sample-1" });
    store.setTimelineFilters({
      eventTypes: ["action_intent"],
      severity: "warn",
      humanIntentOnly: true,
    });
    await store.submitControl({
      commandType: "seek_seq",
      targetSeq: 9,
    });

    const state = store.getSnapshot();
    expect(state.timeline.filters).toEqual({
      eventTypes: ["action_intent"],
      severity: "warn",
      humanIntentOnly: true,
    });
    expect(state.currentSceneSeq).toBe(9);
    expect(state.session?.playback.mode).toBe("paused");
    expect(state.session?.playback.cursorEventSeq).toBe(9);
    expect(client.submitControl).toHaveBeenCalledWith({
      sessionId: "sample-1",
      runId: undefined,
      payload: {
        commandType: "seek_seq",
        targetSeq: 9,
      },
    });
  });

  it("leaves the current scene cursor unchanged when a seek control is rejected", async () => {
    const client = createStubClient({
      submitControl: vi.fn().mockResolvedValue({
        intentId: "control-2",
        state: "rejected",
        relatedEventSeq: 9,
        reason: "seek_denied",
      }),
      loadTimeline: vi.fn().mockResolvedValue(buildTimelinePage("sample-1", [3, 5, 9])),
    });
    const store = createArenaSessionStore(client);

    await store.loadSession({ sessionId: "sample-1" });
    const beforeSeek = store.getSnapshot();

    await store.submitControl({
      commandType: "seek_seq",
      targetSeq: 9,
    });

    const afterSeek = store.getSnapshot();
    expect(afterSeek.currentSceneSeq).toBe(beforeSeek.currentSceneSeq);
    expect(afterSeek.session?.playback.mode).toBe(beforeSeek.session?.playback.mode);
    expect(afterSeek.session?.playback.cursorEventSeq).toBe(
      beforeSeek.session?.playback.cursorEventSeq,
    );
    expect(afterSeek.latestActionReceipt).toEqual({
      intentId: "control-2",
      state: "rejected",
      relatedEventSeq: 9,
      reason: "seek_denied",
    });
  });

  it("falls back to local replay navigation when replay control is rejected because the control channel is unavailable", async () => {
    const client = createStubClient({
      submitControl: vi.fn().mockResolvedValue({
        intentId: "control-3",
        state: "rejected",
        reason: "action_queue_not_available",
      }),
      loadTimeline: vi.fn().mockResolvedValue(buildTimelinePage("sample-1", [5, 10, 15, 20])),
    });
    const store = createArenaSessionStore(client);

    await store.loadSession({ sessionId: "sample-1" });
    expect(store.getSnapshot().currentSceneSeq).toBe(20);

    await store.submitControl({
      commandType: "replay",
    });

    const state = store.getSnapshot();
    expect(state.currentSceneSeq).toBe(5);
    expect(state.session?.playback.mode).toBe("replay_playing");
    expect(state.session?.playback.cursorEventSeq).toBe(5);
    expect(state.latestActionReceipt).toEqual({
      intentId: "control-3",
      state: "accepted",
      relatedEventSeq: 5,
      reason: "local_playback_fallback",
    });
  });

  it("rewinds to the related event seq when replay control is accepted by the backend", async () => {
    const client = createStubClient({
      submitControl: vi.fn().mockResolvedValue({
        intentId: "control-4",
        state: "accepted",
        relatedEventSeq: 5,
        reason: "playback_applied",
      }),
      loadTimeline: vi.fn().mockResolvedValue(buildTimelinePage("sample-1", [5, 10, 15, 20])),
    });
    const store = createArenaSessionStore(client);

    await store.loadSession({ sessionId: "sample-1" });
    expect(store.getSnapshot().currentSceneSeq).toBe(20);

    await store.submitControl({
      commandType: "replay",
    });

    const state = store.getSnapshot();
    expect(state.currentSceneSeq).toBe(5);
    expect(state.session?.playback.mode).toBe("replay_playing");
    expect(state.session?.playback.cursorEventSeq).toBe(5);
    expect(state.latestActionReceipt).toEqual({
      intentId: "control-4",
      state: "accepted",
      relatedEventSeq: 5,
      reason: "playback_applied",
    });
  });

  it("applies replay follow-up controls from backend receipts", async () => {
    const submitControl = vi
      .fn()
      .mockResolvedValueOnce({
        intentId: "control-pause",
        state: "accepted",
        relatedEventSeq: 10,
        reason: "playback_applied",
      })
      .mockResolvedValueOnce({
        intentId: "control-speed",
        state: "accepted",
        relatedEventSeq: 10,
        reason: "playback_applied",
      })
      .mockResolvedValueOnce({
        intentId: "control-step",
        state: "accepted",
        relatedEventSeq: 15,
        reason: "playback_applied",
      })
      .mockResolvedValueOnce({
        intentId: "control-end",
        state: "accepted",
        relatedEventSeq: 20,
        reason: "playback_applied",
      })
      .mockResolvedValueOnce({
        intentId: "control-tail",
        state: "accepted",
        relatedEventSeq: 20,
        reason: "playback_applied",
      });
    const client = createStubClient({
      submitControl,
      loadTimeline: vi.fn().mockResolvedValue(buildTimelinePage("sample-1", [5, 10, 15, 20])),
      loadSession: vi.fn().mockResolvedValue(
        buildSession("sample-1", {
          lifecycle: "live_ended",
          playback: {
            mode: "replay_playing",
            cursorTs: 1000,
            cursorEventSeq: 10,
            speed: 1,
            canSeek: true,
          },
        }),
      ),
    });
    const store = createArenaSessionStore(client);

    await store.loadSession({ sessionId: "sample-1" });

    await store.submitControl({
      commandType: "pause",
    });
    let state = store.getSnapshot();
    expect(state.session?.playback.mode).toBe("paused");
    expect(state.currentSceneSeq).toBe(10);

    await store.submitControl({
      commandType: "set_speed",
      speed: 2,
    });
    state = store.getSnapshot();
    expect(state.session?.playback.speed).toBe(2);
    expect(state.currentSceneSeq).toBe(10);

    await store.submitControl({
      commandType: "step",
      stepDelta: 1,
    });
    state = store.getSnapshot();
    expect(state.session?.playback.mode).toBe("paused");
    expect(state.currentSceneSeq).toBe(15);
    expect(state.session?.playback.cursorEventSeq).toBe(15);

    await store.submitControl({
      commandType: "seek_end",
    });
    state = store.getSnapshot();
    expect(state.session?.playback.mode).toBe("paused");
    expect(state.currentSceneSeq).toBe(20);

    await store.submitControl({
      commandType: "back_to_tail",
    });
    state = store.getSnapshot();
    expect(state.session?.playback.mode).toBe("live_tail");
    expect(state.currentSceneSeq).toBe(20);
    expect(state.session?.playback.cursorEventSeq).toBe(20);
  });

  it("ignores a stale control receipt after the host switches to a newer session", async () => {
    const staleControl = deferred<{
      intentId: string;
      state: "accepted";
      relatedEventSeq: number;
      reason: string;
    }>();
    const client = createStubClient({
      submitControl: vi
        .fn()
        .mockImplementationOnce(() => staleControl.promise)
        .mockResolvedValue({
          intentId: "control-2",
          state: "accepted",
          relatedEventSeq: 4,
          reason: "applied",
        }),
      loadSession: vi
        .fn()
        .mockResolvedValueOnce(buildSession("sample-1"))
        .mockResolvedValueOnce(buildSession("sample-2")),
      loadTimeline: vi
        .fn()
        .mockResolvedValueOnce(buildTimelinePage("sample-1", [1, 2]))
        .mockResolvedValueOnce(buildTimelinePage("sample-2", [3, 4])),
    });
    const store = createArenaSessionStore(client);

    await store.loadSession({ sessionId: "sample-1" });
    const controlRequest = store.submitControl({
      commandType: "seek_seq",
      targetSeq: 9,
    });
    await store.loadSession({ sessionId: "sample-2" });

    staleControl.resolve({
      intentId: "control-1",
      state: "accepted",
      relatedEventSeq: 9,
      reason: "seek_applied",
    });
    await controlRequest;

    const state = store.getSnapshot();
    expect(state.sessionRequest?.sessionId).toBe("sample-2");
    expect(state.session?.sessionId).toBe("sample-2");
    expect(state.currentSceneSeq).toBe(4);
    expect(state.session?.playback.cursorEventSeq).toBe(0);
    expect(state.latestActionReceipt).toBeUndefined();
  });

  it("resets timeline filters when loading a new session", async () => {
    const client = createStubClient({
      loadSession: vi
        .fn()
        .mockResolvedValueOnce(buildSession("sample-1"))
        .mockResolvedValueOnce(buildSession("sample-2")),
      loadTimeline: vi
        .fn()
        .mockResolvedValueOnce(buildTimelinePage("sample-1", [1, 2]))
        .mockResolvedValueOnce(buildTimelinePage("sample-2", [3, 4])),
    });
    const store = createArenaSessionStore(client);

    await store.loadSession({ sessionId: "sample-1" });
    store.setTimelineFilters({
      eventTypes: ["action_intent"],
      severity: "warn",
      humanIntentOnly: true,
    });

    const loading = store.loadSession({ sessionId: "sample-2" });
    expect(store.getSnapshot().timeline.filters).toEqual({
      eventTypes: [],
      severity: "all",
      humanIntentOnly: false,
    });
    await loading;

    expect(store.getSnapshot().timeline.filters).toEqual({
      eventTypes: [],
      severity: "all",
      humanIntentOnly: false,
    });
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

  it("submits chat through the gateway client and stores the latest host receipt", async () => {
    const client = createStubClient({
      submitChat: vi.fn().mockResolvedValue({
        intentId: "chat-1",
        state: "accepted",
        relatedEventSeq: 6,
        reason: "queued",
      }),
    });
    const store = createArenaSessionStore(client);

    await store.loadSession({ sessionId: "sample-1" });
    await store.submitChat({
      playerId: "player_0",
      text: "hello host",
    });

    expect(client.submitChat).toHaveBeenCalledWith({
      sessionId: "sample-1",
      runId: undefined,
      payload: {
        playerId: "player_0",
        text: "hello host",
      },
    });
    expect(store.getSnapshot().latestActionReceipt).toEqual({
      intentId: "chat-1",
      state: "accepted",
      relatedEventSeq: 6,
      reason: "queued",
    });
  });

  it("refreshes the current scene after an accepted chat receipt so stage bubbles can update", async () => {
    const client = createStubClient({
      loadSession: vi
        .fn()
        .mockResolvedValueOnce(
          buildSession("sample-1", {
            playback: {
              mode: "live_tail",
              cursorTs: 1004,
              cursorEventSeq: 4,
              speed: 1,
              canSeek: true,
            },
          }),
        )
        .mockResolvedValueOnce(
          buildSession("sample-1", {
            playback: {
              mode: "live_tail",
              cursorTs: 1005,
              cursorEventSeq: 4,
              speed: 1,
              canSeek: true,
            },
          }),
        ),
      loadScene: vi
        .fn()
        .mockResolvedValueOnce(buildScene(4))
        .mockResolvedValueOnce({
          ...buildScene(4, { observerPlayerId: "player_0" }),
          body: {
            seq: 4,
            status: {
              observerPlayerId: "player_0",
            },
            panels: {
              chatLog: [
                {
                  playerId: "player_0",
                  text: "hello host",
                },
              ],
            },
          },
        }),
      submitChat: vi.fn().mockResolvedValue({
        intentId: "chat-2",
        state: "accepted",
        relatedEventSeq: 4,
        reason: "queued",
      }),
    });
    const store = createArenaSessionStore(client);

    await store.loadSession({ sessionId: "sample-1" });
    await store.loadScene({ seq: 4 });
    await store.submitChat({
      playerId: "player_0",
      text: "hello host",
    });

    expect(client.loadSession).toHaveBeenCalledTimes(2);
    expect(client.loadScene).toHaveBeenCalledTimes(2);
    expect((store.getSnapshot().scene?.body as Record<string, unknown>)?.panels).toEqual({
      chatLog: [
        {
          playerId: "player_0",
          text: "hello host",
        },
      ],
    });
  });

  it("ignores a stale chat receipt after the host switches to a newer session", async () => {
    const staleChat = deferred<{
      intentId: string;
      state: "accepted";
      relatedEventSeq: number;
      reason: string;
    }>();
    const client = createStubClient({
      submitChat: vi
        .fn()
        .mockImplementationOnce(() => staleChat.promise)
        .mockResolvedValue({
          intentId: "chat-2",
          state: "accepted",
          relatedEventSeq: 4,
          reason: "queued",
        }),
      loadSession: vi
        .fn()
        .mockResolvedValueOnce(buildSession("sample-1"))
        .mockResolvedValueOnce(buildSession("sample-2")),
      loadTimeline: vi
        .fn()
        .mockResolvedValueOnce(buildTimelinePage("sample-1", [1, 2]))
        .mockResolvedValueOnce(buildTimelinePage("sample-2", [3, 4])),
    });
    const store = createArenaSessionStore(client);

    await store.loadSession({ sessionId: "sample-1" });
    const chatRequest = store.submitChat({
      playerId: "player_0",
      text: "hello host",
    });
    await store.loadSession({ sessionId: "sample-2" });

    staleChat.resolve({
      intentId: "chat-1",
      state: "accepted",
      relatedEventSeq: 9,
      reason: "queued",
    });
    await chatRequest;

    const state = store.getSnapshot();
    expect(state.sessionRequest?.sessionId).toBe("sample-2");
    expect(state.session?.sessionId).toBe("sample-2");
    expect(state.currentSceneSeq).toBe(4);
    expect(state.latestActionReceipt).toBeUndefined();
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

function buildTypedTimelinePage(
  sessionId: string,
  entries: Array<[number, TimelinePage["events"][number]["type"]]>,
): TimelinePage {
  return {
    sessionId,
    afterSeq: null,
    nextAfterSeq: entries.at(-1)?.[0] ?? null,
    limit: 50,
    hasMore: false,
    events: entries.map(([seq, type]) => ({
      seq,
      tsMs: 1000 + seq,
      type,
      label: `${type}-${seq}`,
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
    submitActionLowLatency: vi.fn().mockResolvedValue(undefined),
    submitChat: vi.fn().mockResolvedValue({
      intentId: "chat-1",
      state: "accepted",
    }),
    submitControl: vi.fn().mockResolvedValue({
      intentId: "control-1",
      state: "accepted",
    }),
    buildRealtimeActionSocketUrl: vi.fn().mockReturnValue(
      "ws://arena.local/arena_visual/sessions/sample-1/actions/ws",
    ),
    buildLiveUpdatesStreamUrl: vi.fn().mockReturnValue(
      "http://arena.local/arena_visual/sessions/sample-1/events",
    ),
    buildMediaUrl: vi.fn().mockReturnValue("/media/frame-1"),
    ...overrides,
  };
}

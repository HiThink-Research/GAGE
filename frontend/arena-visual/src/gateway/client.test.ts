import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  ArenaGatewayError,
  createArenaGatewayClient,
} from "./client";

function jsonResponse(payload: unknown, init?: ResponseInit): Response {
  return new Response(JSON.stringify(payload), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init,
  });
}

describe("createArenaGatewayClient", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("loads session and timeline page with optional runId", async () => {
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
          activeActorId: "player_0",
          windowId: "window-3",
        },
        capabilities: { supportsReplay: true },
        summary: { turn: 1 },
        timeline: { eventCount: 4, tailSeq: 4 },
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
            payload: {
              playerId: "player_0",
              action: { move: "fire" },
            },
          },
          {
            seq: 4,
            tsMs: 1004,
            type: "decision_window_close",
            label: "decision_window_close",
            payload: {
              windowId: "window-3",
              reason: "committed",
            },
          },
        ],
      }),
    );

    const client = createArenaGatewayClient({ baseUrl: "http://arena.local/" });
    const session = await client.loadSession({ sessionId: "sample-1", runId: "run-7" });
    const page = await client.loadTimeline({
      sessionId: "sample-1",
      afterSeq: 2,
      limit: 2,
      runId: "run-7",
    });

    expect(session.scheduling.phase).toBe("waiting_for_intent");
    expect(session.scheduling.acceptsHumanIntent).toBe(true);
    expect(page.events.map((event) => event.type)).toEqual([
      "action_committed",
      "decision_window_close",
    ]);
    expect(page.events[1]?.payload).toEqual({
      windowId: "window-3",
      reason: "committed",
    });
    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "http://arena.local/arena_visual/sessions/sample-1?run_id=run-7",
      expect.objectContaining({ method: "GET" }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "http://arena.local/arena_visual/sessions/sample-1/timeline?after_seq=2&limit=2&run_id=run-7",
      expect.objectContaining({ method: "GET" }),
    );
  });

  it("parses action intent receipt states", async () => {
    const fetchMock = vi.mocked(fetch);
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        intentId: "intent-9",
        state: "rejected",
        reason: "window_closed",
      }),
    );

    const client = createArenaGatewayClient({ baseUrl: "http://arena.local" });
    const receipt = await client.submitAction({
      sessionId: "sample-1",
      payload: {
        playerId: "player_0",
        action: { move: "fire" },
      },
    });

    expect(receipt).toEqual({
      intentId: "intent-9",
      state: "rejected",
      relatedEventSeq: undefined,
      reason: "window_closed",
    });
    expect(fetchMock).toHaveBeenCalledWith(
      "http://arena.local/arena_visual/sessions/sample-1/actions",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          playerId: "player_0",
          action: { move: "fire" },
        }),
      }),
    );
  });

  it("builds realtime input websocket URLs with optional runId", () => {
    const client = createArenaGatewayClient({ baseUrl: "https://arena.local/base/" });

    expect(
      client.buildRealtimeActionSocketUrl({
        sessionId: "sample-1",
        runId: "run-7",
      }),
    ).toBe("wss://arena.local/base/arena_visual/sessions/sample-1/actions/ws?run_id=run-7");
    expect(
      client.buildRealtimeActionSocketUrl({
        sessionId: "sample-2",
      }),
    ).toBe("wss://arena.local/base/arena_visual/sessions/sample-2/actions/ws");
  });

  it("builds live update stream URLs with observer and cursor context", () => {
    const client = createArenaGatewayClient({ baseUrl: "http://arena.local/base/" });

    expect(
      client.buildLiveUpdatesStreamUrl({
        sessionId: "sample-1",
        runId: "run-7",
        afterSeq: 12,
        observer: {
          observerId: "player_0",
          observerKind: "player",
        },
      }),
    ).toBe(
      "http://arena.local/base/arena_visual/sessions/sample-1/events?after_seq=12&observer_kind=player&observer_id=player_0&run_id=run-7",
    );
  });

  it("submits chat and control payloads to dedicated routes", async () => {
    const fetchMock = vi.mocked(fetch);
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        intentId: "chat-1",
        state: "accepted",
        reason: "queued",
      }),
    );
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        intentId: "control-1",
        state: "accepted",
        reason: "queued",
      }),
    );

    const client = createArenaGatewayClient({ baseUrl: "http://arena.local" });
    const chatReceipt = await client.submitChat({
      sessionId: "sample-1",
      payload: {
        playerId: "player_0",
        text: "hello",
      },
    });
    const controlReceipt = await client.submitControl({
      sessionId: "sample-1",
      payload: {
        commandType: "pause",
      },
    });

    expect(chatReceipt).toEqual({
      intentId: "chat-1",
      state: "accepted",
      relatedEventSeq: undefined,
      reason: "queued",
    });
    expect(controlReceipt).toEqual({
      intentId: "control-1",
      state: "accepted",
      relatedEventSeq: undefined,
      reason: "queued",
    });
    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "http://arena.local/arena_visual/sessions/sample-1/chat",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          playerId: "player_0",
          text: "hello",
        }),
      }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "http://arena.local/arena_visual/sessions/sample-1/control",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          commandType: "pause",
        }),
      }),
    );
  });

  it("throws stable gateway errors for non-ok responses", async () => {
    const fetchMock = vi.mocked(fetch);
    fetchMock.mockResolvedValueOnce(
      jsonResponse(
        {
          error: {
            code: "invalid_param",
            message: "limit must be positive",
          },
        },
        { status: 400 },
      ),
    );

    const client = createArenaGatewayClient({ baseUrl: "http://arena.local" });

    await expect(
      client.loadTimeline({
        sessionId: "sample-1",
        limit: 0,
      }),
    ).rejects.toEqual(
      new ArenaGatewayError({
        message: "limit must be positive",
        status: 400,
        code: "invalid_param",
      }),
    );
  });

  it("preserves stable string error codes from the Python gateway", async () => {
    const fetchMock = vi.mocked(fetch);
    fetchMock.mockResolvedValueOnce(
      jsonResponse(
        {
          error: "invalid_limit",
        },
        { status: 400 },
      ),
    );

    const client = createArenaGatewayClient({ baseUrl: "http://arena.local" });

    await expect(
      client.loadTimeline({
        sessionId: "sample-1",
        limit: 0,
      }),
    ).rejects.toEqual(
      new ArenaGatewayError({
        message: "invalid_limit",
        status: 400,
        code: "invalid_limit",
      }),
    );
  });

  it("carries observer override on session and scene reads", async () => {
    const fetchMock = vi.mocked(fetch);
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
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
        observer: { observerId: "player_0", observerKind: "player" },
        scheduling: {
          family: "turn",
          phase: "waiting_for_intent",
          acceptsHumanIntent: true,
        },
        capabilities: {},
        summary: {},
        timeline: { eventCount: 7 },
      }),
    );
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        sceneId: "sample-1:seq:7",
        gameId: "doudizhu",
        pluginId: "arena.visualization.doudizhu.table_v1",
        kind: "table",
        tsMs: 1007,
        seq: 7,
        phase: "replay",
        activePlayerId: "player_0",
        legalActions: [],
        summary: {},
        body: {},
      }),
    );

    const client = createArenaGatewayClient({ baseUrl: "http://arena.local" });
    await client.loadSession({
      sessionId: "sample-1",
      observer: { observerId: "player_0", observerKind: "player" },
    });
    await client.loadScene({
      sessionId: "sample-1",
      seq: 7,
      observer: { observerId: "player_0", observerKind: "player" },
    });

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "http://arena.local/arena_visual/sessions/sample-1?observer_kind=player&observer_id=player_0",
      expect.objectContaining({ method: "GET" }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "http://arena.local/arena_visual/sessions/sample-1/scene?seq=7&observer_kind=player&observer_id=player_0",
      expect.objectContaining({ method: "GET" }),
    );
  });

  it("builds media content urls separately from metadata reads", () => {
    const client = createArenaGatewayClient({ baseUrl: "http://arena.local" });

    const mediaUrl = client.buildMediaUrl({
      sessionId: "sample-1",
      mediaId: "frame-1",
      runId: "run-7",
    });

    expect(mediaUrl).toBe(
      "http://arena.local/arena_visual/sessions/sample-1/media/frame-1?content=1&run_id=run-7",
    );
  });
});

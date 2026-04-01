import { useEffect, useRef, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";

import { usePlaybackControls } from "../hooks/usePlaybackControls";
import {
  createArenaSessionStore,
  type ArenaControlPayload,
  type ArenaSessionStore,
} from "../store/arenaSessionStore";
import { useArenaStoreSelector } from "../store/useArenaStoreSelector";
import { createArenaGatewayClient } from "../../gateway/client";
import { createArenaLiveUpdateStream } from "../../gateway/liveUpdateStream";
import { createRealtimeInputSocket } from "../../gateway/realtimeInputSocket";
import {
  createArenaMediaResolver,
  type ArenaMediaResolver,
} from "../../gateway/media";
import type { ActionIntentReceipt } from "../../gateway/types";
import { resolveArenaPlugin } from "../../plugins/registry";
import { useInputBridge } from "../../plugins/sdk/useInputBridge";
import { GlobalControlBar } from "../../ui/controls/GlobalControlBar";
import { ArenaLayout } from "../../ui/layout/ArenaLayout";
import { SharedSidePanel } from "../../ui/panes/SharedSidePanel";
import { TimelineView } from "../../ui/timeline/TimelineView";

const LIVE_SESSION_REFRESH_POLL_MS = 150;
const LIVE_TIMELINE_POLL_MS = 150;
const LIVE_LOW_LATENCY_TIMELINE_POLL_MS = 500;
const POST_LIVE_AUTO_FINISH_MS = 15000;
const REPLAY_TICK_BASE_MS = 800;
const WIDE_STAGE_GAMES = new Set(["mahjong", "doudizhu"]);

function isStageFullscreenTarget(stageElement: HTMLElement | null): boolean {
  if (!stageElement || !document.fullscreenElement) {
    return false;
  }
  return (
    document.fullscreenElement === stageElement ||
    stageElement.contains(document.fullscreenElement) ||
    document.fullscreenElement.contains(stageElement)
  );
}

function isAcceptedReceipt(receipt: ActionIntentReceipt | undefined): boolean {
  return receipt !== undefined && ["accepted", "committed"].includes(receipt.state);
}

function readGatewayBaseUrl(): string {
  const envBaseUrl = import.meta.env.VITE_ARENA_GATEWAY_BASE_URL;
  if (typeof envBaseUrl === "string" && envBaseUrl.trim()) {
    return envBaseUrl.trim();
  }
  return window.location.origin;
}

function hasLowLatencyPrimaryMedia(scene: ReturnType<ArenaSessionStore["getSnapshot"]>["scene"]): boolean {
  return scene?.phase === "live" && scene.media?.primary?.transport === "low_latency_channel";
}

function canReuseLowLatencyLiveScene(
  session: ReturnType<ArenaSessionStore["getSnapshot"]>["session"],
  scene: ReturnType<ArenaSessionStore["getSnapshot"]>["scene"],
): boolean {
  return (
    session?.lifecycle === "live_running" &&
    session.playback.mode === "live_tail" &&
    hasLowLatencyPrimaryMedia(scene)
  );
}

function shouldCompactLiveWorkspace(
  session: ReturnType<ArenaSessionStore["getSnapshot"]>["session"],
  scene: ReturnType<ArenaSessionStore["getSnapshot"]>["scene"],
): boolean {
  return (
    session?.lifecycle === "live_running" &&
    session.playback.mode === "live_tail" &&
    session.scheduling.family === "real_time_tick" &&
    session.capabilities.supportsLiveUpdateStream === true &&
    (session.capabilities.supportsRealtimeInputWebSocket === true ||
      session.capabilities.supportsLowLatencyRealtimeInput === true) &&
    hasLowLatencyPrimaryMedia(scene)
  );
}

export function SessionPage() {
  const { sessionId } = useParams();
  const [searchParams] = useSearchParams();
  const runIdParam = searchParams.get("run_id")?.trim() || undefined;
  const [client] = useState(() =>
    createArenaGatewayClient({ baseUrl: readGatewayBaseUrl() }),
  );
  const [store] = useState(() => createArenaSessionStore(client));
  const [mediaResolver] = useState(() => createArenaMediaResolver(client));
  const sessionGameId = useArenaStoreSelector(
    store,
    (snapshot) => snapshot.session?.gameId ?? snapshot.scene?.gameId ?? "",
  );
  const session = useArenaStoreSelector(store, (snapshot) => snapshot.session);
  const scene = useArenaStoreSelector(store, (snapshot) => snapshot.scene);
  const arenaLayoutMode = WIDE_STAGE_GAMES.has(sessionGameId) ? "wide-stage" : "default";
  const compactLiveWorkspace = shouldCompactLiveWorkspace(session, scene);

  return (
    <main className="app-shell__body">
      <SessionRuntimeEffects
        client={client}
        runIdParam={runIdParam}
        sessionId={sessionId}
        store={store}
      />

      <section className="hero-panel">
        <p className="eyebrow">Session Workspace</p>
        <h1>{sessionId ?? "Unknown session"}</h1>
        <p className="hero-copy">
          The arena host is rendering the active plugin, synchronized timeline,
          shared controls, and observer-aware inspector for this session.
        </p>
      </section>

      <ArenaLayout
        layoutMode={arenaLayoutMode}
        controls={<SessionControls store={store} />}
        stage={<SessionStage client={client} mediaResolver={mediaResolver} store={store} />}
        timeline={compactLiveWorkspace ? null : <SessionTimeline store={store} />}
        sidePanel={compactLiveWorkspace ? null : <SessionSidePanelRegion store={store} />}
      />

      <div className="app-shell__footer">
        <Link className="app-shell__nav-link" to="/">
          Back to host
        </Link>
      </div>
    </main>
  );
}

function SessionRuntimeEffects({
  client,
  sessionId,
  runIdParam,
  store,
}: {
  client: ReturnType<typeof createArenaGatewayClient>;
  sessionId?: string;
  runIdParam?: string;
  store: ArenaSessionStore;
}) {
  const status = useArenaStoreSelector(store, (snapshot) => snapshot.status);
  const sceneStatus = useArenaStoreSelector(store, (snapshot) => snapshot.sceneStatus);
  const currentSceneSeq = useArenaStoreSelector(store, (snapshot) => snapshot.currentSceneSeq);
  const sceneSeq = useArenaStoreSelector(store, (snapshot) => snapshot.scene?.seq);
  const session = useArenaStoreSelector(store, (snapshot) => snapshot.session);
  const scene = useArenaStoreSelector(store, (snapshot) => snapshot.scene);
  const playbackMode = useArenaStoreSelector(
    store,
    (snapshot) => snapshot.session?.playback.mode ?? "live_tail",
  );
  const observerId = useArenaStoreSelector(
    store,
    (snapshot) =>
      snapshot.sessionRequest?.observer?.observerId ?? snapshot.session?.observer.observerId ?? "",
  );
  const observerKind = useArenaStoreSelector(
    store,
    (snapshot) =>
      snapshot.sessionRequest?.observer?.observerKind ?? snapshot.session?.observer.observerKind ?? "spectator",
  );
  const supportsLiveUpdateStream = useArenaStoreSelector(
    store,
    (snapshot) => snapshot.session?.capabilities.supportsLiveUpdateStream === true,
  );
  const playbackSpeed = useArenaStoreSelector(
    store,
    (snapshot) => snapshot.session?.playback.speed ?? 1,
  );

  useEffect(() => {
    if (!sessionId) {
      return;
    }

    void store.loadSession({ sessionId, runId: runIdParam });
  }, [runIdParam, sessionId, store]);

  useEffect(() => {
    if (
      status !== "ready" ||
      sceneStatus === "loading" ||
      currentSceneSeq === undefined ||
      sceneSeq === currentSceneSeq ||
      canReuseLowLatencyLiveScene(session, scene)
    ) {
      return;
    }

    void store.loadScene({ seq: currentSceneSeq }).catch(() => {});
  }, [currentSceneSeq, scene, sceneSeq, sceneStatus, session, status, store]);

  useEffect(() => {
    if (
      status !== "ready" ||
      playbackMode !== "live_tail" ||
      session?.lifecycle !== "live_running" ||
      !supportsLiveUpdateStream ||
      typeof client.buildLiveUpdatesStreamUrl !== "function" ||
      typeof store.applyLiveSession !== "function" ||
      typeof store.applyTimelinePage !== "function" ||
      !sessionId
    ) {
      return;
    }

    const current = store.getSnapshot();
    const streamUrl = client.buildLiveUpdatesStreamUrl({
      sessionId,
      runId: runIdParam,
      afterSeq: current.timeline.events.at(-1)?.seq ?? current.timeline.nextAfterSeq ?? undefined,
      observer: current.sessionRequest?.observer ?? current.session?.observer,
    });
    const stream = createArenaLiveUpdateStream({
      url: streamUrl,
      onDelta: ({ session: nextSession, timeline }) => {
        if (nextSession) {
          store.applyLiveSession?.(nextSession);
        }
        if (timeline) {
          store.applyTimelinePage?.(timeline);
        }
      },
      onError: () => {
        void store.refreshSession?.().catch(() => {});
      },
    });

    return () => {
      stream.close();
    };
  }, [
    client,
    observerId,
    observerKind,
    playbackMode,
    runIdParam,
    session?.lifecycle,
    sessionId,
    status,
    store,
    supportsLiveUpdateStream,
  ]);

  useEffect(() => {
    if (
      status !== "ready" ||
      playbackMode !== "live_tail" ||
      supportsLiveUpdateStream
    ) {
      return;
    }

    let lastTimelinePollAtMs = 0;
    const timer = window.setInterval(() => {
      const current = store.getSnapshot();
      if (
        current.status !== "ready" ||
        current.session?.playback.mode !== "live_tail" ||
        current.session === undefined
      ) {
        return;
      }
      const refreshTasks: Array<Promise<unknown>> = [
        store.refreshSession?.() ?? Promise.resolve(),
      ];
      const lowLatencyLive = canReuseLowLatencyLiveScene(current.session, current.scene);
      const nowMs = Date.now();
      const timelinePollIntervalMs = lowLatencyLive
        ? LIVE_LOW_LATENCY_TIMELINE_POLL_MS
        : LIVE_TIMELINE_POLL_MS;
      const shouldPollTimeline =
        current.timeline.status !== "loading" &&
        (lastTimelinePollAtMs === 0 || nowMs - lastTimelinePollAtMs >= timelinePollIntervalMs);
      if (shouldPollTimeline) {
        lastTimelinePollAtMs = nowMs;
        refreshTasks.push(store.loadMoreTimeline());
      }
      void Promise.allSettled(refreshTasks);
    }, LIVE_SESSION_REFRESH_POLL_MS);

    return () => {
      window.clearInterval(timer);
    };
  }, [playbackMode, status, store, supportsLiveUpdateStream]);

  useEffect(() => {
    if (status !== "ready" || playbackMode !== "replay_playing") {
      return;
    }

    const timer = window.setInterval(() => {
      const current = store.getSnapshot();
      if (
        current.status !== "ready" ||
        current.session?.playback.mode !== "replay_playing" ||
        current.sceneStatus === "loading"
      ) {
        return;
      }
      store.advanceReplayPlayback?.();
    }, Math.max(120, Math.round(REPLAY_TICK_BASE_MS / Math.max(playbackSpeed, 0.25))));

    return () => {
      window.clearInterval(timer);
    };
  }, [playbackMode, playbackSpeed, status, store]);

  return null;
}

function SessionControls({ store }: { store: ArenaSessionStore }) {
  const status = useArenaStoreSelector(store, (snapshot) => snapshot.status);
  const session = useArenaStoreSelector(store, (snapshot) => snapshot.session);
  const currentSceneSeq = useArenaStoreSelector(
    store,
    (snapshot) => snapshot.currentSceneSeq,
  );
  const timelineEvents = useArenaStoreSelector(store, (snapshot) => snapshot.timeline.events);
  const playbackControls = usePlaybackControls(store);
  const [manualFinishRequired, setManualFinishRequired] = useState(false);
  const [autoFinishStartedAtMs, setAutoFinishStartedAtMs] = useState<number | null>(null);
  const [nowMs, setNowMs] = useState(() => Date.now());
  const [pendingCommand, setPendingCommand] = useState<ArenaControlPayload["commandType"] | null>(
    null,
  );
  const [finishRequested, setFinishRequested] = useState(false);
  const [closeRequested, setCloseRequested] = useState(false);
  const sessionKey = `${session?.sessionId ?? "unknown"}:${session?.lifecycle ?? "idle"}`;
  const resultSeen = timelineEvents.some((event) => event.type === "result");
  const isPostLiveWindow =
    session !== undefined &&
    session.lifecycle !== "closed" &&
    (session.lifecycle === "live_ended" || resultSeen);
  const canStopSession = session !== undefined && session.lifecycle !== "closed";
  const liveSessionStillRunning = session?.lifecycle === "live_running";
  const canRestartRound =
    liveSessionStillRunning && session?.capabilities.supportsRestart === true;
  const playbackMode = session?.playback.mode ?? "live_tail";
  const hasTimeline = timelineEvents.length > 0;
  const canSeek = session?.playback.canSeek ?? false;
  const headSeq = timelineEvents[0]?.seq;
  const tailSeq = timelineEvents.at(-1)?.seq ?? session?.playback.cursorEventSeq;
  const isAtHead =
    currentSceneSeq !== undefined && headSeq !== undefined && currentSceneSeq <= headSeq;
  const isAtTail =
    currentSceneSeq !== undefined && tailSeq !== undefined && currentSceneSeq >= tailSeq;
  const controlsLocked = pendingCommand !== null || finishRequested;

  useEffect(() => {
    setManualFinishRequired(false);
    setAutoFinishStartedAtMs(null);
    setNowMs(Date.now());
    setPendingCommand(null);
    setFinishRequested(false);
    setCloseRequested(false);
  }, [sessionKey]);

  useEffect(() => {
    if (!isPostLiveWindow) {
      setManualFinishRequired(false);
      setAutoFinishStartedAtMs(null);
      return;
    }
    if (manualFinishRequired || autoFinishStartedAtMs !== null) {
      return;
    }
    const startedAt = Date.now();
    setAutoFinishStartedAtMs(startedAt);
    setNowMs(startedAt);
  }, [autoFinishStartedAtMs, isPostLiveWindow, manualFinishRequired]);

  useEffect(() => {
    if (!isPostLiveWindow || manualFinishRequired || autoFinishStartedAtMs === null) {
      return;
    }
    const timer = window.setInterval(() => {
      setNowMs(Date.now());
    }, 250);
    return () => {
      window.clearInterval(timer);
    };
  }, [autoFinishStartedAtMs, isPostLiveWindow, manualFinishRequired]);

  useEffect(() => {
    if (!finishRequested || status !== "ready" || !store.refreshSession) {
      return;
    }

    void store.refreshSession().catch(() => {});

    const timer = window.setInterval(() => {
      const current = store.getSnapshot();
      if (
        current.status !== "ready" ||
        current.session?.lifecycle === "closed"
      ) {
        return;
      }
      void store.refreshSession?.().catch(() => {});
    }, LIVE_TIMELINE_POLL_MS);

    return () => {
      window.clearInterval(timer);
    };
  }, [finishRequested, status, store]);

  useEffect(() => {
    if (!finishRequested || closeRequested || session?.lifecycle !== "closed") {
      return;
    }
    setCloseRequested(true);
    try {
      window.open("", "_self");
    } catch {
      // Best-effort browser close only.
    }
    try {
      window.close();
    } catch {
      // Best-effort browser close only.
    }
  }, [closeRequested, finishRequested, session?.lifecycle]);

  async function runControl(
    commandType: ArenaControlPayload["commandType"],
    action: () => Promise<ActionIntentReceipt>,
  ): Promise<void> {
    if (controlsLocked) {
      return;
    }

    setPendingCommand(commandType);
    let keepLocked = false;

    try {
      const receipt = await action();
      if (commandType === "finish") {
        if (isAcceptedReceipt(receipt)) {
          setFinishRequested(true);
          keepLocked = true;
        }
        return;
      }
      if (isPostLiveWindow && isAcceptedReceipt(receipt)) {
        setManualFinishRequired(true);
      }
    } catch {
      keepLocked = false;
    } finally {
      if (!keepLocked) {
        setPendingCommand(null);
      }
    }
  }

  let postLiveStatusLabel: string | undefined;
  if (finishRequested) {
    postLiveStatusLabel = liveSessionStillRunning ? "Stopping session..." : "Finishing session...";
  } else if (isPostLiveWindow) {
    if (manualFinishRequired) {
      postLiveStatusLabel = "Replay active · click Finish to exit";
    } else if (autoFinishStartedAtMs !== null) {
      const remainingMs = Math.max(
        0,
        POST_LIVE_AUTO_FINISH_MS - (nowMs - autoFinishStartedAtMs),
      );
      postLiveStatusLabel = `Auto finish in ${Math.max(1, Math.ceil(remainingMs / 1000))}s`;
    }
  }

  const controlAvailability = {
    playLiveDisabled: !canSeek || playbackMode === "live_tail",
    pauseDisabled: playbackMode === "paused",
    replayDisabled: !canSeek || playbackMode === "replay_playing",
    restartDisabled: !canRestartRound || finishRequested,
    speedDisabled: playbackMode === "live_tail",
    stepBackwardDisabled:
      !canSeek || playbackMode === "live_tail" || (hasTimeline && isAtHead),
    stepForwardDisabled:
      !canSeek || playbackMode === "live_tail" || (hasTimeline && isAtTail),
    seekEndDisabled:
      !canSeek || playbackMode === "live_tail" || (hasTimeline && isAtTail),
    backToTailDisabled: !canSeek || playbackMode === "live_tail",
    finishDisabled: finishRequested,
  };

  return (
    <GlobalControlBar
      playbackMode={playbackMode}
      playbackSpeed={session?.playback.speed ?? 1}
      disabled={status !== "ready" || session === undefined || controlsLocked}
      scheduling={session?.scheduling}
      postLiveStatusLabel={postLiveStatusLabel}
      controlAvailability={controlAvailability}
      finishLabel={
        finishRequested
          ? liveSessionStillRunning
            ? "Stopping..."
            : "Finishing..."
          : liveSessionStillRunning
            ? "Stop"
            : "Finish"
      }
      onPause={() => {
        void runControl("pause", () => playbackControls.pause());
      }}
      onPlayLive={() => {
        void runControl("follow_tail", () => playbackControls.playLive());
      }}
      onReplay={() => {
        void runControl("replay", () => playbackControls.playReplay());
      }}
      onRestart={
        canRestartRound
          ? () => {
              void runControl("restart", () => playbackControls.restart());
            }
          : undefined
      }
      onSetSpeed={(speed) => {
        void runControl("set_speed", () => playbackControls.setSpeed(speed));
      }}
      onStep={(delta) => {
        if (delta > 0) {
          void runControl("step", () => playbackControls.stepForward());
          return;
        }
        void runControl("step", () => playbackControls.stepBackward());
      }}
      onSeekEnd={() => {
        void runControl("seek_end", () => playbackControls.seekEnd());
      }}
      onBackToTail={() => {
        void runControl("back_to_tail", () => playbackControls.backToTail());
      }}
      onFinish={
        canStopSession
          ? () => {
              void runControl("finish", () => playbackControls.finish());
            }
          : undefined
      }
    />
  );
}

function SessionStage({
  client,
  store,
  mediaResolver,
}: {
  client: ReturnType<typeof createArenaGatewayClient>;
  store: ArenaSessionStore;
  mediaResolver: ArenaMediaResolver;
}) {
  const stageRef = useRef<HTMLElement | null>(null);
  const status = useArenaStoreSelector(store, (snapshot) => snapshot.status);
  const error = useArenaStoreSelector(store, (snapshot) => snapshot.error);
  const sessionRequest = useArenaStoreSelector(store, (snapshot) => snapshot.sessionRequest);
  const session = useArenaStoreSelector(store, (snapshot) => snapshot.session);
  const scene = useArenaStoreSelector(store, (snapshot) => snapshot.scene);
  const sceneStatus = useArenaStoreSelector(store, (snapshot) => snapshot.sceneStatus);
  const currentSceneSeq = useArenaStoreSelector(
    store,
    (snapshot) => snapshot.currentSceneSeq,
  );
  const latestActionReceipt = useArenaStoreSelector(
    store,
    (snapshot) => snapshot.latestActionReceipt,
  );
  const plugin =
    session !== undefined ? resolveArenaPlugin(session.pluginId) : undefined;
  const inputBridge = useInputBridge({
    latestReceipt: latestActionReceipt,
    submitAction: store.submitAction,
    interpreter: plugin?.inputInterpreter,
  });
  const PluginView = plugin?.render;
  const playbackMode = session?.playback.mode ?? "live_tail";
  const [isFullscreen, setIsFullscreen] = useState(false);
  const previousPlaybackModeRef = useRef(playbackMode);
  const [hardTransitionTargetSeq, setHardTransitionTargetSeq] = useState<number | null>(null);
  const realtimeInputSocketRef = useRef<ReturnType<typeof createRealtimeInputSocket> | null>(null);
  const reusableLowLatencyLiveScene = canReuseLowLatencyLiveScene(session, scene);
  const hasSceneMismatch =
    session !== undefined &&
    scene !== undefined &&
    currentSceneSeq !== undefined &&
    scene.seq !== currentSceneSeq &&
    !reusableLowLatencyLiveScene;

  useEffect(() => {
    const previousPlaybackMode = previousPlaybackModeRef.current;
    if (
      playbackMode === "replay_playing" &&
      previousPlaybackMode !== "replay_playing" &&
      hasSceneMismatch &&
      currentSceneSeq !== undefined
    ) {
      setHardTransitionTargetSeq(currentSceneSeq);
    }
    previousPlaybackModeRef.current = playbackMode;
  }, [currentSceneSeq, hasSceneMismatch, playbackMode]);

  useEffect(() => {
    if (playbackMode !== "replay_playing") {
      setHardTransitionTargetSeq(null);
      return;
    }
    if (
      hardTransitionTargetSeq !== null &&
      scene !== undefined &&
      currentSceneSeq !== undefined &&
      scene.seq === hardTransitionTargetSeq &&
      currentSceneSeq === hardTransitionTargetSeq
    ) {
      setHardTransitionTargetSeq(null);
    }
  }, [currentSceneSeq, hardTransitionTargetSeq, playbackMode, scene]);

  useEffect(() => {
    const syncFullscreenState = () => {
      setIsFullscreen(isStageFullscreenTarget(stageRef.current));
    };

    document.addEventListener("fullscreenchange", syncFullscreenState);
    return () => {
      document.removeEventListener("fullscreenchange", syncFullscreenState);
    };
  }, []);

  const useRealtimeInputSocketPath =
    session?.capabilities.supportsRealtimeInputWebSocket === true &&
    plugin?.inputInterpreter !== undefined;
  const useLowLatencyRealtimeInputPath =
    !useRealtimeInputSocketPath &&
    session?.capabilities.supportsLowLatencyRealtimeInput === true &&
    plugin?.inputInterpreter !== undefined;

  useEffect(() => {
    realtimeInputSocketRef.current?.close();
    realtimeInputSocketRef.current = null;
    if (!useRealtimeInputSocketPath || !sessionRequest) {
      return;
    }
    const socket = createRealtimeInputSocket({
      url: client.buildRealtimeActionSocketUrl({
        sessionId: sessionRequest.sessionId,
        runId: sessionRequest.runId,
      }),
    });
    realtimeInputSocketRef.current = socket;
    return () => {
      socket.close();
      if (realtimeInputSocketRef.current === socket) {
        realtimeInputSocketRef.current = null;
      }
    };
  }, [client, sessionRequest, useRealtimeInputSocketPath]);

  const handleToggleFullscreen = async () => {
    const stageElement = stageRef.current;
    if (!stageElement) {
      return;
    }

    try {
      if (isStageFullscreenTarget(stageElement)) {
        await document.exitFullscreen?.();
        return;
      }
      await stageElement.requestFullscreen?.();
    } catch {
      // Fullscreen is a convenience control, not a hard dependency.
    }
  };

  const showHardSceneTransition =
    hardTransitionTargetSeq !== null &&
    currentSceneSeq !== undefined &&
    hardTransitionTargetSeq === currentSceneSeq &&
    hasSceneMismatch;
  const showSoftSceneTransition =
    !showHardSceneTransition &&
    hasSceneMismatch &&
    sceneStatus === "loading" &&
    playbackMode === "paused";
  const showSceneTransition = showHardSceneTransition || showSoftSceneTransition;
  const transitionLabel = showHardSceneTransition
    ? "Loading replay scene..."
    : "Syncing scene...";
  const isImmersivePlugin =
    session?.pluginId === "arena.visualization.retro_platformer.frame_v1";
  const submitPluginInput = useRealtimeInputSocketPath
    ? async (event: unknown): Promise<void> => {
        if (!plugin?.inputInterpreter) {
          throw new Error("Plugin input interpreter is not available.");
        }
        const payload =
          plugin.inputInterpreter.interpret(event as never) as Record<string, unknown>;
        const realtimeSocket = realtimeInputSocketRef.current;
        if (realtimeSocket) {
          await realtimeSocket.submit(payload);
          return;
        }
        await store.submitActionLowLatency(payload);
      }
    : useLowLatencyRealtimeInputPath
    ? async (event: unknown): Promise<void> => {
        if (!plugin?.inputInterpreter) {
          throw new Error("Plugin input interpreter is not available.");
        }
        await store.submitActionLowLatency(
          plugin.inputInterpreter.interpret(event as never) as Record<string, unknown>,
        );
      }
    : inputBridge.submitInput;
  const showStageChrome = !(isFullscreen && isImmersivePlugin);

  if (session && plugin && PluginView) {
    return (
      <section
        className={[
          "session-stage",
          isFullscreen ? "session-stage--fullscreen" : "",
          isImmersivePlugin ? "session-stage--immersive" : "",
          showSceneTransition ? "is-transitioning" : "",
          showHardSceneTransition ? "is-hard-transition" : "",
        ]
          .filter((value) => value !== "")
          .join(" ")}
        ref={stageRef}
        aria-busy={showSceneTransition}
      >
        {showStageChrome ? (
          <div className="session-stage__chrome">
            <button
              className="session-stage__fullscreen-button"
              onClick={() => {
                void handleToggleFullscreen();
              }}
              type="button"
              aria-pressed={isFullscreen}
            >
              {isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
            </button>
          </div>
        ) : null}
        <div className="session-stage__surface">
          <PluginView
            session={session}
            scene={scene}
            latestActionReceipt={inputBridge.latestReceipt}
            submitAction={inputBridge.submitAction}
            submitInput={submitPluginInput}
            mediaSubscribe={mediaResolver.subscribe}
            isFallback={plugin.isFallback}
            requestedPluginId={plugin.requestedPluginId}
          />
        </div>
        {showSceneTransition ? (
          <div className="session-stage__overlay" role="status" aria-live="polite">
            <div className="session-stage__overlay-card">
              <p className="eyebrow">Scene Sync</p>
              <h2>{transitionLabel}</h2>
              <p className="plugin-stage-card__copy">
                Keeping the last stable scene mounted until the requested frame is ready.
              </p>
            </div>
          </div>
        ) : null}
      </section>
    );
  }

  return (
    <section className="plugin-stage-card">
      <p className="eyebrow">Host State</p>
      <h2>{status === "loading" ? "Loading session..." : "No session loaded"}</h2>
      <p className="plugin-stage-card__copy">
        {status === "error"
          ? error ?? "The session failed to load."
          : "Open a valid session id to populate the visual host."}
      </p>
    </section>
  );
}

function SessionTimeline({ store }: { store: ArenaSessionStore }) {
  const timeline = useArenaStoreSelector(store, (snapshot) => snapshot.timeline);
  const currentSceneSeq = useArenaStoreSelector(store, (snapshot) => snapshot.currentSceneSeq);
  const playbackControls = usePlaybackControls(store);

  return (
    <TimelineView
      events={timeline.events}
      filters={timeline.filters}
      currentSeq={currentSceneSeq}
      status={timeline.status}
      hasMore={timeline.hasMore}
      onSelectEvent={(seq) => {
        void playbackControls.selectEvent(seq);
      }}
      onLoadMore={() => {
        void playbackControls.loadMoreTimeline();
      }}
      onFiltersChange={(filters) => {
        store.setTimelineFilters(filters);
      }}
    />
  );
}

function SessionSidePanelRegion({ store }: { store: ArenaSessionStore }) {
  const session = useArenaStoreSelector(store, (snapshot) => snapshot.session);
  const scene = useArenaStoreSelector(store, (snapshot) => snapshot.scene);
  const latestActionReceipt = useArenaStoreSelector(
    store,
    (snapshot) => snapshot.latestActionReceipt,
  );
  const error = useArenaStoreSelector(store, (snapshot) => snapshot.error);

  return (
    <SharedSidePanel
      session={session}
      scene={scene}
      latestActionReceipt={latestActionReceipt}
      error={error}
      onObserverChange={(observer) => {
        void store.setObserver(observer).catch(() => {});
      }}
      onChatSubmit={(payload) => store.submitChat(payload)}
    />
  );
}

import {
  startTransition,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
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
import type { ActionIntentReceipt, PlaybackMode } from "../../gateway/types";
import {
  isRecord,
  readOptionalNumber,
  readTrimmedString,
} from "../../lib/sceneReaders";
import { resolveArenaPlugin } from "../../plugins/registry";
import { useInputBridge } from "../../plugins/sdk/useInputBridge";
import { GlobalControlBar } from "../../ui/controls/GlobalControlBar";
import { ArenaLayout } from "../../ui/layout/ArenaLayout";
import {
  SharedSidePanel,
  type SharedSidePanelControlPanel,
  type SharedSidePanelTab,
} from "../../ui/panes/SharedSidePanel";
import { TimelineView } from "../../ui/timeline/TimelineView";

const LIVE_SESSION_REFRESH_POLL_MS = 150;
const LIVE_TIMELINE_POLL_MS = 150;
const LIVE_LOW_LATENCY_TIMELINE_POLL_MS = 500;
const POST_LIVE_AUTO_FINISH_MS = 15000;
const REPLAY_TICK_BASE_MS = 800;
const DEFAULT_SIDE_PANEL_TAB: SharedSidePanelTab = "Players";
const SIDE_PANEL_TABS: Array<{
  tab: SharedSidePanelTab;
  label: string;
  cue: string;
}> = [
  { tab: "Control", label: "Control", cue: "Deck" },
  { tab: "Players", label: "Players", cue: "Roster" },
  { tab: "Events", label: "Events", cue: "Pulse" },
  { tab: "Chat", label: "Chat", cue: "Comms" },
  { tab: "Trace", label: "Trace", cue: "Logs" },
];

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

function shallowEqualRecord<T extends Record<string, unknown>>(left: T, right: T): boolean {
  if (Object.is(left, right)) {
    return true;
  }
  const leftKeys = Object.keys(left) as Array<keyof T>;
  const rightKeys = Object.keys(right) as Array<keyof T>;
  if (leftKeys.length !== rightKeys.length) {
    return false;
  }
  return leftKeys.every((key) => Object.is(left[key], right[key]));
}

function formatOutcomeLabel(value: string): string {
  return value
    .split(/[_\-\s]+/)
    .filter((part) => part !== "")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1).toLowerCase())
    .join(" ");
}

interface StageResultBanner {
  eyebrow: string;
  title: string;
  details: string[];
}

function readStageResultBanner(
  session: ReturnType<ArenaSessionStore["getSnapshot"]>["session"],
): StageResultBanner | null {
  const summary = session?.summary;
  if (!isRecord(summary)) {
    return null;
  }

  const rawResult = isRecord(summary.result) ? summary.result : summary;
  const winner =
    readTrimmedString(rawResult.winner) ??
    readTrimmedString(rawResult.winner_player_id) ??
    readTrimmedString(rawResult.winnerPlayerId);
  const resultValue = readTrimmedString(rawResult.result)?.toLowerCase() ?? null;
  const reason =
    readTrimmedString(rawResult.reason) ??
    readTrimmedString(rawResult.termination_reason) ??
    readTrimmedString(rawResult.terminationReason);
  const moveCount = readOptionalNumber(rawResult.move_count) ?? readOptionalNumber(rawResult.moveCount);

  let title: string | null = null;
  if (winner && (resultValue === "win" || resultValue === null)) {
    title = `${winner} wins`;
  } else if (winner && resultValue) {
    title = `${winner} ${formatOutcomeLabel(resultValue)}`;
  } else if (resultValue === "draw") {
    title = "Draw";
  } else if (resultValue) {
    title = formatOutcomeLabel(resultValue);
  } else if (
    session?.scheduling.phase === "completed" ||
    session?.lifecycle === "live_ended" ||
    session?.lifecycle === "closed"
  ) {
    title = "Match complete";
  }

  if (!title) {
    return null;
  }

  const details = [
    reason ? formatOutcomeLabel(reason) : null,
    moveCount && moveCount > 0 ? `${moveCount} ${moveCount === 1 ? "move" : "moves"}` : null,
  ].filter((entry): entry is string => entry !== null);

  return {
    eyebrow: "Match complete",
    title,
    details,
  };
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
  const sessionOverview = useArenaStoreSelector(
    store,
    (snapshot) => ({
      session: snapshot.session,
      scene: snapshot.scene,
      currentSceneSeq: snapshot.currentSceneSeq,
      timeline: snapshot.timeline,
    }),
    shallowEqualRecord,
  );
  const { session, scene, currentSceneSeq, timeline } = sessionOverview;
  const sessionPlugin =
    session !== undefined ? resolveArenaPlugin(session.pluginId) : undefined;
  const arenaLayoutMode = sessionPlugin?.manifest.layoutMode ?? "default";
  const compactLiveWorkspace = shouldCompactLiveWorkspace(session, scene);
  const playbackMode = session?.playback.mode ?? "live_tail";
  const [timelineExpanded, setTimelineExpanded] = useState(playbackMode !== "live_tail");
  const [controlsExpanded, setControlsExpanded] = useState(false);
  const [activeSidePanelTab, setActiveSidePanelTab] =
    useState<SharedSidePanelTab>(DEFAULT_SIDE_PANEL_TAB);
  const [sidePanelOpen, setSidePanelOpen] = useState(false);

  useEffect(() => {
    startTransition(() => {
      setTimelineExpanded(playbackMode !== "live_tail");
    });
  }, [playbackMode, sessionId]);

  useEffect(() => {
    startTransition(() => {
      setActiveSidePanelTab(DEFAULT_SIDE_PANEL_TAB);
      setSidePanelOpen(false);
    });
    setControlsExpanded(false);
  }, [sessionId]);

  const eventCounts = useMemo(
    () => ({
      total: timeline.events.length,
      warn: timeline.events.filter((event) => event.severity === "warn").length,
      critical: timeline.events.filter((event) => event.severity === "critical").length,
    }),
    [timeline.events],
  );

  const handleSidePanelToggle = (tab: SharedSidePanelTab) => {
    startTransition(() => {
      setActiveSidePanelTab(tab);
      setSidePanelOpen((current) => !(current && activeSidePanelTab === tab));
    });
  };

  return (
    <main className="session-workspace">
      <SessionRuntimeEffects
        client={client}
        runIdParam={runIdParam}
        sessionId={sessionId}
        store={store}
      />

      <ArenaLayout
        layoutMode={arenaLayoutMode}
        timelineExpanded={timelineExpanded}
        sidePanelOpen={sidePanelOpen}
        controls={
          <SessionCommandDeck
            session={session}
            sessionId={sessionId}
            pluginDisplayName={sessionPlugin?.displayName}
            controlsExpanded={controlsExpanded}
            currentSceneSeq={currentSceneSeq}
            onToggleControls={() => {
              setControlsExpanded((current) => !current);
            }}
          >
            <SessionControls store={store} />
          </SessionCommandDeck>
        }
        stage={
          <SessionStage
            client={client}
            mediaResolver={mediaResolver}
            pluginDefinition={sessionPlugin}
            store={store}
          />
        }
        utilityRail={
          <SessionUtilityRail
            activeTab={activeSidePanelTab}
            sidePanelOpen={sidePanelOpen}
            onToggle={handleSidePanelToggle}
          />
        }
        timeline={
          compactLiveWorkspace ? null : (
            <SessionTimelineDrawer
              store={store}
              expanded={timelineExpanded}
              eventCounts={eventCounts}
              onToggle={() => {
                setTimelineExpanded((current) => !current);
              }}
            />
          )
        }
        sidePanel={
          <SessionSidePanelRegion
            activeTab={activeSidePanelTab}
            store={store}
            onActiveTabChange={setActiveSidePanelTab}
            onClose={() => {
              setSidePanelOpen(false);
            }}
          />
        }
      />
    </main>
  );
}

function formatPlaybackModeLabel(playbackMode: PlaybackMode | undefined): string {
  switch (playbackMode) {
    case "paused":
      return "Paused";
    case "replay_playing":
      return "Replay";
    case "live_tail":
    default:
      return "Live tail";
  }
}

function formatLifecycleLabel(lifecycle: string | undefined): string {
  if (!lifecycle) {
    return "warming";
  }
  return lifecycle.replaceAll("_", " ");
}

function formatStageInputMode(session: ReturnType<ArenaSessionStore["getSnapshot"]>["session"]): string {
  if (session?.capabilities.supportsRealtimeInputWebSocket === true) {
    return "Realtime socket";
  }
  if (session?.capabilities.supportsLowLatencyRealtimeInput === true) {
    return "Low latency input";
  }
  return "Host bridge";
}

function readStageReadinessSignals({
  session,
  playbackMode,
  isFullscreen,
}: {
  session?: ReturnType<ArenaSessionStore["getSnapshot"]>["session"];
  playbackMode: PlaybackMode;
  isFullscreen: boolean;
}): string[] {
  const signals: string[] = [];

  signals.push(playbackMode === "live_tail" ? "Tail locked" : formatPlaybackModeLabel(playbackMode));

  if (session?.scheduling.acceptsHumanIntent === true) {
    signals.push("Human input enabled");
  }

  if (isFullscreen) {
    signals.push("Esc exits fullscreen");
  }

  return signals;
}

function buildStageControlPanel({
  session,
  scene,
  playbackMode,
  currentSceneSeq,
  isFullscreen,
  operatorHint,
}: {
  session?: ReturnType<ArenaSessionStore["getSnapshot"]>["session"];
  scene?: ReturnType<ArenaSessionStore["getSnapshot"]>["scene"];
  playbackMode: PlaybackMode;
  currentSceneSeq?: number;
  isFullscreen: boolean;
  operatorHint?: string;
}): SharedSidePanelControlPanel {
  return {
    title: formatStageInputMode(session),
    meta: [
      {
        label: "Mode",
        value: formatPlaybackModeLabel(playbackMode),
      },
      {
        label: "Seq",
        value: `${currentSceneSeq ?? scene?.seq ?? "--"}`,
      },
      {
        label: "Actor",
        value:
          session?.scheduling.activeActorId ??
          scene?.activePlayerId ??
          session?.observer.observerId ??
          "standby",
      },
      {
        label: "Observer",
        value:
          session?.observer.observerKind === "player" && session.observer.observerId !== ""
            ? session.observer.observerId
            : session?.observer.observerKind ?? "spectator",
      },
    ],
    signals: readStageReadinessSignals({
      session,
      playbackMode,
      isFullscreen,
    }),
    operatorHint,
  };
}

function SessionCommandDeck({
  session,
  sessionId,
  pluginDisplayName,
  controlsExpanded,
  currentSceneSeq,
  children,
  onToggleControls,
}: {
  session?: ReturnType<ArenaSessionStore["getSnapshot"]>["session"];
  sessionId?: string;
  pluginDisplayName?: string;
  controlsExpanded: boolean;
  currentSceneSeq?: number;
  children: ReactNode;
  onToggleControls: () => void;
}) {
  return (
    <section
      className={[
        "session-command-deck",
        controlsExpanded ? "is-expanded" : "is-collapsed",
      ]
        .filter((value) => value !== "")
        .join(" ")}
      aria-label="Session command deck"
    >
      <div className="session-command-deck__overview">
        <div className="session-command-deck__identity">
          <div
            className="session-command-deck__brand"
            aria-label="GAGE GAME ARENA"
            data-testid="session-brand"
          >
            <span className="session-command-deck__brand-mark">GAGE</span>
            <span className="session-command-deck__brand-wordmark">GAME ARENA</span>
          </div>
          <p className="eyebrow">Live Session Workspace</p>
          <div className="session-command-deck__headline">
            <div>
              <h1>{sessionId ?? "Unknown session"}</h1>
              {controlsExpanded ? (
                <p className="session-command-deck__subtitle">
                  Stage-first workspace for live operation, human intervention, and immediate fullscreen
                  takeover.
                </p>
              ) : null}
            </div>
            <div className="session-command-deck__actions">
              <button
                type="button"
                className="session-command-deck__drawer-button session-command-deck__drawer-button--accent"
                aria-expanded={controlsExpanded}
                aria-controls="session-command-deck-controls"
                onClick={onToggleControls}
              >
                {controlsExpanded ? "Collapse session controls" : "Expand session controls"}
              </button>
              <Link className="session-command-deck__back-link" to="/">
                Back to host
              </Link>
            </div>
          </div>
        </div>

        <div className="session-command-deck__status-strip">
          <span className="session-command-deck__status-pill session-command-deck__status-pill--mode">
            {formatPlaybackModeLabel(session?.playback.mode)}
          </span>
          <span className="session-command-deck__status-pill">
            {pluginDisplayName ?? session?.gameId ?? "game loading"}
          </span>
          <span className="session-command-deck__status-pill">
            {formatLifecycleLabel(session?.lifecycle)}
          </span>
          <span className="session-command-deck__status-pill">
            seq {currentSceneSeq ?? "--"}
          </span>
          <span className="session-command-deck__status-pill">
            observer {session?.observer.observerKind ?? "spectator"}
          </span>
          <span className="session-command-deck__status-pill">
            {session?.scheduling.family ?? "idle"} / {session?.scheduling.phase ?? "idle"}
          </span>
        </div>
      </div>

      {controlsExpanded ? (
        <div className="session-command-deck__control-strip" id="session-command-deck-controls">
          {children}
        </div>
      ) : null}
    </section>
  );
}

function SessionUtilityRail({
  activeTab,
  sidePanelOpen,
  onToggle,
}: {
  activeTab: SharedSidePanelTab;
  sidePanelOpen: boolean;
  onToggle: (tab: SharedSidePanelTab) => void;
}) {
  return (
    <div className="session-utility-rail" aria-label="Session utility rail">
      {SIDE_PANEL_TABS.map(({ tab, label, cue }) => {
        const isActive = sidePanelOpen && activeTab === tab;
        return (
          <button
            key={tab}
            type="button"
            className={isActive ? "session-utility-rail__button is-active" : "session-utility-rail__button"}
            aria-pressed={isActive}
            data-panel-tab={tab.toLowerCase()}
            onClick={() => {
              onToggle(tab);
            }}
          >
            <span className="session-utility-rail__button-label">{label}</span>
            <span aria-hidden="true" className="session-utility-rail__button-cue">
              {cue}
            </span>
          </button>
        );
      })}
    </div>
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
  const runtimeState = useArenaStoreSelector(
    store,
    (snapshot) => ({
      status: snapshot.status,
      sceneStatus: snapshot.sceneStatus,
      currentSceneSeq: snapshot.currentSceneSeq,
      sceneSeq: snapshot.scene?.seq,
      session: snapshot.session,
      scene: snapshot.scene,
      playbackMode: snapshot.session?.playback.mode ?? "live_tail",
      observerId:
        snapshot.sessionRequest?.observer?.observerId ?? snapshot.session?.observer.observerId ?? "",
      observerKind:
        snapshot.sessionRequest?.observer?.observerKind ??
        snapshot.session?.observer.observerKind ??
        "spectator",
      supportsLiveUpdateStream: snapshot.session?.capabilities.supportsLiveUpdateStream === true,
      playbackSpeed: snapshot.session?.playback.speed ?? 1,
    }),
    shallowEqualRecord,
  );
  const {
    status,
    sceneStatus,
    currentSceneSeq,
    sceneSeq,
    session,
    scene,
    playbackMode,
    observerId,
    observerKind,
    supportsLiveUpdateStream,
    playbackSpeed,
  } = runtimeState;

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
      sceneSeq === currentSceneSeq
    ) {
      return;
    }

    // Low-latency frame transport can keep the same media ref alive, but the
    // scene metadata still has to advance so tick/lastMove overlays stay fresh.
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
  const controlState = useArenaStoreSelector(
    store,
    (snapshot) => ({
      status: snapshot.status,
      session: snapshot.session,
      currentSceneSeq: snapshot.currentSceneSeq,
      timelineEvents: snapshot.timeline.events,
    }),
    shallowEqualRecord,
  );
  const { status, session, currentSceneSeq, timelineEvents } = controlState;
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
  pluginDefinition,
}: {
  client: ReturnType<typeof createArenaGatewayClient>;
  store: ArenaSessionStore;
  mediaResolver: ArenaMediaResolver;
  pluginDefinition?: ReturnType<typeof resolveArenaPlugin>;
}) {
  const stageRef = useRef<HTMLElement | null>(null);
  const stageState = useArenaStoreSelector(
    store,
    (snapshot) => ({
      status: snapshot.status,
      error: snapshot.error,
      sessionRequest: snapshot.sessionRequest,
      session: snapshot.session,
      scene: snapshot.scene,
      sceneStatus: snapshot.sceneStatus,
      currentSceneSeq: snapshot.currentSceneSeq,
      latestActionReceipt: snapshot.latestActionReceipt,
    }),
    shallowEqualRecord,
  );
  const {
    status,
    error,
    sessionRequest,
    session,
    scene,
    sceneStatus,
    currentSceneSeq,
    latestActionReceipt,
  } = stageState;
  const plugin =
    pluginDefinition ?? (session !== undefined ? resolveArenaPlugin(session.pluginId) : undefined);
  const inputBridge = useInputBridge({
    latestReceipt: latestActionReceipt,
    submitAction: store.submitAction,
    interpreter: plugin?.inputInterpreter,
  });
  const mediaSubscribe = useCallback(
    (
      request: Parameters<ArenaMediaResolver["subscribe"]>[0],
      listener: Parameters<ArenaMediaResolver["subscribe"]>[1],
    ) =>
      mediaResolver.subscribe(
        {
          ...request,
          runId: request.runId ?? sessionRequest?.runId,
        },
        listener,
      ),
    [mediaResolver, sessionRequest?.runId],
  );
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
  const realtimeInputSessionId = sessionRequest?.sessionId;
  const realtimeInputRunId = sessionRequest?.runId;

  useEffect(() => {
    realtimeInputSocketRef.current?.close();
    realtimeInputSocketRef.current = null;
    if (!useRealtimeInputSocketPath || !realtimeInputSessionId) {
      return;
    }
    const socket = createRealtimeInputSocket({
      url: client.buildRealtimeActionSocketUrl({
        sessionId: realtimeInputSessionId,
        runId: realtimeInputRunId,
      }),
    });
    realtimeInputSocketRef.current = socket;
    return () => {
      socket.close();
      if (realtimeInputSocketRef.current === socket) {
        realtimeInputSocketRef.current = null;
      }
    };
  }, [client, realtimeInputRunId, realtimeInputSessionId, useRealtimeInputSocketPath]);

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
  const resultBanner = readStageResultBanner(session);
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
        }
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
        data-testid="session-stage"
        onDoubleClick={() => {
          void handleToggleFullscreen();
        }}
      >
        {showStageChrome ? (
          <>
            <div className="session-stage__hud session-stage__hud--top-left">
              <span className="session-stage__hud-pill session-stage__hud-pill--accent">
                {plugin?.displayName ?? session.gameId}
              </span>
              <span className="session-stage__hud-pill">{scene?.kind ?? "scene"}</span>
            </div>
          </>
        ) : null}
        {resultBanner ? (
          <div className="session-stage__hud session-stage__hud--bottom-left" role="status" aria-live="polite">
            <div
              className="session-stage__telemetry-card session-stage__result-banner"
              data-testid="session-stage-result-banner"
            >
              <span>{resultBanner.eyebrow}</span>
              <strong>{resultBanner.title}</strong>
              {resultBanner.details.length > 0 ? (
                <div className="session-stage__result-details">
                  {resultBanner.details.map((detail) => (
                    <span key={detail} className="session-stage__cue-pill">
                      {detail}
                    </span>
                  ))}
                </div>
              ) : null}
            </div>
          </div>
        ) : null}
        <div className="session-stage__surface">
          <button
            className="session-stage__fullscreen-button"
            data-testid="session-stage-fullscreen-button"
            onClick={() => {
              void handleToggleFullscreen();
            }}
            type="button"
            aria-label={isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
            title={isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
            aria-pressed={isFullscreen}
          >
            <span
              className="session-stage__fullscreen-button-icon"
              aria-hidden="true"
            />
          </button>
          <PluginView
            session={session}
            scene={scene}
            latestActionReceipt={inputBridge.latestReceipt}
            submitAction={inputBridge.submitAction}
            submitInput={submitPluginInput}
            mediaSubscribe={mediaSubscribe}
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
  const timelineState = useArenaStoreSelector(
    store,
    (snapshot) => ({
      timeline: snapshot.timeline,
      currentSceneSeq: snapshot.currentSceneSeq,
    }),
    shallowEqualRecord,
  );
  const { timeline, currentSceneSeq } = timelineState;
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

function SessionTimelineDrawer({
  store,
  expanded,
  eventCounts,
  onToggle,
}: {
  store: ArenaSessionStore;
  expanded: boolean;
  eventCounts: {
    total: number;
    warn: number;
    critical: number;
  };
  onToggle: () => void;
}) {
  const timelineDrawerState = useArenaStoreSelector(
    store,
    (snapshot) => ({
      timeline: snapshot.timeline,
      currentSceneSeq: snapshot.currentSceneSeq,
      timelineStatus: snapshot.timeline.status,
    }),
    shallowEqualRecord,
  );
  const { timeline, currentSceneSeq, timelineStatus } = timelineDrawerState;
  const playbackControls = usePlaybackControls(store);
  const previewEvents = timeline.events.slice(-3).reverse();

  return (
    <section
      className={expanded ? "session-timeline-drawer is-expanded" : "session-timeline-drawer"}
      aria-label="Session timeline drawer"
    >
      <div className="session-timeline-drawer__topline">
        <div>
          <p className="eyebrow">Timeline Rail</p>
          <h2>Event flow</h2>
        </div>
        <button
          type="button"
          className="session-command-deck__drawer-button"
          aria-expanded={expanded}
          onClick={onToggle}
        >
          {expanded ? "Hide timeline drawer" : "Show timeline drawer"}
        </button>
      </div>
      <div className="session-timeline-drawer__rail">
        <span>seq {currentSceneSeq ?? "--"}</span>
        <span>{timelineStatus}</span>
        <span>{eventCounts.total} events</span>
        <span>{eventCounts.warn} warn</span>
        <span>{eventCounts.critical} critical</span>
      </div>
      {previewEvents.length > 0 ? (
        <div className="session-timeline-drawer__preview-list" aria-label="Recent timeline events">
          {previewEvents.map((event) => (
            <button
              key={event.seq}
              type="button"
              className={[
                "session-timeline-drawer__preview-event",
                event.severity ? `is-${event.severity}` : "",
              ]
                .filter((value) => value !== "")
                .join(" ")}
              onClick={() => {
                if (!expanded) {
                  onToggle();
                }
                void playbackControls.selectEvent(event.seq);
              }}
            >
              <span className="session-timeline-drawer__preview-seq">#{event.seq}</span>
              <span className="session-timeline-drawer__preview-copy">
                <strong>{event.label}</strong>
                <small>{event.type}</small>
              </span>
            </button>
          ))}
        </div>
      ) : null}
      {expanded ? <SessionTimeline store={store} /> : null}
    </section>
  );
}

function SessionSidePanelRegion({
  store,
  activeTab,
  onActiveTabChange,
  onClose,
}: {
  store: ArenaSessionStore;
  activeTab: SharedSidePanelTab;
  onActiveTabChange: (tab: SharedSidePanelTab) => void;
  onClose: () => void;
}) {
  const sidePanelState = useArenaStoreSelector(
    store,
    (snapshot) => ({
      session: snapshot.session,
      scene: snapshot.scene,
      currentSceneSeq: snapshot.currentSceneSeq,
      latestActionReceipt: snapshot.latestActionReceipt,
      error: snapshot.error,
    }),
    shallowEqualRecord,
  );
  const { session, scene, currentSceneSeq, latestActionReceipt, error } = sidePanelState;
  const pluginDefinition =
    session !== undefined ? resolveArenaPlugin(session.pluginId) : undefined;
  const playbackMode = session?.playback.mode ?? "live_tail";

  return (
    <div className="session-side-drawer">
      <div className="session-side-drawer__header">
        <div>
          <p className="eyebrow">Utility Drawer</p>
          <h2>{activeTab}</h2>
        </div>
        <button
          type="button"
          className="session-command-deck__drawer-button"
          onClick={onClose}
        >
          Close utility drawer
        </button>
      </div>
      <SharedSidePanel
        session={session}
        scene={scene}
        latestActionReceipt={latestActionReceipt}
        error={error}
        activeTab={activeTab}
        controlPanel={buildStageControlPanel({
          session,
          scene,
          playbackMode,
          currentSceneSeq,
          isFullscreen: false,
          operatorHint: pluginDefinition?.operatorHint,
        })}
        onActiveTabChange={onActiveTabChange}
        onObserverChange={(observer) => {
          void store.setObserver(observer).catch(() => {});
        }}
        onChatSubmit={(payload) => store.submitChat(payload)}
      />
    </div>
  );
}

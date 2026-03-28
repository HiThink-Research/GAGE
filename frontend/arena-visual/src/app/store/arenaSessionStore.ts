import type { ArenaGatewayClient } from "../../gateway/client";
import type {
  ActionIntentReceipt,
  ObserverRef,
  PlaybackMode,
  TimelineEvent,
  TimelinePage,
  VisualScene,
  VisualSession,
} from "../../gateway/types";

type StoreListener = () => void;
type StoreStatus = "idle" | "loading" | "ready" | "error";

interface SessionRequest {
  sessionId: string;
  runId?: string;
  observer?: ObserverRef;
}

export interface ArenaTimelineState {
  status: StoreStatus;
  events: TimelineEvent[];
  nextAfterSeq?: number | null;
  hasMore: boolean;
  limit: number;
  error?: string;
}

export interface ArenaSessionStoreSnapshot {
  status: StoreStatus;
  sessionRequest?: SessionRequest;
  session?: VisualSession;
  sceneStatus: StoreStatus;
  scene?: VisualScene;
  currentSceneSeq?: number;
  timeline: ArenaTimelineState;
  latestActionReceipt?: ActionIntentReceipt;
  error?: string;
}

interface LoadSessionInput extends SessionRequest {}

interface LoadMoreTimelineInput {
  limit?: number;
}

interface LoadSceneInput {
  seq: number;
}

export interface ArenaSessionStore {
  getSnapshot: () => ArenaSessionStoreSnapshot;
  subscribe: (listener: StoreListener) => () => void;
  loadSession: (input: LoadSessionInput) => Promise<void>;
  loadMoreTimeline: (input?: LoadMoreTimelineInput) => Promise<void>;
  loadScene: (input: LoadSceneInput) => Promise<void>;
  setCurrentSceneSeq: (seq: number) => void;
  setPlaybackMode: (mode: PlaybackMode) => void;
  setObserver: (observer: ObserverRef) => Promise<void>;
  submitAction: (payload: Record<string, unknown>) => Promise<ActionIntentReceipt>;
  clearLatestActionReceipt: () => void;
}

const DEFAULT_TIMELINE_LIMIT = 50;

export function createArenaSessionStore(
  client: ArenaGatewayClient,
): ArenaSessionStore {
  const listeners = new Set<StoreListener>();
  let sessionRequestVersion = 0;
  let sceneRequestVersion = 0;
  let state: ArenaSessionStoreSnapshot = {
    status: "idle",
    sceneStatus: "idle",
    timeline: {
      status: "idle",
      events: [],
      nextAfterSeq: undefined,
      hasMore: false,
      limit: DEFAULT_TIMELINE_LIMIT,
    },
  };

  function notify(): void {
    for (const listener of listeners) {
      listener();
    }
  }

  function setState(
    updater: ArenaSessionStoreSnapshot | ((previous: ArenaSessionStoreSnapshot) => ArenaSessionStoreSnapshot),
  ): void {
    state = typeof updater === "function" ? updater(state) : updater;
    notify();
  }

  function mergeTimelineEvents(events: TimelineEvent[]): TimelineEvent[] {
    const merged = new Map<number, TimelineEvent>();
    for (const event of state.timeline.events) {
      merged.set(event.seq, event);
    }
    for (const event of events) {
      merged.set(event.seq, event);
    }
    return Array.from(merged.values()).sort((left, right) => left.seq - right.seq);
  }

  function updateSessionPlayback(mode: PlaybackMode, cursorEventSeq?: number): void {
    if (!state.session) {
      return;
    }

    state = {
      ...state,
      session: {
        ...state.session,
        playback: {
          ...state.session.playback,
          mode,
          cursorEventSeq: cursorEventSeq ?? state.session.playback.cursorEventSeq,
        },
      },
    };
    notify();
  }

  async function loadSession({ sessionId, runId, observer }: LoadSessionInput): Promise<void> {
    sessionRequestVersion += 1;
    sceneRequestVersion += 1;
    const requestVersion = sessionRequestVersion;
    setState((previous) => ({
      ...previous,
      status: "loading",
      sessionRequest: { sessionId, runId, observer },
      session: undefined,
      error: undefined,
      latestActionReceipt: undefined,
      scene: undefined,
      sceneStatus: "idle",
      currentSceneSeq: undefined,
      timeline: {
        ...previous.timeline,
        status: "loading",
        events: [],
        nextAfterSeq: undefined,
        hasMore: false,
        error: undefined,
      },
    }));

    try {
      const [session, timelinePage] = await Promise.all([
        client.loadSession({ sessionId, runId, observer }),
        client.loadTimeline({ sessionId, runId }),
      ]);
      if (requestVersion !== sessionRequestVersion) {
        return;
      }
      const lastEvent = timelinePage.events.at(-1);
      setState((previous) => ({
        ...previous,
        status: "ready",
        session,
        sceneStatus: "idle",
        currentSceneSeq:
          lastEvent?.seq ?? (session.playback.cursorEventSeq > 0 ? session.playback.cursorEventSeq : undefined),
        timeline: timelineFromPage(timelinePage),
      }));
    } catch (caughtError) {
      if (requestVersion !== sessionRequestVersion) {
        return;
      }
      setState((previous) => ({
        ...previous,
        status: "error",
        timeline: {
          ...previous.timeline,
          status: "error",
          error: toErrorMessage(caughtError),
        },
        error: toErrorMessage(caughtError),
      }));
      throw caughtError;
    }
  }

  async function loadMoreTimeline({
    limit = DEFAULT_TIMELINE_LIMIT,
  }: LoadMoreTimelineInput = {}): Promise<void> {
    if (!state.sessionRequest) {
      return;
    }
    const requestContext = state.sessionRequest;

    setState((previous) => ({
      ...previous,
      timeline: {
        ...previous.timeline,
        status: "loading",
        error: undefined,
      },
    }));

    try {
      const page = await client.loadTimeline({
        sessionId: requestContext.sessionId,
        runId: requestContext.runId,
        afterSeq: state.timeline.nextAfterSeq ?? undefined,
        limit,
      });
      if (
        state.sessionRequest?.sessionId !== requestContext.sessionId ||
        state.sessionRequest?.runId !== requestContext.runId
      ) {
        return;
      }
      const mergedEvents = mergeTimelineEvents(page.events);
      const lastEvent = mergedEvents.at(-1);
      setState((previous) => ({
        ...previous,
        currentSceneSeq:
          previous.session?.playback.mode === "live_tail" && lastEvent
            ? lastEvent.seq
            : previous.currentSceneSeq,
        timeline: {
          ...timelineFromPage(page),
          events: mergedEvents,
        },
      }));
    } catch (caughtError) {
      if (
        state.sessionRequest?.sessionId !== requestContext.sessionId ||
        state.sessionRequest?.runId !== requestContext.runId
      ) {
        return;
      }
      setState((previous) => ({
        ...previous,
        timeline: {
          ...previous.timeline,
          status: "error",
          error: toErrorMessage(caughtError),
        },
      }));
      throw caughtError;
    }
  }

  async function loadScene({ seq }: LoadSceneInput): Promise<void> {
    if (!state.sessionRequest) {
      return;
    }
    sceneRequestVersion += 1;
    const requestVersion = sceneRequestVersion;
    const requestContext = state.sessionRequest;

    setState((previous) => ({
      ...previous,
      currentSceneSeq: seq,
      scene: undefined,
      sceneStatus: "loading",
      error: undefined,
    }));

    try {
      const scene = await client.loadScene({
        sessionId: requestContext.sessionId,
        runId: requestContext.runId,
        seq,
        observer: requestContext.observer,
      });
      if (
        requestVersion !== sceneRequestVersion ||
        state.sessionRequest?.sessionId !== requestContext.sessionId ||
        state.sessionRequest?.runId !== requestContext.runId
      ) {
        return;
      }
      setState((previous) => ({
        ...previous,
        scene,
        sceneStatus: "ready",
      }));
    } catch (caughtError) {
      if (
        requestVersion !== sceneRequestVersion ||
        state.sessionRequest?.sessionId !== requestContext.sessionId ||
        state.sessionRequest?.runId !== requestContext.runId
      ) {
        return;
      }
      setState((previous) => ({
        ...previous,
        scene: undefined,
        sceneStatus: "error",
        error: toErrorMessage(caughtError),
      }));
      throw caughtError;
    }
  }

  function setCurrentSceneSeq(seq: number): void {
    setState((previous) => ({
      ...previous,
      currentSceneSeq: seq,
    }));
    updateSessionPlayback("paused", seq);
  }

  function setPlaybackMode(mode: PlaybackMode): void {
    const nextCursorSeq =
      mode === "live_tail" ? state.timeline.events.at(-1)?.seq ?? state.currentSceneSeq : state.currentSceneSeq;
    setState((previous) => ({
      ...previous,
      currentSceneSeq: mode === "live_tail" ? nextCursorSeq : previous.currentSceneSeq,
    }));
    updateSessionPlayback(mode, nextCursorSeq);
  }

  async function setObserver(observer: ObserverRef): Promise<void> {
    if (!state.sessionRequest) {
      return;
    }

    sessionRequestVersion += 1;
    sceneRequestVersion = sessionRequestVersion;
    const requestVersion = sessionRequestVersion;
    const requestContext: SessionRequest = {
      ...state.sessionRequest,
      observer,
    };
    const sceneSeq = state.currentSceneSeq;

    setState((previous) => ({
      ...previous,
      status: "loading",
      sessionRequest: requestContext,
      session:
        previous.session === undefined
          ? previous.session
          : {
              ...previous.session,
              observer,
            },
      sceneStatus: sceneSeq === undefined ? previous.sceneStatus : "loading",
      error: undefined,
    }));

    try {
      const [session, scene] = await Promise.all([
        client.loadSession({
          sessionId: requestContext.sessionId,
          runId: requestContext.runId,
          observer,
        }),
        sceneSeq === undefined
          ? Promise.resolve(undefined)
          : client.loadScene({
              sessionId: requestContext.sessionId,
              runId: requestContext.runId,
              seq: sceneSeq,
              observer,
            }),
      ]);
      if (
        requestVersion !== sessionRequestVersion ||
        requestVersion !== sceneRequestVersion ||
        state.sessionRequest?.sessionId !== requestContext.sessionId ||
        state.sessionRequest?.runId !== requestContext.runId
      ) {
        return;
      }
      setState((previous) => ({
        ...previous,
        status: "ready",
        session,
        scene: scene ?? previous.scene,
        sceneStatus: sceneSeq === undefined ? previous.sceneStatus : "ready",
      }));
    } catch (caughtError) {
      if (
        requestVersion !== sessionRequestVersion ||
        requestVersion !== sceneRequestVersion ||
        state.sessionRequest?.sessionId !== requestContext.sessionId ||
        state.sessionRequest?.runId !== requestContext.runId
      ) {
        return;
      }
      setState((previous) => ({
        ...previous,
        status: "error",
        sceneStatus: sceneSeq === undefined ? previous.sceneStatus : "error",
        error: toErrorMessage(caughtError),
      }));
      throw caughtError;
    }
  }

  async function submitAction(
    payload: Record<string, unknown>,
  ): Promise<ActionIntentReceipt> {
    if (!state.sessionRequest) {
      throw new Error("No session is loaded.");
    }

    setState((previous) => ({
      ...previous,
      latestActionReceipt: {
        intentId: "pending-intent",
        state: "pending",
      },
    }));

    const receipt = await client.submitAction({
      sessionId: state.sessionRequest.sessionId,
      runId: state.sessionRequest.runId,
      payload,
    });
    setState((previous) => ({
      ...previous,
      latestActionReceipt: receipt,
    }));
    return receipt;
  }

  async function submitActionWithFailureHandling(
    payload: Record<string, unknown>,
  ): Promise<ActionIntentReceipt> {
    try {
      return await submitAction(payload);
    } catch (caughtError) {
      setState((previous) => ({
        ...previous,
        latestActionReceipt: {
          intentId: "rejected-intent",
          state: "rejected",
          reason: toErrorMessage(caughtError),
        },
      }));
      throw caughtError;
    }
  }

  return {
    getSnapshot: () => state,
    subscribe(listener) {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
    loadSession,
    loadMoreTimeline,
    loadScene,
    setCurrentSceneSeq,
    setPlaybackMode,
    setObserver,
    submitAction: submitActionWithFailureHandling,
    clearLatestActionReceipt() {
      setState((previous) => ({
        ...previous,
        latestActionReceipt: undefined,
      }));
    },
  };
}

function timelineFromPage(page: TimelinePage): ArenaTimelineState {
  return {
    status: "ready",
    events: [...page.events],
    nextAfterSeq: page.nextAfterSeq,
    hasMore: page.hasMore,
    limit: page.limit,
  };
}

function toErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "Unexpected error.";
}

import type { ArenaGatewayClient } from "../../gateway/client";
import type {
  ActionIntentReceipt,
  ObserverRef,
  PlaybackMode,
  TimelineEvent,
  TimelinePage,
  TimelineEventType,
  TimelineSeverity,
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
  filters: TimelineFilterState;
  error?: string;
}

export type TimelineFilterSeverity = TimelineSeverity | "all";

export interface TimelineFilterState {
  eventTypes: TimelineEventType[];
  severity: TimelineFilterSeverity;
  humanIntentOnly: boolean;
}

export interface ArenaControlPayload {
  commandType:
    | "follow_tail"
    | "pause"
    | "replay"
    | "seek_seq"
    | "seek_end"
    | "step"
    | "set_speed"
    | "back_to_tail"
    | "finish";
  targetSeq?: number;
  stepDelta?: -1 | 1;
  speed?: number;
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
  advanceReplayPlayback?: () => void;
  setCurrentSceneSeq: (seq: number) => void;
  setPlaybackMode: (mode: PlaybackMode) => void;
  setTimelineFilters: (filters: TimelineFilterState) => void;
  setObserver: (observer: ObserverRef) => Promise<void>;
  submitChat: (payload: Record<string, unknown>) => Promise<ActionIntentReceipt>;
  submitControl: (payload: ArenaControlPayload) => Promise<ActionIntentReceipt>;
  submitAction: (payload: Record<string, unknown>) => Promise<ActionIntentReceipt>;
  clearLatestActionReceipt: () => void;
}

const DEFAULT_TIMELINE_LIMIT = 50;
const DEFAULT_TIMELINE_FILTERS: TimelineFilterState = {
  eventTypes: [],
  severity: "all",
  humanIntentOnly: false,
};
const LOCAL_PLAYBACK_FALLBACK_REASON = "local_playback_fallback";
const LOCAL_PLAYBACK_UNAVAILABLE_REASONS = new Set([
  "action_queue_not_available",
  "control_queue_not_available",
  "control_submitter_not_configured",
  "sample_route_not_found",
]);

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
      filters: DEFAULT_TIMELINE_FILTERS,
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

  function setTimelineFilters(filters: TimelineFilterState): void {
    setState((previous) => ({
      ...previous,
      timeline: {
        ...previous.timeline,
        filters,
      },
    }));
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
        filters: { ...DEFAULT_TIMELINE_FILTERS },
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
          session.playback.mode === "live_tail"
            ? lastEvent?.seq ??
              (session.playback.cursorEventSeq > 0 ? session.playback.cursorEventSeq : undefined)
            : session.playback.cursorEventSeq > 0
              ? session.playback.cursorEventSeq
              : lastEvent?.seq,
        timeline: timelineFromPage(timelinePage, previous.timeline.filters),
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
          ...timelineFromPage(page, previous.timeline.filters),
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

  function advanceReplayPlayback(): void {
    setState((previous) => {
      if (!previous.session || previous.session.playback.mode !== "replay_playing") {
        return previous;
      }

      const currentSeq = previous.currentSceneSeq ?? previous.session.playback.cursorEventSeq;
      const nextSeq = resolveStepTargetSeq(previous.timeline.events, currentSeq, 1);
      if (nextSeq === undefined || nextSeq === currentSeq) {
        return {
          ...previous,
          session: {
            ...previous.session,
            playback: {
              ...previous.session.playback,
              mode: "paused",
            },
          },
        };
      }

      return {
        ...previous,
        currentSceneSeq: nextSeq,
        session: {
          ...previous.session,
          playback: {
            ...previous.session.playback,
            cursorEventSeq: nextSeq,
          },
        },
      };
    });
  }

  function applyControlReceipt(
    previous: ArenaSessionStoreSnapshot,
    payload: ArenaControlPayload,
    receipt: ActionIntentReceipt,
  ): ArenaSessionStoreSnapshot {
    if (!previous.session || !["accepted", "committed"].includes(receipt.state)) {
      return {
        ...previous,
        latestActionReceipt: receipt,
      };
    }

    const latestTimelineSeq =
      previous.timeline.events.at(-1)?.seq ??
      previous.currentSceneSeq ??
      previous.session.playback.cursorEventSeq;
    const stepTargetSeq =
      receipt.relatedEventSeq ?? resolveStepTargetSeq(previous.timeline.events, previous.currentSceneSeq, payload.stepDelta);

    let nextMode = previous.session.playback.mode;
    let nextCursorSeq = previous.currentSceneSeq ?? previous.session.playback.cursorEventSeq;
    let nextSpeed = previous.session.playback.speed;

    switch (payload.commandType) {
      case "pause":
        nextMode = "paused";
        nextCursorSeq = receipt.relatedEventSeq ?? nextCursorSeq;
        break;
      case "replay":
        nextMode = "replay_playing";
        nextCursorSeq = receipt.relatedEventSeq ?? previous.timeline.events[0]?.seq ?? nextCursorSeq;
        break;
      case "follow_tail":
      case "back_to_tail":
        nextMode = "live_tail";
        nextCursorSeq = receipt.relatedEventSeq ?? latestTimelineSeq;
        break;
      case "seek_seq":
        nextMode = "paused";
        nextCursorSeq = receipt.relatedEventSeq ?? payload.targetSeq ?? nextCursorSeq;
        break;
      case "seek_end":
        nextMode = "paused";
        nextCursorSeq = receipt.relatedEventSeq ?? latestTimelineSeq;
        break;
      case "step":
        nextMode = "paused";
        nextCursorSeq = stepTargetSeq ?? nextCursorSeq;
        break;
      case "set_speed":
        nextSpeed = payload.speed ?? nextSpeed;
        break;
    }

    return {
      ...previous,
      currentSceneSeq: nextCursorSeq,
      latestActionReceipt: receipt,
      session: {
        ...previous.session,
        playback: {
          ...previous.session.playback,
          mode: nextMode,
          cursorEventSeq: nextCursorSeq,
          speed: nextSpeed,
        },
      },
    };
  }

  function applyLocalPlaybackFallback(
    previous: ArenaSessionStoreSnapshot,
    payload: ArenaControlPayload,
    receipt: ActionIntentReceipt,
  ): ArenaSessionStoreSnapshot {
    if (!previous.session) {
      return {
        ...previous,
        latestActionReceipt: receipt,
      };
    }

    const headSeq = previous.timeline.events[0]?.seq;
    const tailSeq =
      previous.timeline.events.at(-1)?.seq ??
      previous.currentSceneSeq ??
      previous.session.playback.cursorEventSeq;
    const currentSeq = previous.currentSceneSeq ?? previous.session.playback.cursorEventSeq;
    let nextMode = previous.session.playback.mode;
    let nextCursorSeq = currentSeq;
    let nextSpeed = previous.session.playback.speed;

    switch (payload.commandType) {
      case "pause":
        nextMode = "paused";
        break;
      case "replay":
        nextMode = "replay_playing";
        nextCursorSeq = headSeq ?? currentSeq;
        break;
      case "follow_tail":
      case "back_to_tail":
        nextMode = "live_tail";
        nextCursorSeq = tailSeq;
        break;
      case "seek_end":
        nextMode = "paused";
        nextCursorSeq = tailSeq;
        break;
      case "seek_seq":
        nextMode = "paused";
        nextCursorSeq = payload.targetSeq ?? currentSeq;
        break;
      case "step":
        nextMode = "paused";
        nextCursorSeq =
          resolveStepTargetSeq(previous.timeline.events, previous.currentSceneSeq, payload.stepDelta) ??
          currentSeq;
        break;
      case "set_speed":
        nextSpeed = payload.speed ?? nextSpeed;
        break;
    }

    const fallbackReceipt: ActionIntentReceipt = {
      intentId: receipt.intentId,
      state: "accepted",
      relatedEventSeq: nextCursorSeq,
      reason: LOCAL_PLAYBACK_FALLBACK_REASON,
    };

    return {
      ...previous,
      currentSceneSeq: nextCursorSeq,
      latestActionReceipt: fallbackReceipt,
      session: {
        ...previous.session,
        playback: {
          ...previous.session.playback,
          mode: nextMode,
          cursorEventSeq: nextCursorSeq,
          speed: nextSpeed,
        },
      },
    };
  }

  function shouldApplyLocalPlaybackFallback(
    payload: ArenaControlPayload,
    receipt: ActionIntentReceipt,
  ): boolean {
    if (receipt.state !== "rejected") {
      return false;
    }
    if (!LOCAL_PLAYBACK_UNAVAILABLE_REASONS.has(receipt.reason ?? "")) {
      return false;
    }
    if (payload.commandType === "finish") {
      return false;
    }
    return payload.commandType !== "set_speed" || payload.speed !== undefined;
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

  async function submitChat(
    payload: Record<string, unknown>,
  ): Promise<ActionIntentReceipt> {
    if (!state.sessionRequest) {
      throw new Error("No session is loaded.");
    }
    const requestVersion = sessionRequestVersion;
    const requestContext = state.sessionRequest;

    setState((previous) => ({
      ...previous,
      latestActionReceipt: {
        intentId: "pending-chat",
        state: "pending",
      },
    }));

    try {
      const receipt = await client.submitChat({
        sessionId: requestContext.sessionId,
        runId: requestContext.runId,
        payload,
      });
      if (
        requestVersion !== sessionRequestVersion ||
        state.sessionRequest?.sessionId !== requestContext.sessionId ||
        state.sessionRequest?.runId !== requestContext.runId
      ) {
        return receipt;
      }
      setState((previous) => ({
        ...previous,
        latestActionReceipt: receipt,
      }));
      return receipt;
    } catch (caughtError) {
      if (
        requestVersion !== sessionRequestVersion ||
        state.sessionRequest?.sessionId !== requestContext.sessionId ||
        state.sessionRequest?.runId !== requestContext.runId
      ) {
        throw caughtError;
      }
      setState((previous) => ({
        ...previous,
        latestActionReceipt: {
          intentId: "rejected-chat",
          state: "rejected",
          reason: toErrorMessage(caughtError),
        },
      }));
      throw caughtError;
    }
  }

  async function submitControl(payload: ArenaControlPayload): Promise<ActionIntentReceipt> {
    if (!state.sessionRequest) {
      throw new Error("No session is loaded.");
    }
    const requestVersion = sessionRequestVersion;
    const requestContext = state.sessionRequest;

    setState((previous) => ({
      ...previous,
      latestActionReceipt: {
        intentId: "pending-control",
        state: "pending",
      },
    }));

    try {
      const receipt = await client.submitControl({
        sessionId: requestContext.sessionId,
        runId: requestContext.runId,
        payload: payload as unknown as Record<string, unknown>,
      });
      if (
        requestVersion !== sessionRequestVersion ||
        state.sessionRequest?.sessionId !== requestContext.sessionId ||
        state.sessionRequest?.runId !== requestContext.runId
      ) {
        return receipt;
      }
      if (shouldApplyLocalPlaybackFallback(payload, receipt)) {
        const fallbackSnapshot = applyLocalPlaybackFallback(state, payload, receipt);
        setState(fallbackSnapshot);
        return fallbackSnapshot.latestActionReceipt ?? receipt;
      }
      setState((previous) => applyControlReceipt(previous, payload, receipt));
      return receipt;
    } catch (caughtError) {
      if (
        requestVersion !== sessionRequestVersion ||
        state.sessionRequest?.sessionId !== requestContext.sessionId ||
        state.sessionRequest?.runId !== requestContext.runId
      ) {
        throw caughtError;
      }
      setState((previous) => ({
        ...previous,
        latestActionReceipt: {
          intentId: "rejected-control",
          state: "rejected",
          reason: toErrorMessage(caughtError),
        },
      }));
      throw caughtError;
    }
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
    advanceReplayPlayback,
    setCurrentSceneSeq,
    setPlaybackMode,
    setTimelineFilters,
    setObserver,
    submitChat,
    submitControl,
    submitAction: submitActionWithFailureHandling,
    clearLatestActionReceipt() {
      setState((previous) => ({
        ...previous,
        latestActionReceipt: undefined,
      }));
    },
  };
}

function timelineFromPage(
  page: TimelinePage,
  filters: TimelineFilterState,
): ArenaTimelineState {
  return {
    status: "ready",
    events: [...page.events],
    nextAfterSeq: page.nextAfterSeq,
    hasMore: page.hasMore,
    limit: page.limit,
    filters: { ...filters },
  };
}

function resolveStepTargetSeq(
  events: TimelineEvent[],
  currentSceneSeq: number | undefined,
  stepDelta: number | undefined,
): number | undefined {
  if (currentSceneSeq === undefined || stepDelta === undefined) {
    return currentSceneSeq;
  }
  const checkpoints = resolvePlaybackCheckpointSeqs(events);
  if (checkpoints.length === 0) {
    return currentSceneSeq;
  }

  if (stepDelta > 0) {
    for (const seq of checkpoints) {
      if (seq > currentSceneSeq) {
        return seq;
      }
    }
    return checkpoints.at(-1) ?? currentSceneSeq;
  }

  if (stepDelta < 0) {
    for (let index = checkpoints.length - 1; index >= 0; index -= 1) {
      const seq = checkpoints[index];
      if (seq < currentSceneSeq) {
        return seq;
      }
    }
    return checkpoints[0] ?? currentSceneSeq;
  }

  return currentSceneSeq;
}

function resolvePlaybackCheckpointSeqs(events: TimelineEvent[]): number[] {
  const turnBasedCheckpoints = events
    .filter((event) => event.type === "decision_window_open" || event.type === "result")
    .map((event) => event.seq);
  if (turnBasedCheckpoints.length > 0) {
    return turnBasedCheckpoints;
  }

  const genericCheckpoints = events
    .filter(
      (event) =>
        event.type === "snapshot" ||
        event.type === "frame_ref" ||
        event.type === "result",
    )
    .map((event) => event.seq);
  if (genericCheckpoints.length > 0) {
    return genericCheckpoints;
  }

  return events.map((event) => event.seq);
}

function toErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "Unexpected error.";
}

export type VisualSessionLifecycle =
  | "initializing"
  | "live_running"
  | "live_ended"
  | "closed";

export type PlaybackMode = "live_tail" | "paused" | "replay_playing";

export type ObserverKind = "global" | "player" | "spectator" | "camera";

export type SchedulingFamily =
  | "turn"
  | "agent_cycle"
  | "record_cadence"
  | "real_time_tick";

export type SchedulingPhase =
  | "idle"
  | "waiting_for_intent"
  | "advancing"
  | "recording"
  | "completed";

export type TimelineEventType =
  | "action_intent"
  | "action_committed"
  | "decision_window_open"
  | "decision_window_close"
  | "snapshot"
  | "frame_ref"
  | "chat"
  | "system_marker"
  | "result";

export type TimelineSeverity = "info" | "warn" | "critical";

export type VisualSceneKind = "board" | "table" | "frame" | "rts";

export type VisualScenePhase = "live" | "replay";

export type MediaTransport =
  | "artifact_ref"
  | "http_pull"
  | "binary_stream"
  | "low_latency_channel";

export type ActionIntentReceiptState =
  | "pending"
  | "accepted"
  | "committed"
  | "rejected"
  | "expired";

export interface PlaybackState {
  mode: PlaybackMode;
  cursorTs: number;
  cursorEventSeq: number;
  speed: number;
  canSeek: boolean;
}

export interface ObserverRef {
  observerId: string;
  observerKind: ObserverKind;
}

export interface SchedulingState {
  family: SchedulingFamily;
  phase: SchedulingPhase;
  acceptsHumanIntent: boolean;
  activeActorId?: string;
  windowId?: string;
}

export interface TimelineEvent {
  seq: number;
  tsMs: number;
  type: TimelineEventType;
  label: string;
  actorId?: string;
  refSnapshotSeq?: number;
  detail?: string;
  severity?: TimelineSeverity;
  tags?: string[];
  payload?: unknown;
}

export interface TimelinePage {
  sessionId: string;
  afterSeq?: number | null;
  nextAfterSeq?: number | null;
  limit: number;
  hasMore: boolean;
  events: TimelineEvent[];
}

export interface MediaSourceRef {
  mediaId: string;
  transport: MediaTransport;
  mimeType?: string;
  url?: string;
  previewRef?: string;
}

export interface VisualSceneMedia {
  primary?: MediaSourceRef;
  auxiliary?: MediaSourceRef[];
}

export interface VisualScene {
  sceneId: string;
  gameId: string;
  pluginId: string;
  kind: VisualSceneKind;
  tsMs: number;
  seq: number;
  phase: VisualScenePhase;
  activePlayerId: string | null;
  legalActions: Array<Record<string, unknown>>;
  summary: Record<string, unknown>;
  body: unknown;
  media?: VisualSceneMedia;
  overlays?: Array<Record<string, unknown>>;
}

export interface VisualSession {
  sessionId: string;
  gameId: string;
  pluginId: string;
  lifecycle: VisualSessionLifecycle;
  playback: PlaybackState;
  observer: ObserverRef;
  scheduling: SchedulingState;
  capabilities: Record<string, unknown>;
  summary: Record<string, unknown>;
  timeline: Record<string, unknown>;
}

export interface ActionIntentReceipt {
  intentId: string;
  state: ActionIntentReceiptState;
  relatedEventSeq?: number;
  reason?: string;
}

export interface GamePluginManifest {
  sceneKinds: VisualSceneKind[];
  supportedObservers: ObserverKind[];
  acceptsHumanIntent: boolean;
  extensionPanels?: string[];
}

import type {
  ActionIntentReceipt,
  MediaSourceRef,
  ObserverRef,
  TimelinePage,
  VisualScene,
  VisualSession,
} from "./types";

export interface ArenaGatewayClientOptions {
  baseUrl: string;
  fetchFn?: typeof fetch;
}

interface RequestContext {
  sessionId: string;
  runId?: string;
}

interface ObserverReadContext extends RequestContext {
  observer?: ObserverRef;
}

interface TimelineRequest extends RequestContext {
  afterSeq?: number | null;
  limit?: number;
}

interface SceneRequest extends ObserverReadContext {
  seq: number;
}

interface MarkerRequest extends RequestContext {
  marker: string;
}

interface MediaRequest extends RequestContext {
  mediaId: string;
}

interface SubmitActionRequest extends RequestContext {
  payload: Record<string, unknown>;
}

export interface ArenaGatewayClient {
  loadSession(input: ObserverReadContext): Promise<VisualSession>;
  loadTimeline(input: TimelineRequest): Promise<TimelinePage>;
  loadScene(input: SceneRequest): Promise<VisualScene>;
  loadMarkers(input: MarkerRequest): Promise<{ sessionId: string; marker: string; seqs: number[] }>;
  loadMedia(input: MediaRequest): Promise<MediaSourceRef>;
  submitAction(input: SubmitActionRequest): Promise<ActionIntentReceipt>;
  buildMediaUrl(input: MediaRequest): string;
}

interface ArenaGatewayErrorInit {
  message: string;
  status?: number;
  code?: string;
  cause?: unknown;
}

export class ArenaGatewayError extends Error {
  readonly status?: number;
  readonly code?: string;

  constructor({ message, status, code, cause }: ArenaGatewayErrorInit) {
    super(message, { cause });
    this.name = "ArenaGatewayError";
    this.status = status;
    this.code = code;
  }
}

export function createArenaGatewayClient({
  baseUrl,
  fetchFn = fetch,
}: ArenaGatewayClientOptions): ArenaGatewayClient {
  const normalizedBaseUrl = baseUrl.replace(/\/+$/, "");

  const requestJson = async <T>(
    path: string,
    init: RequestInit & { runId?: string } = {},
  ): Promise<T> => {
    const { runId, ...requestInit } = init;
    const url = new URL(`${normalizedBaseUrl}${path}`);
    if (runId) {
      url.searchParams.set("run_id", runId);
    }

    let response: Response;
    try {
      response = await fetchFn(url.toString(), requestInit);
    } catch (error) {
      throw new ArenaGatewayError({
        message: "Network request failed",
        code: "network_error",
        cause: error,
      });
    }

    const payload = await parseResponseBody(response);
    if (!response.ok) {
      throw toGatewayError(response.status, payload);
    }
    return payload as T;
  };

  return {
    loadSession: ({ sessionId, runId, observer }) =>
      requestJson<VisualSession>(buildSessionPath(sessionId, observer), { method: "GET", runId }),

    loadTimeline: ({ sessionId, afterSeq, limit, runId }) =>
      requestJson<TimelinePage>(buildTimelinePath(sessionId, { afterSeq, limit }), {
        method: "GET",
        runId,
      }),

    loadScene: ({ sessionId, seq, runId, observer }) =>
      requestJson<VisualScene>(buildScenePath(sessionId, { seq, observer }), {
        method: "GET",
        runId,
      }),

    loadMarkers: ({ sessionId, marker, runId }) =>
      requestJson<{ sessionId: string; marker: string; seqs: number[] }>(
        `${buildSessionPath(sessionId)}/markers?marker=${encodeURIComponent(marker)}`,
        { method: "GET", runId },
      ),

    loadMedia: ({ sessionId, mediaId, runId }) =>
      requestJson<MediaSourceRef>(`${buildSessionPath(sessionId)}/media/${encodeURIComponent(mediaId)}`, {
        method: "GET",
        runId,
      }),

    submitAction: ({ sessionId, payload, runId }) =>
      requestJson<ActionIntentReceipt>(`${buildSessionPath(sessionId)}/actions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        runId,
      }),

    buildMediaUrl: ({ sessionId, mediaId, runId }) => {
      const url = new URL(
        `${normalizedBaseUrl}${buildSessionPath(sessionId)}/media/${encodeURIComponent(mediaId)}`,
      );
      url.searchParams.set("content", "1");
      if (runId) {
        url.searchParams.set("run_id", runId);
      }
      return url.toString();
    },
  };
}

function buildSessionPath(sessionId: string, observer?: ObserverRef): string {
  const query = new URLSearchParams();
  appendObserverParams(query, observer);
  const suffix = query.size > 0 ? `?${query.toString()}` : "";
  return `/arena_visual/sessions/${encodeURIComponent(sessionId)}${suffix}`;
}

function buildTimelinePath(
  sessionId: string,
  {
    afterSeq,
    limit,
  }: {
    afterSeq?: number | null;
    limit?: number;
  },
): string {
  const query = new URLSearchParams();
  if (afterSeq !== undefined && afterSeq !== null) {
    query.set("after_seq", String(afterSeq));
  }
  if (limit !== undefined) {
    query.set("limit", String(limit));
  }
  const suffix = query.size > 0 ? `?${query.toString()}` : "";
  return `${buildSessionPath(sessionId)}/timeline${suffix}`;
}

function buildScenePath(
  sessionId: string,
  {
    seq,
    observer,
  }: {
    seq: number;
    observer?: ObserverRef;
  },
): string {
  const query = new URLSearchParams();
  query.set("seq", String(seq));
  appendObserverParams(query, observer);
  return `${buildSessionPath(sessionId)}/scene?${query.toString()}`;
}

function appendObserverParams(query: URLSearchParams, observer?: ObserverRef): void {
  if (!observer) {
    return;
  }
  query.set("observer_kind", observer.observerKind);
  if (observer.observerId.trim() !== "") {
    query.set("observer_id", observer.observerId);
  }
}

async function parseResponseBody(response: Response): Promise<unknown> {
  const contentType = response.headers.get("Content-Type") ?? "";
  if (contentType.includes("application/json")) {
    return response.json();
  }

  const text = await response.text();
  if (!text) {
    return undefined;
  }

  try {
    return JSON.parse(text) as unknown;
  } catch {
    return text;
  }
}

function toGatewayError(status: number, payload: unknown): ArenaGatewayError {
  const errorPayload = getErrorPayload(payload);
  return new ArenaGatewayError({
    message: errorPayload.message ?? `Request failed with status ${status}`,
    code: errorPayload.code ?? `http_${status}`,
    status,
  });
}

function getErrorPayload(payload: unknown): { message?: string; code?: string } {
  if (!payload || typeof payload !== "object") {
    return {};
  }

  const direct = payload as Record<string, unknown>;
  const nested = direct.error;
  if (typeof nested === "string") {
    return {
      message: nested,
      code: nested,
    };
  }
  if (nested && typeof nested === "object") {
    const record = nested as Record<string, unknown>;
    return {
      message: typeof record.message === "string" ? record.message : undefined,
      code: typeof record.code === "string" ? record.code : undefined,
    };
  }

  return {
    message: typeof direct.message === "string" ? direct.message : undefined,
    code: typeof direct.code === "string" ? direct.code : undefined,
  };
}

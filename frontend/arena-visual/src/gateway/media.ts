import type { ArenaGatewayClient } from "./client";
import type { MediaSourceRef } from "./types";

export interface MediaSubscriptionRequest {
  sessionId: string;
  mediaId: string;
  runId?: string;
}

export type ResolvedMediaStatus = "loading" | "ready" | "error";

export interface ResolvedMediaSource {
  mediaId: string;
  ref?: MediaSourceRef;
  src?: string;
  status: ResolvedMediaStatus;
  error?: unknown;
}

type MediaListener = (state: ResolvedMediaSource) => void;

interface MediaEntry {
  listeners: Set<MediaListener>;
  state: ResolvedMediaSource;
  objectUrl?: string;
  loadPromise?: Promise<void>;
  disposed?: boolean;
}

export interface ArenaMediaResolver {
  resolve(request: MediaSubscriptionRequest): Promise<ResolvedMediaSource>;
  subscribe(request: MediaSubscriptionRequest, listener: MediaListener): () => void;
}

type MediaResolverClient = Pick<ArenaGatewayClient, "loadMedia"> &
  Partial<Pick<ArenaGatewayClient, "buildMediaUrl">>;

export function createArenaMediaResolver(
  client: MediaResolverClient,
): ArenaMediaResolver {
  const entries = new Map<string, MediaEntry>();

  const ensureEntry = (request: MediaSubscriptionRequest): MediaEntry => {
    const key = createKey(request);
    let entry = entries.get(key);
    if (!entry) {
      entry = {
        listeners: new Set(),
        state: {
          mediaId: request.mediaId,
          status: "loading",
        },
      };
      entries.set(key, entry);
    }
    entry.disposed = false;

    if (!entry.loadPromise) {
      entry.loadPromise = loadIntoEntry(request, entry);
    }
    return entry;
  };

  const notify = (entry: MediaEntry): void => {
    for (const listener of entry.listeners) {
      listener(entry.state);
    }
  };

  const loadIntoEntry = async (
    request: MediaSubscriptionRequest,
    entry: MediaEntry,
  ): Promise<void> => {
    try {
      const ref = await client.loadMedia(request);
      const src = await resolveMediaSrc(client, request, ref);
      entry.state = {
        mediaId: request.mediaId,
        ref,
        src,
        status: "ready",
      };
      if (src.startsWith("blob:")) {
        entry.objectUrl = src;
      }
      if (entry.disposed) {
        if (entry.objectUrl) {
          URL.revokeObjectURL(entry.objectUrl);
        }
        entries.delete(createKey(request));
        return;
      }
    } catch (error) {
      entry.state = {
        mediaId: request.mediaId,
        status: "error",
        error,
      };
      entry.loadPromise = undefined;
    }
    notify(entry);
  };

  return {
    async resolve(request) {
      const entry = ensureEntry(request);
      await entry.loadPromise;
      return entry.state;
    },

    subscribe(request, listener) {
      const key = createKey(request);
      const entry = ensureEntry(request);
      entry.listeners.add(listener);
      listener(entry.state);

      if (entry.state.status !== "ready" && entry.state.status !== "error") {
        void entry.loadPromise;
      }

      return () => {
        const currentEntry = entries.get(key);
        if (!currentEntry) {
          return;
        }

        currentEntry.listeners.delete(listener);
        if (currentEntry.listeners.size > 0) {
          return;
        }

        currentEntry.disposed = true;
        if (currentEntry.objectUrl) {
          URL.revokeObjectURL(currentEntry.objectUrl);
          entries.delete(key);
          return;
        }
        if (currentEntry.state.status !== "loading") {
          entries.delete(key);
        }
      };
    },
  };
}

function createKey({ sessionId, mediaId, runId }: MediaSubscriptionRequest): string {
  return `${sessionId}::${mediaId}::${runId ?? ""}`;
}

async function resolveMediaSrc(
  client: Partial<Pick<ArenaGatewayClient, "buildMediaUrl">>,
  request: MediaSubscriptionRequest,
  ref: MediaSourceRef,
): Promise<string> {
  if (ref.url && ref.transport !== "binary_stream" && isAbsoluteMediaUrl(ref.url)) {
    return ref.url;
  }

  const fallbackUrl =
    typeof client.buildMediaUrl === "function"
      ? client.buildMediaUrl(request)
      : `/arena_visual/sessions/${encodeURIComponent(request.sessionId)}/media/${encodeURIComponent(
          request.mediaId,
        )}?content=1`;
  if (ref.url && ref.transport !== "binary_stream" && !ref.url.startsWith("blob:")) {
    return fallbackUrl;
  }
  const response = await fetch(fallbackUrl);
  if (!response.ok) {
    throw new Error(`Media fetch failed with status ${response.status}`);
  }

  const blob = await response.blob();
  return URL.createObjectURL(blob);
}

function isAbsoluteMediaUrl(url: string): boolean {
  return /^(https?:\/\/|data:)/i.test(url);
}

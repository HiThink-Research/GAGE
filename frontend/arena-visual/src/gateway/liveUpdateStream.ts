import type { TimelinePage, VisualSession } from "./types";

interface ArenaLiveUpdateDelta {
  session?: VisualSession;
  timeline?: TimelinePage;
}

interface ArenaLiveUpdateStreamOptions {
  url: string;
  onDelta: (delta: ArenaLiveUpdateDelta) => void;
  onError?: () => void;
  EventSourceCtor?: typeof EventSource;
}

export interface ArenaLiveUpdateStream {
  close(): void;
}

export function createArenaLiveUpdateStream({
  url,
  onDelta,
  onError,
  EventSourceCtor = globalThis.EventSource,
}: ArenaLiveUpdateStreamOptions): ArenaLiveUpdateStream {
  if (!EventSourceCtor) {
    return {
      close() {
        // EventSource is unavailable in this environment.
      },
    };
  }

  const source = new EventSourceCtor(url);
  source.addEventListener("delta", (event) => {
    const payload = JSON.parse((event as MessageEvent<string>).data) as ArenaLiveUpdateDelta;
    onDelta(payload);
  });
  source.onerror = () => {
    if (typeof onError === "function") {
      onError();
    }
  };

  return {
    close(): void {
      source.close();
    },
  };
}

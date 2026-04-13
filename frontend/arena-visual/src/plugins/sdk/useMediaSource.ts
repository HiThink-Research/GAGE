import { useEffect, useState } from "react";

import type {
  MediaSubscriptionRequest,
  ResolvedMediaSource,
} from "../../gateway/media";

interface UseMediaSourceOptions extends MediaSubscriptionRequest {
  subscribe: (
    request: MediaSubscriptionRequest,
    listener: (state: ResolvedMediaSource) => void,
  ) => () => void;
}

interface StoredMediaState {
  scopeKey: string;
  source: ResolvedMediaSource;
}

export function useMediaSource({
  sessionId,
  mediaId,
  runId,
  subscribe,
}: UseMediaSourceOptions): ResolvedMediaSource | undefined {
  const [state, setState] = useState<StoredMediaState>();
  const scopeKey = `${sessionId}::${runId ?? ""}`;

  useEffect(() => {
    if (!mediaId) {
      setState(undefined);
      return undefined;
    }

    return subscribe({ sessionId, mediaId, runId }, (nextState) => {
      setState((currentState) => {
        if (
          nextState.status === "loading" &&
          currentState?.scopeKey === scopeKey &&
          currentState.source.status === "ready" &&
          typeof currentState.source.src === "string" &&
          currentState.source.src !== ""
        ) {
          return currentState;
        }
        return {
          scopeKey,
          source: nextState,
        };
      });
    });
  }, [mediaId, runId, sessionId, subscribe]);

  if (!mediaId) {
    return undefined;
  }
  return state?.scopeKey === scopeKey ? state.source : undefined;
}

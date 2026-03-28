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

export function useMediaSource({
  sessionId,
  mediaId,
  runId,
  subscribe,
}: UseMediaSourceOptions): ResolvedMediaSource | undefined {
  const [state, setState] = useState<ResolvedMediaSource>();

  useEffect(() => {
    if (!mediaId) {
      setState(undefined);
      return undefined;
    }

    return subscribe({ sessionId, mediaId, runId }, setState);
  }, [mediaId, runId, sessionId, subscribe]);

  return state;
}

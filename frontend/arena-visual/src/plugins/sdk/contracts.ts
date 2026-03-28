import type { ComponentType } from "react";

import type {
  ActionIntentReceipt,
  GamePluginManifest,
  VisualScene,
  VisualSession,
} from "../../gateway/types";
import type {
  MediaSubscriptionRequest,
  ResolvedMediaSource,
} from "../../gateway/media";

export type ArenaMediaSubscriber = (
  request: MediaSubscriptionRequest,
  listener: (state: ResolvedMediaSource) => void,
) => () => void;

export interface ArenaPluginRenderProps {
  session: VisualSession;
  scene?: VisualScene;
  latestActionReceipt?: ActionIntentReceipt;
  submitAction: (payload: Record<string, unknown>) => Promise<void>;
  mediaSubscribe: ArenaMediaSubscriber;
  isFallback: boolean;
  requestedPluginId?: string;
}

export interface ArenaPluginDefinition {
  pluginId: string;
  displayName: string;
  manifest: GamePluginManifest;
  render: ComponentType<ArenaPluginRenderProps>;
  isFallback: boolean;
  requestedPluginId?: string;
}

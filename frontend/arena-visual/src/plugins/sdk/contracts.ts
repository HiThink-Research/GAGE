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
import type { ActionIntent, InputInterpreter } from "./input";

export type ArenaMediaSubscriber = (
  request: MediaSubscriptionRequest,
  listener: (state: ResolvedMediaSource) => void,
) => () => void;

export interface ArenaPluginRenderProps<
  TDeviceEvent = never,
  TIntent extends ActionIntent = ActionIntent,
> {
  session: VisualSession;
  scene?: VisualScene;
  latestActionReceipt?: ActionIntentReceipt;
  submitAction: (payload: TIntent) => Promise<void>;
  submitInput: (event: TDeviceEvent) => Promise<void>;
  mediaSubscribe: ArenaMediaSubscriber;
  isFallback: boolean;
  requestedPluginId?: string;
}

export interface ArenaPluginDefinition<
  TDeviceEvent = never,
  TIntent extends ActionIntent = ActionIntent,
> {
  pluginId: string;
  displayName: string;
  manifest: GamePluginManifest;
  render: ComponentType<ArenaPluginRenderProps<TDeviceEvent, TIntent>>;
  inputInterpreter?: InputInterpreter<TDeviceEvent, TIntent>;
  isFallback: boolean;
  requestedPluginId?: string;
}

export type AnyArenaPluginDefinition = ArenaPluginDefinition<any, ActionIntent>;

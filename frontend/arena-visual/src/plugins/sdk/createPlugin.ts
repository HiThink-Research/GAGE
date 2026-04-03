import type { ComponentType } from "react";

import type { GamePluginManifest } from "../../gateway/types";
import type { ArenaPluginDefinition, ArenaPluginRenderProps } from "./contracts";
import type { ActionIntent, InputInterpreter } from "./input";

interface CreatePluginInput<
  TDeviceEvent = never,
  TIntent extends ActionIntent = ActionIntent,
> {
  pluginId: string;
  displayName: string;
  operatorHint?: string;
  manifest: GamePluginManifest;
  render: ComponentType<ArenaPluginRenderProps<TDeviceEvent, TIntent>>;
  inputInterpreter?: InputInterpreter<TDeviceEvent, TIntent>;
  isFallback?: boolean;
  requestedPluginId?: string;
}

export function createPlugin<
  TDeviceEvent = never,
  TIntent extends ActionIntent = ActionIntent,
>({
  pluginId,
  displayName,
  operatorHint,
  manifest,
  render,
  inputInterpreter,
  isFallback = false,
  requestedPluginId,
}: CreatePluginInput<TDeviceEvent, TIntent>): ArenaPluginDefinition<TDeviceEvent, TIntent> {
  return {
    pluginId,
    displayName,
    operatorHint,
    manifest,
    render,
    inputInterpreter,
    isFallback,
    requestedPluginId,
  };
}

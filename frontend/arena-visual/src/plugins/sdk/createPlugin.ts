import type { ComponentType } from "react";

import type { GamePluginManifest } from "../../gateway/types";
import type { ArenaPluginDefinition, ArenaPluginRenderProps } from "./contracts";

interface CreatePluginInput {
  pluginId: string;
  displayName: string;
  manifest?: GamePluginManifest;
  render: ComponentType<ArenaPluginRenderProps>;
  isFallback?: boolean;
  requestedPluginId?: string;
}

export function createPlugin({
  pluginId,
  displayName,
  manifest = {
    sceneKinds: [],
    supportedObservers: [],
    acceptsHumanIntent: false,
  },
  render,
  isFallback = false,
  requestedPluginId,
}: CreatePluginInput): ArenaPluginDefinition {
  return {
    pluginId,
    displayName,
    manifest,
    render,
    isFallback,
    requestedPluginId,
  };
}

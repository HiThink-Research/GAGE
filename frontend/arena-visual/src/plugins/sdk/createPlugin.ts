import type { ComponentType } from "react";

import type { ArenaPluginDefinition, ArenaPluginRenderProps } from "./contracts";

interface CreatePluginInput {
  pluginId: string;
  displayName: string;
  render: ComponentType<ArenaPluginRenderProps>;
  isFallback?: boolean;
  requestedPluginId?: string;
}

export function createPlugin({
  pluginId,
  displayName,
  render,
  isFallback = false,
  requestedPluginId,
}: CreatePluginInput): ArenaPluginDefinition {
  return {
    pluginId,
    displayName,
    render,
    isFallback,
    requestedPluginId,
  };
}

import { describe, expect, it } from "vitest";

import { resolveArenaPlugin } from "./registry";

describe("resolveArenaPlugin", () => {
  it("resolves known plugins by exact plugin_id", () => {
    const plugin = resolveArenaPlugin("arena.visualization.gomoku.board_v1");
    expect(plugin.pluginId).toBe("arena.visualization.gomoku.board_v1");
    expect(plugin.isFallback).toBe(false);
  });

  it("returns a host fallback renderer for unknown plugin_id", () => {
    const plugin = resolveArenaPlugin("arena.visualization.unknown.future_v1");
    expect(plugin.isFallback).toBe(true);
    expect(plugin.requestedPluginId).toBe("arena.visualization.unknown.future_v1");
  });
});

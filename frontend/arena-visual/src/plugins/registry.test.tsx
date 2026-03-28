import { describe, expect, it } from "vitest";

import { resolveArenaPlugin } from "./registry";

describe("resolveArenaPlugin", () => {
  const knownPluginIds = [
    "arena.visualization.gomoku.board_v1",
    "arena.visualization.tictactoe.board_v1",
    "arena.visualization.doudizhu.table_v1",
    "arena.visualization.mahjong.table_v1",
    "arena.visualization.pettingzoo.frame_v1",
    "arena.visualization.vizdoom.frame_v1",
    "arena.visualization.retro_platformer.frame_v1",
  ] as const;

  it("resolves known plugins by exact plugin_id", () => {
    const plugin = resolveArenaPlugin("arena.visualization.gomoku.board_v1");
    expect(plugin.pluginId).toBe("arena.visualization.gomoku.board_v1");
    expect(plugin.isFallback).toBe(false);
  });

  it("exposes non-empty manifests for known shipped plugins", () => {
    for (const pluginId of knownPluginIds) {
      const plugin = resolveArenaPlugin(pluginId);
      expect(plugin.isFallback).toBe(false);
      expect(plugin.manifest.sceneKinds.length).toBeGreaterThan(0);
      expect(plugin.manifest.supportedObservers.length).toBeGreaterThan(0);
    }
  });

  it("returns a host fallback renderer for unknown plugin_id", () => {
    const plugin = resolveArenaPlugin("arena.visualization.unknown.future_v1");
    expect(plugin.isFallback).toBe(true);
    expect(plugin.requestedPluginId).toBe("arena.visualization.unknown.future_v1");
    expect(plugin.manifest.sceneKinds).toEqual(["frame"]);
    expect(plugin.manifest.supportedObservers).toEqual(["global"]);
    expect(plugin.manifest.acceptsHumanIntent).toBe(false);
  });
});

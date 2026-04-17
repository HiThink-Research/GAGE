import { describe, expect, it } from "vitest";

import { resolveArenaPlugin } from "./registry";

describe("resolveArenaPlugin", () => {
  const expectedKnownPluginManifests = {
    "arena.visualization.gomoku.board_v1": {
      sceneKinds: ["board"],
      supportedObservers: ["player", "global"],
      acceptsHumanIntent: true,
    },
    "arena.visualization.tictactoe.board_v1": {
      sceneKinds: ["board"],
      supportedObservers: ["player", "global"],
      acceptsHumanIntent: true,
    },
    "arena.visualization.doudizhu.table_v1": {
      sceneKinds: ["table"],
      supportedObservers: ["global", "spectator", "camera", "player"],
      acceptsHumanIntent: true,
      layoutMode: "wide-stage",
    },
    "arena.visualization.mahjong.table_v1": {
      sceneKinds: ["table"],
      supportedObservers: ["global", "spectator", "camera", "player"],
      acceptsHumanIntent: true,
      layoutMode: "wide-stage",
    },
    "arena.visualization.pettingzoo.frame_v1": {
      sceneKinds: ["frame"],
      supportedObservers: ["player", "global"],
      acceptsHumanIntent: true,
    },
    "arena.visualization.vizdoom.frame_v1": {
      sceneKinds: ["frame"],
      supportedObservers: ["player", "camera"],
      acceptsHumanIntent: true,
    },
    "arena.visualization.retro_platformer.frame_v1": {
      sceneKinds: ["frame"],
      supportedObservers: ["player", "camera"],
      acceptsHumanIntent: true,
    },
  } as const;

  it("resolves known plugins by exact plugin_id", () => {
    const plugin = resolveArenaPlugin("arena.visualization.gomoku.board_v1");
    expect(plugin.pluginId).toBe("arena.visualization.gomoku.board_v1");
    expect(plugin.isFallback).toBe(false);
    expect(plugin.inputInterpreter).toBeDefined();
  });

  it("exposes exact manifests for known shipped plugins", () => {
    for (const [pluginId, expectedManifest] of Object.entries(
      expectedKnownPluginManifests,
    )) {
      const plugin = resolveArenaPlugin(pluginId);
      expect(plugin.isFallback).toBe(false);
      expect(plugin.manifest).toEqual(expectedManifest);
      expect(plugin.inputInterpreter).toBeDefined();
    }
  });

  it("returns a host fallback renderer for unknown plugin_id", () => {
    const plugin = resolveArenaPlugin("arena.visualization.unknown.future_v1");
    expect(plugin.isFallback).toBe(true);
    expect(plugin.requestedPluginId).toBe("arena.visualization.unknown.future_v1");
    expect(plugin.manifest.sceneKinds).toEqual(["frame"]);
    expect(plugin.manifest.supportedObservers).toEqual(["global"]);
    expect(plugin.manifest.acceptsHumanIntent).toBe(false);
    expect(plugin.inputInterpreter).toBeUndefined();
  });
});

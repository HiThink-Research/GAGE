import { describe, expect, it } from "vitest";

import type { GamePluginManifest } from "../../gateway/types";
import { createPlugin } from "./createPlugin";

describe("createPlugin", () => {
  it("preserves plugin metadata manifest and render entrypoint", () => {
    const render = () => null;
    const manifest: GamePluginManifest = {
      sceneKinds: ["board"],
      supportedObservers: ["player", "global"],
      acceptsHumanIntent: true,
      extensionPanels: ["timeline"],
    };
    const plugin = createPlugin({
      pluginId: "arena.visualization.gomoku.board_v1",
      displayName: "Gomoku",
      manifest,
      render,
    });

    expect(plugin.pluginId).toBe("arena.visualization.gomoku.board_v1");
    expect(plugin.displayName).toBe("Gomoku");
    expect(plugin.manifest).toEqual(manifest);
    expect(plugin.manifest.supportedObservers).toEqual(["player", "global"]);
    expect(plugin.render).toBe(render);
    expect(plugin.isFallback).toBe(false);
  });

  it("preserves explicit fallback metadata", () => {
    const plugin = createPlugin({
      pluginId: "arena.visualization.host.fallback_v1",
      displayName: "Host Fallback",
      manifest: {
        sceneKinds: ["frame"],
        supportedObservers: ["global"],
        acceptsHumanIntent: false,
      },
      render: () => null,
      isFallback: true,
      requestedPluginId: "arena.visualization.unknown.future_v1",
    });

    expect(plugin.isFallback).toBe(true);
    expect(plugin.requestedPluginId).toBe("arena.visualization.unknown.future_v1");
    expect(plugin.manifest.sceneKinds).toEqual(["frame"]);
    expect(plugin.manifest.supportedObservers).toEqual(["global"]);
  });

  it("preserves optional operator hints for shell-level guidance", () => {
    const plugin = createPlugin({
      pluginId: "arena.visualization.retro_platformer.frame_v1",
      displayName: "Retro Mario",
      manifest: {
        sceneKinds: ["frame"],
        supportedObservers: ["player", "camera"],
        acceptsHumanIntent: true,
      },
      render: () => null,
      operatorHint: "Keyboard: arrows/WASD move, Space/J/Z jump, X/K run.",
    });

    expect(plugin.operatorHint).toBe(
      "Keyboard: arrows/WASD move, Space/J/Z jump, X/K run.",
    );
  });
});

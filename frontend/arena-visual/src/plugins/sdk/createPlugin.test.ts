import { describe, expect, it } from "vitest";

import { createPlugin } from "./createPlugin";

describe("createPlugin", () => {
  it("preserves plugin metadata and render entrypoint", () => {
    const render = () => null;
    const plugin = createPlugin({
      pluginId: "arena.visualization.gomoku.board_v1",
      displayName: "Gomoku",
      manifest: {
        sceneKinds: ["board"],
        supportedObservers: ["player", "global"],
        acceptsHumanIntent: true,
      },
      render,
    });

    expect(plugin.pluginId).toBe("arena.visualization.gomoku.board_v1");
    expect(plugin.displayName).toBe("Gomoku");
    expect(plugin.manifest.supportedObservers).toEqual(["player", "global"]);
    expect(plugin.render).toBe(render);
    expect(plugin.isFallback).toBe(false);
  });
});

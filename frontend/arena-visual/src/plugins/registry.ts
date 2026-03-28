import { createElement } from "react";

import type { ArenaPluginRenderProps, ArenaPluginDefinition } from "./sdk/contracts";
import { createPlugin } from "./sdk/createPlugin";
import { DoudizhuPlugin } from "./doudizhu/DoudizhuPlugin";
import { GomokuPlugin } from "./gomoku/GomokuPlugin";
import { MahjongPlugin } from "./mahjong/MahjongPlugin";
import { PettingZooPlugin } from "./pettingzoo/PettingZooPlugin";
import { RetroMarioPlugin } from "./retro-mario/RetroMarioPlugin";
import { TicTacToePlugin } from "./tictactoe/TicTacToePlugin";
import { VizDoomPlugin } from "./vizdoom/VizDoomPlugin";

const KNOWN_PLUGIN_META = [
  {
    pluginId: "arena.visualization.gomoku.board_v1",
    displayName: "Gomoku",
  },
  {
    pluginId: "arena.visualization.tictactoe.board_v1",
    displayName: "Tic-Tac-Toe",
  },
  {
    pluginId: "arena.visualization.doudizhu.table_v1",
    displayName: "Doudizhu",
  },
  {
    pluginId: "arena.visualization.mahjong.table_v1",
    displayName: "Mahjong",
  },
  {
    pluginId: "arena.visualization.pettingzoo.frame_v1",
    displayName: "PettingZoo",
  },
  {
    pluginId: "arena.visualization.vizdoom.frame_v1",
    displayName: "VizDoom",
  },
  {
    pluginId: "arena.visualization.retro_platformer.frame_v1",
    displayName: "Retro Mario",
  },
] as const;

function PlaceholderPluginStage({
  session,
  scene,
  requestedPluginId,
  isFallback,
}: ArenaPluginRenderProps) {
  return createElement(
    "section",
    { className: "plugin-stage-card" },
    createElement(
      "p",
      { className: "eyebrow" },
      isFallback ? "Fallback Plugin" : "Registered Plugin",
    ),
    createElement(
      "h2",
      null,
      isFallback ? requestedPluginId ?? "Unknown plugin" : session.pluginId,
    ),
    createElement(
      "p",
      { className: "plugin-stage-card__copy" },
      isFallback
        ? "This plugin is not implemented yet, so the host fallback surface is rendering the current session."
        : "The host shell is active. Game-specific rendering will replace this placeholder in the next slice.",
    ),
    createElement(
      "dl",
      { className: "plugin-stage-card__meta" },
      createElement(
        "div",
        null,
        createElement("dt", null, "Game"),
        createElement("dd", null, session.gameId),
      ),
      createElement(
        "div",
        null,
        createElement("dt", null, "Lifecycle"),
        createElement("dd", null, session.lifecycle),
      ),
      createElement(
        "div",
        null,
        createElement("dt", null, "Scene"),
        createElement("dd", null, scene ? `${scene.kind} · seq ${scene.seq}` : "Not loaded yet"),
      ),
    ),
  );
}

const KNOWN_PLUGINS = new Map<string, ArenaPluginDefinition>(
  KNOWN_PLUGIN_META.map(({ pluginId, displayName }) => [
    pluginId,
    createPlugin({
      pluginId,
      displayName,
      render:
        pluginId === "arena.visualization.gomoku.board_v1"
          ? GomokuPlugin
          : pluginId === "arena.visualization.tictactoe.board_v1"
            ? TicTacToePlugin
            : pluginId === "arena.visualization.doudizhu.table_v1"
              ? DoudizhuPlugin
              : pluginId === "arena.visualization.mahjong.table_v1"
                ? MahjongPlugin
                : pluginId === "arena.visualization.pettingzoo.frame_v1"
                  ? PettingZooPlugin
                  : pluginId === "arena.visualization.vizdoom.frame_v1"
                    ? VizDoomPlugin
                    : pluginId === "arena.visualization.retro_platformer.frame_v1"
                      ? RetroMarioPlugin
                      : PlaceholderPluginStage,
    }),
  ]),
);

export function resolveArenaPlugin(pluginId: string): ArenaPluginDefinition {
  const plugin = KNOWN_PLUGINS.get(pluginId);
  if (plugin) {
    return plugin;
  }

  return createPlugin({
    pluginId: "arena.visualization.host.fallback_v1",
    displayName: "Host Fallback",
    render: PlaceholderPluginStage,
    isFallback: true,
    requestedPluginId: pluginId,
  });
}

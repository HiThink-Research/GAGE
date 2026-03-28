import { createElement } from "react";

import type { GamePluginManifest } from "../gateway/types";
import type {
  ArenaPluginRenderProps,
  AnyArenaPluginDefinition,
} from "./sdk/contracts";
import { createPlugin } from "./sdk/createPlugin";
import { createInputInterpreter, type ActionIntent } from "./sdk/input";
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
    manifest: {
      sceneKinds: ["board"],
      supportedObservers: ["player", "global"],
      acceptsHumanIntent: true,
    },
  },
  {
    pluginId: "arena.visualization.tictactoe.board_v1",
    displayName: "Tic-Tac-Toe",
    manifest: {
      sceneKinds: ["board"],
      supportedObservers: ["player", "global"],
      acceptsHumanIntent: true,
    },
  },
  {
    pluginId: "arena.visualization.doudizhu.table_v1",
    displayName: "Doudizhu",
    manifest: {
      sceneKinds: ["table"],
      supportedObservers: ["player", "global"],
      acceptsHumanIntent: true,
    },
  },
  {
    pluginId: "arena.visualization.mahjong.table_v1",
    displayName: "Mahjong",
    manifest: {
      sceneKinds: ["table"],
      supportedObservers: ["player", "global"],
      acceptsHumanIntent: true,
    },
  },
  {
    pluginId: "arena.visualization.pettingzoo.frame_v1",
    displayName: "PettingZoo",
    manifest: {
      sceneKinds: ["frame"],
      supportedObservers: ["player", "global"],
      acceptsHumanIntent: true,
    },
  },
  {
    pluginId: "arena.visualization.vizdoom.frame_v1",
    displayName: "VizDoom",
    manifest: {
      sceneKinds: ["frame"],
      supportedObservers: ["player", "camera"],
      acceptsHumanIntent: true,
    },
  },
  {
    pluginId: "arena.visualization.retro_platformer.frame_v1",
    displayName: "Retro Mario",
    manifest: {
      sceneKinds: ["frame"],
      supportedObservers: ["player", "camera"],
      acceptsHumanIntent: true,
    },
  },
] as const satisfies ReadonlyArray<{
  pluginId: string;
  displayName: string;
  manifest: GamePluginManifest;
}>;

function PlaceholderPluginStage({
  session,
  scene,
  requestedPluginId,
  isFallback,
}: ArenaPluginRenderProps<any>) {
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

interface BoardDeviceEvent {
  playerId: string;
  coord: string;
}

interface TableDeviceEvent {
  playerId: string;
  actionText: string;
}

interface FrameDeviceEvent {
  playerId: string;
  actionPayload: ActionIntent["action"];
}

const boardInputInterpreter = createInputInterpreter<BoardDeviceEvent>(
  ({ playerId, coord }) => ({
    playerId,
    action: { move: coord },
  }),
);

const tableInputInterpreter = createInputInterpreter<TableDeviceEvent>(
  ({ playerId, actionText }) => ({
    playerId,
    action: { move: actionText },
  }),
);

const frameInputInterpreter = createInputInterpreter<FrameDeviceEvent>(
  ({ playerId, actionPayload }) => ({
    playerId,
    action: actionPayload,
  }),
);

const KNOWN_PLUGINS = new Map<string, AnyArenaPluginDefinition>([
  [
    "arena.visualization.gomoku.board_v1",
    createPlugin({
      ...KNOWN_PLUGIN_META[0],
      render: GomokuPlugin,
      inputInterpreter: boardInputInterpreter,
    }),
  ],
  [
    "arena.visualization.tictactoe.board_v1",
    createPlugin({
      ...KNOWN_PLUGIN_META[1],
      render: TicTacToePlugin,
      inputInterpreter: boardInputInterpreter,
    }),
  ],
  [
    "arena.visualization.doudizhu.table_v1",
    createPlugin({
      ...KNOWN_PLUGIN_META[2],
      render: DoudizhuPlugin,
      inputInterpreter: tableInputInterpreter,
    }),
  ],
  [
    "arena.visualization.mahjong.table_v1",
    createPlugin({
      ...KNOWN_PLUGIN_META[3],
      render: MahjongPlugin,
      inputInterpreter: tableInputInterpreter,
    }),
  ],
  [
    "arena.visualization.pettingzoo.frame_v1",
    createPlugin({
      ...KNOWN_PLUGIN_META[4],
      render: PettingZooPlugin,
      inputInterpreter: frameInputInterpreter,
    }),
  ],
  [
    "arena.visualization.vizdoom.frame_v1",
    createPlugin({
      ...KNOWN_PLUGIN_META[5],
      render: VizDoomPlugin,
      inputInterpreter: frameInputInterpreter,
    }),
  ],
  [
    "arena.visualization.retro_platformer.frame_v1",
    createPlugin({
      ...KNOWN_PLUGIN_META[6],
      render: RetroMarioPlugin,
      inputInterpreter: frameInputInterpreter,
    }),
  ],
]);

export function resolveArenaPlugin(pluginId: string): AnyArenaPluginDefinition {
  const plugin = KNOWN_PLUGINS.get(pluginId);
  if (plugin) {
    return plugin;
  }

  return createPlugin({
    pluginId: "arena.visualization.host.fallback_v1",
    displayName: "Host Fallback",
    manifest: {
      sceneKinds: ["frame"],
      supportedObservers: ["global"],
      acceptsHumanIntent: false,
    },
    render: PlaceholderPluginStage,
    isFallback: true,
    requestedPluginId: pluginId,
  });
}

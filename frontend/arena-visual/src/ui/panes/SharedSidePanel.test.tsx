import { fireEvent, render, screen } from "@testing-library/react";

import type { VisualScene } from "../../gateway/types";
import { SharedSidePanel } from "./SharedSidePanel";

describe("SharedSidePanel", () => {
  it("renders controlled scene panels such as chat logs", () => {
    render(
      <SharedSidePanel
        session={{
          sessionId: "doudizhu-sample",
          gameId: "doudizhu",
          pluginId: "arena.visualization.doudizhu.table_v1",
          lifecycle: "live_running",
          playback: {
            mode: "paused",
            cursorTs: 1007,
            cursorEventSeq: 7,
            speed: 1,
            canSeek: true,
          },
          observer: {
            observerId: "player_0",
            observerKind: "player",
          },
          scheduling: {
            family: "turn",
            phase: "waiting_for_intent",
            acceptsHumanIntent: true,
            activeActorId: "player_0",
          },
          capabilities: {},
          summary: {},
          timeline: {},
        }}
        scene={
          {
            sceneId: "doudizhu-sample:seq:7",
            gameId: "doudizhu",
            pluginId: "arena.visualization.doudizhu.table_v1",
            kind: "table",
            tsMs: 1007,
            seq: 7,
            phase: "replay",
            activePlayerId: "player_0",
            legalActions: [],
            summary: {},
            body: {
              panels: {
                chatLog: [
                  {
                    playerId: "player_1",
                    text: "watch this",
                  },
                ],
              },
            },
          } as VisualScene
        }
      />,
    );

    expect(screen.getByRole("heading", { name: /chat log/i })).toBeInTheDocument();
    expect(screen.getByText(/player_1/i)).toBeInTheDocument();
    expect(screen.getByText(/watch this/i)).toBeInTheDocument();
  });

  it("offers a neutral observer option and scene-derived player observer options", () => {
    const onObserverChange = vi.fn();

    render(
      <SharedSidePanel
        session={{
          sessionId: "gomoku-sample",
          gameId: "gomoku",
          pluginId: "arena.visualization.gomoku.board_v1",
          lifecycle: "live_running",
          playback: {
            mode: "paused",
            cursorTs: 1007,
            cursorEventSeq: 7,
            speed: 1,
            canSeek: true,
          },
          observer: {
            observerId: "host",
            observerKind: "global",
          },
          scheduling: {
            family: "turn",
            phase: "waiting_for_intent",
            acceptsHumanIntent: true,
            activeActorId: "player_0",
          },
          capabilities: {},
          summary: {},
          timeline: {},
        }}
        scene={
          {
            sceneId: "gomoku-sample:seq:7",
            gameId: "gomoku",
            pluginId: "arena.visualization.gomoku.board_v1",
            kind: "board",
            tsMs: 1007,
            seq: 7,
            phase: "replay",
            activePlayerId: "player_0",
            legalActions: [],
            summary: {},
            body: {
              players: [
                { playerId: "player_0", playerName: "Alpha", token: "X" },
                { playerId: "player_1", playerName: "Beta", token: "O" },
              ],
            },
          } as VisualScene
        }
        onObserverChange={onObserverChange}
      />,
    );

    const selector = screen.getByLabelText(/observer view/i);
    expect(screen.getByRole("option", { name: /neutral observer/i })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: /Alpha/i })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: /Beta/i })).toBeInTheDocument();

    fireEvent.change(selector, { target: { value: "player:player_1" } });

    expect(onObserverChange).toHaveBeenCalledWith({
      observerId: "player_1",
      observerKind: "player",
    });
  });
});

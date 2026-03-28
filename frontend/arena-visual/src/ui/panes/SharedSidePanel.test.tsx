import { fireEvent, render, screen, within } from "@testing-library/react";

import type { VisualScene } from "../../gateway/types";
import doudizhuScene from "../../test/fixtures/doudizhu.visual.json";
import gomokuScene from "../../test/fixtures/gomoku.visual.json";
import { SharedSidePanel } from "./SharedSidePanel";

describe("SharedSidePanel", () => {
  it("renders fixed host tabs for players, events, chat, and trace", () => {
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

    expect(screen.getByRole("tab", { name: "Players" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Events" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Chat" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Trace" })).toBeInTheDocument();
  });

  it("reads observer options from frozen session capabilities and host labels", () => {
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
            observerId: "",
            observerKind: "camera",
          },
          scheduling: {
            family: "turn",
            phase: "waiting_for_intent",
            acceptsHumanIntent: true,
            activeActorId: "player_0",
          },
          capabilities: {
            observerModes: ["global", "camera", "player"],
          },
          summary: {},
          timeline: {},
        }}
        scene={gomokuScene as VisualScene}
        onObserverChange={onObserverChange}
      />,
    );

    const selector = screen.getByLabelText(/observer view/i);
    expect(screen.getByRole("option", { name: "Host overview" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "Camera view" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "Black" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "White" })).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: /neutral observer/i })).not.toBeInTheDocument();

    fireEvent.change(selector, { target: { value: "camera" } });

    expect(onObserverChange).toHaveBeenCalledWith({
      observerId: "",
      observerKind: "camera",
    });

    fireEvent.change(selector, { target: { value: "player:White" } });

    expect(onObserverChange).toHaveBeenCalledWith({
      observerId: "White",
      observerKind: "player",
    });
  });

  it("renders table-seat players in the Players tab for table scenes", () => {
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
          capabilities: {
            observerModes: ["global", "player"],
          },
          summary: {},
          timeline: {},
        }}
        scene={doudizhuScene as VisualScene}
      />,
    );

    const playersList = screen.getByRole("list");
    const items = within(playersList).getAllByRole("listitem");

    expect(items).toHaveLength(3);
    expect(within(items[0]!).getByText("Player 0")).toBeInTheDocument();
    expect(within(items[1]!).getByText("Player 1")).toBeInTheDocument();
    expect(within(items[2]!).getByText("Player 2")).toBeInTheDocument();
  });
});

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { VisualScene } from "../../gateway/types";
import doudizhuScene from "../../test/fixtures/doudizhu.visual.json";
import { DoudizhuPlugin } from "./DoudizhuPlugin";

describe("DoudizhuPlugin", () => {
  it("renders a non-blank table, masks other seats, and submits action chips", () => {
    const submitInput = vi.fn().mockResolvedValue(undefined);

    render(
      <DoudizhuPlugin
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
        scene={doudizhuScene as VisualScene}
        submitAction={vi.fn()}
        submitInput={submitInput}
        mediaSubscribe={() => () => {}}
        isFallback={false}
      />,
    );

    expect(screen.getByRole("heading", { name: /doudizhu table/i })).toBeInTheDocument();
    expect(screen.getByTestId("table-center-cards")).toHaveTextContent("3");
    expect(screen.getByTestId("seat-player_0-hand")).toHaveTextContent("3");
    expect(screen.getByTestId("seat-player_0-hand")).toHaveTextContent("4");
    expect(screen.getByTestId("seat-player_1-hand")).toHaveTextContent("Hidden hand");
    expect(screen.getByText(/observer: player_0/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /play pass/i }));
    expect(submitInput).toHaveBeenCalledWith({
      playerId: "player_0",
      actionText: "pass",
    });
  });
});

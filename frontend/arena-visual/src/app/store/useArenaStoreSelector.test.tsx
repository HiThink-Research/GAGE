import { act, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { ArenaSessionStoreSnapshot } from "./arenaSessionStore";
import { useArenaStoreSelector } from "./useArenaStoreSelector";

type StoreListener = () => void;

interface SelectorStore {
  getSnapshot: () => ArenaSessionStoreSnapshot;
  subscribe: (listener: StoreListener) => () => void;
}

interface MutableSelectorStore extends SelectorStore {
  setSnapshot: (snapshot: ArenaSessionStoreSnapshot) => void;
}

function createMutableStore(initialSnapshot: ArenaSessionStoreSnapshot): MutableSelectorStore {
  let snapshot = initialSnapshot;
  const listeners = new Set<StoreListener>();
  return {
    getSnapshot: () => snapshot,
    subscribe: (listener) => {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
    setSnapshot: (nextSnapshot) => {
      snapshot = nextSnapshot;
      for (const listener of listeners) {
        listener();
      }
    },
  };
}

function StageProbe({
  store,
  onRender,
}: {
  store: SelectorStore;
  onRender: (seq: number | undefined) => void;
}) {
  const scene = useArenaStoreSelector(store, (snapshot) => snapshot.scene);
  onRender(scene?.seq);
  return <div data-testid="stage-seq">{scene?.seq ?? "none"}</div>;
}

function TimelineProbe({
  store,
  onRender,
}: {
  store: SelectorStore;
  onRender: (count: number) => void;
}) {
  const eventCount = useArenaStoreSelector(store, (snapshot) => snapshot.timeline.events.length);
  onRender(eventCount);
  return <div data-testid="timeline-count">{eventCount}</div>;
}

describe("useArenaStoreSelector", () => {
  it("only rerenders consumers whose selected slice changed", () => {
    const initialSnapshot: ArenaSessionStoreSnapshot = {
      status: "ready",
      sceneStatus: "ready",
      scene: {
        sceneId: "sample-1:seq:1",
        gameId: "retro_platformer",
        pluginId: "arena.visualization.retro_platformer.frame_v1",
        kind: "frame",
        tsMs: 1000,
        seq: 1,
        phase: "live",
        activePlayerId: "mario",
        legalActions: [],
        summary: {},
        body: {
          frame: {
            title: "Retro Mario Frame",
            altText: "Retro Mario frame",
            fit: "contain",
          },
          status: {
            tick: 1,
            step: 1,
            moveCount: 1,
          },
        },
      },
      timeline: {
        status: "ready",
        events: [],
        nextAfterSeq: null,
        hasMore: false,
        limit: 50,
        filters: {
          eventTypes: [],
          severity: "all",
          humanIntentOnly: false,
        },
      },
    };
    const store = createMutableStore(initialSnapshot);
    const stageRenderSpy = vi.fn();
    const timelineRenderSpy = vi.fn();

    render(
      <>
        <StageProbe store={store} onRender={stageRenderSpy} />
        <TimelineProbe store={store} onRender={timelineRenderSpy} />
      </>,
    );

    expect(screen.getByTestId("stage-seq")).toHaveTextContent("1");
    expect(screen.getByTestId("timeline-count")).toHaveTextContent("0");
    expect(stageRenderSpy).toHaveBeenCalledTimes(1);
    expect(timelineRenderSpy).toHaveBeenCalledTimes(1);

    act(() => {
      store.setSnapshot({
        ...initialSnapshot,
        timeline: {
          ...initialSnapshot.timeline,
          events: [
            {
              seq: 2,
              tsMs: 1001,
              type: "snapshot",
              label: "snapshot",
            },
          ],
        },
      });
    });

    expect(screen.getByTestId("stage-seq")).toHaveTextContent("1");
    expect(screen.getByTestId("timeline-count")).toHaveTextContent("1");
    expect(stageRenderSpy).toHaveBeenCalledTimes(1);
    expect(timelineRenderSpy).toHaveBeenCalledTimes(2);
  });
});

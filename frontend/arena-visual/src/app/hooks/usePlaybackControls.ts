import type { ArenaSessionStore } from "../store/arenaSessionStore";

export function usePlaybackControls(store: ArenaSessionStore) {
  return {
    pause() {
      return store.submitControl({ commandType: "pause" });
    },
    playLive() {
      return store.submitControl({ commandType: "follow_tail" });
    },
    playReplay() {
      return store.submitControl({ commandType: "replay" });
    },
    selectEvent(seq: number) {
      return store.submitControl({
        commandType: "seek_seq",
        targetSeq: seq,
      });
    },
    stepBackward() {
      return store.submitControl({
        commandType: "step",
        stepDelta: -1,
      });
    },
    stepForward() {
      return store.submitControl({
        commandType: "step",
        stepDelta: 1,
      });
    },
    followTail() {
      return store.submitControl({ commandType: "follow_tail" });
    },
    seekEnd() {
      return store.submitControl({ commandType: "seek_end" });
    },
    backToTail() {
      return store.submitControl({ commandType: "back_to_tail" });
    },
    setSpeed(speed: number) {
      return store.submitControl({
        commandType: "set_speed",
        speed,
      });
    },
    finish() {
      return store.submitControl({ commandType: "finish" });
    },
    async loadMoreTimeline(limit?: number) {
      await store.loadMoreTimeline({ limit });
    },
  };
}

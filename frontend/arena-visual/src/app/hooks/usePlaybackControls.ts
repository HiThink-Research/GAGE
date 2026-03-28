import type { ArenaSessionStore } from "../store/arenaSessionStore";

export function usePlaybackControls(store: ArenaSessionStore) {
  return {
    async pause() {
      await store.submitControl({ commandType: "pause" });
    },
    async playLive() {
      await store.submitControl({ commandType: "follow_tail" });
    },
    async playReplay() {
      await store.submitControl({ commandType: "replay" });
    },
    async selectEvent(seq: number) {
      await store.submitControl({
        commandType: "seek_seq",
        targetSeq: seq,
      });
    },
    async stepBackward() {
      await store.submitControl({
        commandType: "step",
        stepDelta: -1,
      });
    },
    async stepForward() {
      await store.submitControl({
        commandType: "step",
        stepDelta: 1,
      });
    },
    async followTail() {
      await store.submitControl({ commandType: "follow_tail" });
    },
    async seekEnd() {
      await store.submitControl({ commandType: "seek_end" });
    },
    async backToTail() {
      await store.submitControl({ commandType: "back_to_tail" });
    },
    async setSpeed(speed: number) {
      await store.submitControl({
        commandType: "set_speed",
        speed,
      });
    },
    async loadMoreTimeline(limit?: number) {
      await store.loadMoreTimeline({ limit });
    },
  };
}

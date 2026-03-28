import type { ArenaSessionStore } from "../store/arenaSessionStore";

export function usePlaybackControls(store: ArenaSessionStore) {
  return {
    pause() {
      store.setPlaybackMode("paused");
    },
    playLive() {
      store.setPlaybackMode("live_tail");
    },
    playReplay() {
      store.setPlaybackMode("replay_playing");
    },
    async selectEvent(seq: number) {
      store.setCurrentSceneSeq(seq);
    },
    async loadMoreTimeline(limit?: number) {
      await store.loadMoreTimeline({ limit });
    },
  };
}

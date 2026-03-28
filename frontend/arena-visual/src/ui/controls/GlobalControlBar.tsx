import type { PlaybackMode, SchedulingState } from "../../gateway/types";

interface GlobalControlBarProps {
  playbackMode: PlaybackMode;
  scheduling?: SchedulingState;
  onPause: () => void;
  onPlayLive: () => void;
  onReplay: () => void;
}

export function GlobalControlBar({
  playbackMode,
  scheduling,
  onPause,
  onPlayLive,
  onReplay,
}: GlobalControlBarProps) {
  return (
    <section className="control-bar" aria-label="Playback controls">
      <div className="control-bar__buttons">
        <button
          type="button"
          className={playbackMode === "live_tail" ? "control-chip is-active" : "control-chip"}
          onClick={onPlayLive}
        >
          Live tail
        </button>
        <button
          type="button"
          className={playbackMode === "paused" ? "control-chip is-active" : "control-chip"}
          onClick={onPause}
        >
          Pause
        </button>
        <button
          type="button"
          className={playbackMode === "replay_playing" ? "control-chip is-active" : "control-chip"}
          onClick={onReplay}
        >
          Replay
        </button>
      </div>
      <div className="control-bar__status">
        <span>Scheduler</span>
        <strong>{scheduling ? `${scheduling.family} · ${scheduling.phase}` : "idle"}</strong>
      </div>
    </section>
  );
}

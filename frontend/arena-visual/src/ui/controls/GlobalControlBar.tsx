import type { PlaybackMode, SchedulingState } from "../../gateway/types";

interface GlobalControlBarProps {
  playbackMode: PlaybackMode;
  playbackSpeed: number;
  disabled?: boolean;
  scheduling?: SchedulingState;
  onPause: () => void;
  onPlayLive: () => void;
  onReplay: () => void;
  onSetSpeed: (speed: number) => void;
  onStep: (delta: -1 | 1) => void;
  onSeekEnd: () => void;
  onBackToTail: () => void;
}

export function GlobalControlBar({
  playbackMode,
  playbackSpeed,
  disabled = false,
  scheduling,
  onPause,
  onPlayLive,
  onReplay,
  onSetSpeed,
  onStep,
  onSeekEnd,
  onBackToTail,
}: GlobalControlBarProps) {
  return (
    <section className="control-bar" aria-label="Playback controls">
      <div className="control-bar__buttons">
        <button
          type="button"
          className={playbackMode === "live_tail" ? "control-chip is-active" : "control-chip"}
          disabled={disabled}
          onClick={onPlayLive}
        >
          Live tail
        </button>
        <button
          type="button"
          className={playbackMode === "paused" ? "control-chip is-active" : "control-chip"}
          disabled={disabled}
          onClick={onPause}
        >
          Pause
        </button>
        <button
          type="button"
          className={playbackMode === "replay_playing" ? "control-chip is-active" : "control-chip"}
          disabled={disabled}
          onClick={onReplay}
        >
          Replay
        </button>
        {([0.5, 1, 2] as const).map((speed) => (
          <button
            key={speed}
            type="button"
            className={playbackSpeed === speed ? "control-chip is-active" : "control-chip"}
            disabled={disabled}
            onClick={() => onSetSpeed(speed)}
          >
            {speed}x
          </button>
        ))}
        <button type="button" className="control-chip" disabled={disabled} onClick={() => onStep(-1)}>
          Step -1
        </button>
        <button type="button" className="control-chip" disabled={disabled} onClick={() => onStep(1)}>
          Step +1
        </button>
        <button type="button" className="control-chip" disabled={disabled} onClick={onSeekEnd}>
          End
        </button>
        <button type="button" className="control-chip" disabled={disabled} onClick={onBackToTail}>
          Back to tail
        </button>
      </div>
      <div className="control-bar__status">
        <span>Scheduler</span>
        <strong>{scheduling ? `${scheduling.family} · ${scheduling.phase}` : "idle"}</strong>
      </div>
    </section>
  );
}

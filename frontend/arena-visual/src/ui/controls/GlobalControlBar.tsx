import type { PlaybackMode, SchedulingState } from "../../gateway/types";

interface GlobalControlBarProps {
  playbackMode: PlaybackMode;
  playbackSpeed: number;
  disabled?: boolean;
  scheduling?: SchedulingState;
  postLiveStatusLabel?: string;
  controlAvailability?: {
    playLiveDisabled?: boolean;
    pauseDisabled?: boolean;
    replayDisabled?: boolean;
    speedDisabled?: boolean;
    stepBackwardDisabled?: boolean;
    stepForwardDisabled?: boolean;
    seekEndDisabled?: boolean;
    backToTailDisabled?: boolean;
    finishDisabled?: boolean;
  };
  finishLabel?: string;
  onPause: () => void;
  onPlayLive: () => void;
  onReplay: () => void;
  onSetSpeed: (speed: number) => void;
  onStep: (delta: -1 | 1) => void;
  onSeekEnd: () => void;
  onBackToTail: () => void;
  onFinish?: () => void;
}

export function GlobalControlBar({
  playbackMode,
  playbackSpeed,
  disabled = false,
  scheduling,
  postLiveStatusLabel,
  controlAvailability,
  finishLabel = "Finish",
  onPause,
  onPlayLive,
  onReplay,
  onSetSpeed,
  onStep,
  onSeekEnd,
  onBackToTail,
  onFinish,
}: GlobalControlBarProps) {
  return (
    <section className="control-bar" aria-label="Playback controls">
      <div className="control-bar__buttons">
        <button
          type="button"
          className={playbackMode === "live_tail" ? "control-chip is-active" : "control-chip"}
          disabled={disabled || controlAvailability?.playLiveDisabled}
          onClick={onPlayLive}
        >
          Live tail
        </button>
        <button
          type="button"
          className={playbackMode === "paused" ? "control-chip is-active" : "control-chip"}
          disabled={disabled || controlAvailability?.pauseDisabled}
          onClick={onPause}
        >
          Pause
        </button>
        <button
          type="button"
          className={playbackMode === "replay_playing" ? "control-chip is-active" : "control-chip"}
          disabled={disabled || controlAvailability?.replayDisabled}
          onClick={onReplay}
        >
          Replay
        </button>
        {([0.5, 1, 2] as const).map((speed) => (
          <button
            key={speed}
            type="button"
            className={playbackSpeed === speed ? "control-chip is-active" : "control-chip"}
            disabled={disabled || controlAvailability?.speedDisabled}
            onClick={() => onSetSpeed(speed)}
          >
            {speed}x
          </button>
        ))}
        <button
          type="button"
          className="control-chip"
          disabled={disabled || controlAvailability?.stepBackwardDisabled}
          onClick={() => onStep(-1)}
        >
          Step -1
        </button>
        <button
          type="button"
          className="control-chip"
          disabled={disabled || controlAvailability?.stepForwardDisabled}
          onClick={() => onStep(1)}
        >
          Step +1
        </button>
        <button
          type="button"
          className="control-chip"
          disabled={disabled || controlAvailability?.seekEndDisabled}
          onClick={onSeekEnd}
        >
          End
        </button>
        <button
          type="button"
          className="control-chip"
          disabled={disabled || controlAvailability?.backToTailDisabled}
          onClick={onBackToTail}
        >
          Back to tail
        </button>
        {onFinish ? (
          <button
            type="button"
            className="control-chip"
            disabled={disabled || controlAvailability?.finishDisabled}
            onClick={onFinish}
          >
            {finishLabel}
          </button>
        ) : null}
      </div>
      <div className="control-bar__status">
        <span>Scheduler</span>
        <strong>{scheduling ? `${scheduling.family} · ${scheduling.phase}` : "idle"}</strong>
        {postLiveStatusLabel ? <span>{postLiveStatusLabel}</span> : null}
      </div>
    </section>
  );
}

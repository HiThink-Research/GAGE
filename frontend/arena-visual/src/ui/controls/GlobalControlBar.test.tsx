import { fireEvent, render, screen } from "@testing-library/react";

import { GlobalControlBar } from "./GlobalControlBar";

describe("GlobalControlBar", () => {
  it("exposes playback speed, step, end, and back-to-tail controls", () => {
    const onPause = vi.fn();
    const onPlayLive = vi.fn();
    const onReplay = vi.fn();
    const onSetSpeed = vi.fn();
    const onStep = vi.fn();
    const onSeekEnd = vi.fn();
    const onBackToTail = vi.fn();

    render(
      <GlobalControlBar
        playbackMode="paused"
        playbackSpeed={1}
        scheduling={{
          family: "turn",
          phase: "waiting_for_intent",
          acceptsHumanIntent: true,
        }}
        onPause={onPause}
        onPlayLive={onPlayLive}
        onReplay={onReplay}
        onSetSpeed={onSetSpeed}
        onStep={onStep}
        onSeekEnd={onSeekEnd}
        onBackToTail={onBackToTail}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /2x/i }));
    fireEvent.click(screen.getByRole("button", { name: /step -1/i }));
    fireEvent.click(screen.getByRole("button", { name: /step \+1/i }));
    fireEvent.click(screen.getByRole("button", { name: /end/i }));
    fireEvent.click(screen.getByRole("button", { name: /back to tail/i }));

    expect(onSetSpeed).toHaveBeenCalledWith(2);
    expect(onStep).toHaveBeenNthCalledWith(1, -1);
    expect(onStep).toHaveBeenNthCalledWith(2, 1);
    expect(onSeekEnd).toHaveBeenCalledTimes(1);
    expect(onBackToTail).toHaveBeenCalledTimes(1);
  });

  it("disables playback controls when the host is not ready", () => {
    render(
      <GlobalControlBar
        playbackMode="live_tail"
        playbackSpeed={1}
        disabled
        onPause={vi.fn()}
        onPlayLive={vi.fn()}
        onReplay={vi.fn()}
        onSetSpeed={vi.fn()}
        onStep={vi.fn()}
        onSeekEnd={vi.fn()}
        onBackToTail={vi.fn()}
      />,
    );

    expect(screen.getByRole("button", { name: /live tail/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /pause/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /replay/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /2x/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /step -1/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /step \+1/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /end/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /back to tail/i })).toBeDisabled();
  });

  it("shows the post-live auto-finish hint and manual Finish control when requested", () => {
    const onFinish = vi.fn();

    render(
      <GlobalControlBar
        playbackMode="replay_playing"
        playbackSpeed={1}
        scheduling={{
          family: "turn",
          phase: "completed",
          acceptsHumanIntent: false,
        }}
        postLiveStatusLabel="Auto finish in 15s"
        onPause={vi.fn()}
        onPlayLive={vi.fn()}
        onReplay={vi.fn()}
        onSetSpeed={vi.fn()}
        onStep={vi.fn()}
        onSeekEnd={vi.fn()}
        onBackToTail={vi.fn()}
        onFinish={onFinish}
      />,
    );

    expect(screen.getByText("Auto finish in 15s")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /^finish$/i }));
    expect(onFinish).toHaveBeenCalledTimes(1);
  });

  it("disables playback controls according to the current playback scene", () => {
    render(
      <GlobalControlBar
        playbackMode="live_tail"
        playbackSpeed={1}
        scheduling={{
          family: "turn",
          phase: "advancing",
          acceptsHumanIntent: false,
        }}
        controlAvailability={{
          playLiveDisabled: true,
          pauseDisabled: false,
          replayDisabled: false,
          speedDisabled: true,
          stepBackwardDisabled: true,
          stepForwardDisabled: true,
          seekEndDisabled: true,
          backToTailDisabled: true,
        }}
        onPause={vi.fn()}
        onPlayLive={vi.fn()}
        onReplay={vi.fn()}
        onSetSpeed={vi.fn()}
        onStep={vi.fn()}
        onSeekEnd={vi.fn()}
        onBackToTail={vi.fn()}
      />,
    );

    expect(screen.getByRole("button", { name: /live tail/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /pause/i })).toBeEnabled();
    expect(screen.getByRole("button", { name: /replay/i })).toBeEnabled();
    expect(screen.getByRole("button", { name: /2x/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /step -1/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /step \+1/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /end/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /back to tail/i })).toBeDisabled();
  });
});

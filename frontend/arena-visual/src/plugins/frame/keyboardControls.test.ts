import { describe, expect, it } from "vitest";

import { buildPettingZooKeyboardControls } from "./keyboardControls";

describe("buildPettingZooKeyboardControls", () => {
  it("configures low-latency hold ticks and heartbeat dispatch for Atari controls", () => {
    const controls = buildPettingZooKeyboardControls();

    expect(controls.holdTickMs).toBe(16);
    expect(controls.holdTicksMin).toBe(1);
    expect(controls.holdTicksMax).toBe(4);
    expect(controls.initialHoldTicks).toBe(1);
    expect(controls.heartbeatMs).toBe(33);
    expect(controls.heartbeatHoldTicks).toBe(1);
  });
});

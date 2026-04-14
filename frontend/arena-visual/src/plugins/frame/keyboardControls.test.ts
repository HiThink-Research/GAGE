import { describe, expect, it } from "vitest";

import { buildPettingZooKeyboardControls, buildRetroMarioKeyboardControls } from "./keyboardControls";

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

describe("buildRetroMarioKeyboardControls", () => {
  it("keeps short hold coverage around websocket keyboard heartbeats", () => {
    const controls = buildRetroMarioKeyboardControls();

    expect(controls.holdTickMs).toBe(16);
    expect(controls.holdTicksMin).toBe(1);
    expect(controls.holdTicksMax).toBe(30);
    expect(controls.initialHoldTicks).toBe(3);
    expect(controls.heartbeatMs).toBe(80);
    expect(controls.heartbeatHoldTicks).toBe(5);
  });
});

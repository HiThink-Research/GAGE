import { describe, expect, it } from "vitest";

import {
  fitViewportWithinBounds,
  resolveImmersiveViewportBounds,
} from "./FrameSurface";

describe("fitViewportWithinBounds", () => {
  it("scales a wide viewport by width and height together when height is the limiting edge", () => {
    const fitted = fitViewportWithinBounds(
      { width: 256, height: 224 },
      { width: 1200, height: 500 },
    );

    expect(fitted).not.toBeNull();
    expect(fitted?.width).toBeCloseTo(571.4286, 3);
    expect(fitted?.height).toBeCloseTo(500, 3);
  });

  it("scales a widescreen viewport proportionally when width is the limiting edge", () => {
    const fitted = fitViewportWithinBounds(
      { width: 320, height: 180 },
      { width: 400, height: 600 },
    );

    expect(fitted).not.toBeNull();
    expect(fitted?.width).toBeCloseTo(400, 3);
    expect(fitted?.height).toBeCloseTo(225, 3);
  });

  it("returns null when the available bounds are not measurable", () => {
    expect(
      fitViewportWithinBounds({ width: 320, height: 180 }, { width: 0, height: 600 }),
    ).toBeNull();
  });
});

describe("resolveImmersiveViewportBounds", () => {
  it("prefers the largest measurable stage ancestor to avoid shrink-wrapped scaling", () => {
    const fitted = resolveImmersiveViewportBounds(
      { width: 256, height: 224 },
      [
        { width: 320, height: 280, top: 120 },
        { width: 320, height: 280, top: 120 },
        { width: 1080, height: 720, top: 96 },
      ],
      { windowHeight: 920, fullscreenActive: false },
    );

    expect(fitted).not.toBeNull();
    expect(fitted?.width).toBeCloseTo(822.8571, 3);
    expect(fitted?.height).toBeCloseTo(720, 3);
  });
});

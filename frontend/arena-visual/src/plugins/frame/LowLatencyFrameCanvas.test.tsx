import { render, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { LowLatencyFrameCanvas } from "./LowLatencyFrameCanvas";

function createMultipartResponse(): Response {
  const encoder = new TextEncoder();
  return new Response(
    new ReadableStream({
      start(controller) {
        controller.enqueue(
          encoder.encode(
            "--frame\r\nContent-Type: image/png\r\nContent-Length: 11\r\n\r\n",
          ),
        );
        controller.enqueue(encoder.encode("frame-bytes"));
        controller.enqueue(encoder.encode("\r\n--frame--\r\n"));
        controller.close();
      },
    }),
    {
      status: 200,
      headers: {
        "Content-Type": "multipart/x-mixed-replace; boundary=frame",
      },
    },
  );
}

describe("LowLatencyFrameCanvas", () => {
  it("keeps the low-latency stream connected when the parent rerenders with a new callback identity", async () => {
    const fetchMock = vi.fn().mockImplementation(() => Promise.resolve(createMultipartResponse()));
    const createImageBitmapMock = vi.fn().mockResolvedValue({
      width: 4,
      height: 4,
      close: vi.fn(),
    });
    const getContextMock = vi
      .spyOn(HTMLCanvasElement.prototype, "getContext")
      .mockReturnValue({ drawImage: vi.fn(), clearRect: vi.fn() } as unknown as CanvasRenderingContext2D);
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal("createImageBitmap", createImageBitmapMock);

    try {
      const { rerender } = render(
        <LowLatencyFrameCanvas
          altText="Mario frame"
          className="frame-surface__canvas"
          onFrameSizeChange={() => {}}
          streamUrl="http://arena.local/stream"
        />,
      );

      await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
      await waitFor(() => expect(createImageBitmapMock).toHaveBeenCalledTimes(1));

      rerender(
        <LowLatencyFrameCanvas
          altText="Mario frame"
          className="frame-surface__canvas"
          onFrameSizeChange={() => {}}
          streamUrl="http://arena.local/stream"
        />,
      );

      await new Promise((resolve) => {
        window.setTimeout(resolve, 0);
      });

      expect(fetchMock).toHaveBeenCalledTimes(1);
    } finally {
      getContextMock.mockRestore();
      vi.unstubAllGlobals();
    }
  });
});

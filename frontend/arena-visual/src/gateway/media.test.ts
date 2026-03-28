import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { createArenaMediaResolver } from "./media";
import type { ArenaGatewayClient } from "./client";
import type { MediaSourceRef } from "./types";

describe("createArenaMediaResolver", () => {
  const originalCreateObjectURL = URL.createObjectURL;
  const originalRevokeObjectURL = URL.revokeObjectURL;

  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
    URL.createObjectURL = vi.fn(() => "blob:arena/frame-1");
    URL.revokeObjectURL = vi.fn();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    URL.createObjectURL = originalCreateObjectURL;
    URL.revokeObjectURL = originalRevokeObjectURL;
  });

  it("resolves blob-backed media once and revokes the object URL after the last unsubscribe", async () => {
    const ref: MediaSourceRef = {
      mediaId: "frame-1",
      transport: "binary_stream",
      mimeType: "image/png",
    };
    const client: Pick<ArenaGatewayClient, "loadMedia"> = {
      loadMedia: vi.fn().mockResolvedValue(ref),
    };
    vi.mocked(fetch).mockResolvedValueOnce(
      new Response(new Blob(["frame"], { type: "image/png" }), {
        status: 200,
      }),
    );

    const resolver = createArenaMediaResolver(client);
    const listenerA = vi.fn();
    const listenerB = vi.fn();

    const unsubscribeA = resolver.subscribe(
      { sessionId: "sample-1", mediaId: "frame-1" },
      listenerA,
    );
    const unsubscribeB = resolver.subscribe(
      { sessionId: "sample-1", mediaId: "frame-1" },
      listenerB,
    );

    await vi.waitFor(() => {
      expect(client.loadMedia).toHaveBeenCalledTimes(1);
      expect(fetch).toHaveBeenCalledTimes(1);
      expect(listenerA).toHaveBeenLastCalledWith(
        expect.objectContaining({
          status: "ready",
          src: "blob:arena/frame-1",
        }),
      );
      expect(listenerB).toHaveBeenLastCalledWith(
        expect.objectContaining({
          status: "ready",
          src: "blob:arena/frame-1",
        }),
      );
    });

    unsubscribeA();
    expect(URL.revokeObjectURL).not.toHaveBeenCalled();

    unsubscribeB();
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:arena/frame-1");
  });

  it("normalizes slash-prefixed media refs through the gateway base url", async () => {
    const client: Pick<ArenaGatewayClient, "loadMedia" | "buildMediaUrl"> = {
      loadMedia: vi.fn().mockResolvedValue({
        mediaId: "frame-2",
        transport: "artifact_ref",
        mimeType: "image/png",
        url: "/media/frame-2",
      }),
      buildMediaUrl: vi
        .fn()
        .mockReturnValue("http://arena.local/arena_visual/sessions/sample-1/media/frame-2"),
    };
    const resolver = createArenaMediaResolver(client);
    const listener = vi.fn();

    const unsubscribe = resolver.subscribe(
      { sessionId: "sample-1", mediaId: "frame-2" },
      listener,
    );
    await vi.waitFor(() => {
      expect(listener).toHaveBeenLastCalledWith(
        expect.objectContaining({
          status: "ready",
          src: "http://arena.local/arena_visual/sessions/sample-1/media/frame-2",
        }),
      );
    });

    unsubscribe();
    const listenerReloaded = vi.fn();
    resolver.subscribe({ sessionId: "sample-1", mediaId: "frame-2" }, listenerReloaded);
    await vi.waitFor(() => {
      expect(client.loadMedia).toHaveBeenCalledTimes(2);
      expect(listenerReloaded).toHaveBeenLastCalledWith(
        expect.objectContaining({
          status: "ready",
          src: "http://arena.local/arena_visual/sessions/sample-1/media/frame-2",
        }),
      );
    });

    expect(URL.createObjectURL).not.toHaveBeenCalled();
    expect(URL.revokeObjectURL).not.toHaveBeenCalled();
  });

  it("normalizes relative artifact refs through the gateway media endpoint", async () => {
    const client: Pick<ArenaGatewayClient, "loadMedia" | "buildMediaUrl"> = {
      loadMedia: vi.fn().mockResolvedValue({
        mediaId: "frame-3",
        transport: "artifact_ref",
        mimeType: "image/png",
        url: "frames/frame-3.png",
      }),
      buildMediaUrl: vi
        .fn()
        .mockReturnValue("http://arena.local/arena_visual/sessions/sample-1/media/frame-3"),
    };
    const resolver = createArenaMediaResolver(client);
    const listener = vi.fn();

    const unsubscribe = resolver.subscribe(
      { sessionId: "sample-1", mediaId: "frame-3" },
      listener,
    );
    await Promise.resolve();
    await Promise.resolve();

    expect(listener).toHaveBeenLastCalledWith(
      expect.objectContaining({
        status: "ready",
        src: "http://arena.local/arena_visual/sessions/sample-1/media/frame-3",
      }),
    );

    unsubscribe();
  });

  it("uses inline data urls directly without refetching media content", async () => {
    const client: Pick<ArenaGatewayClient, "loadMedia" | "buildMediaUrl"> = {
      loadMedia: vi.fn().mockResolvedValue({
        mediaId: "frame-inline",
        transport: "http_pull",
        mimeType: "image/png",
        url: "data:image/png;base64,ZmFrZQ==",
      }),
      buildMediaUrl: vi.fn(),
    };
    const resolver = createArenaMediaResolver(client);

    const resolved = await resolver.resolve({
      sessionId: "sample-1",
      mediaId: "frame-inline",
    });

    expect(resolved).toEqual({
      mediaId: "frame-inline",
      ref: {
        mediaId: "frame-inline",
        transport: "http_pull",
        mimeType: "image/png",
        url: "data:image/png;base64,ZmFrZQ==",
      },
      src: "data:image/png;base64,ZmFrZQ==",
      status: "ready",
    });
    expect(client.buildMediaUrl).not.toHaveBeenCalled();
    expect(fetch).not.toHaveBeenCalled();
  });

  it("allows retry after a transient media failure", async () => {
    const client: Pick<ArenaGatewayClient, "loadMedia"> = {
      loadMedia: vi.fn().mockResolvedValue({
        mediaId: "frame-retry",
        transport: "binary_stream",
        mimeType: "image/png",
      }),
    };
    vi.mocked(fetch)
      .mockResolvedValueOnce(new Response("nope", { status: 503 }))
      .mockResolvedValueOnce(
        new Response(new Blob(["frame"], { type: "image/png" }), {
          status: 200,
        }),
      );

    const resolver = createArenaMediaResolver(client);

    const failed = await resolver.resolve({
      sessionId: "sample-1",
      mediaId: "frame-retry",
    });
    const recovered = await resolver.resolve({
      sessionId: "sample-1",
      mediaId: "frame-retry",
    });

    expect(failed.status).toBe("error");
    expect(recovered.status).toBe("ready");
    expect(fetch).toHaveBeenCalledTimes(2);
  });
});

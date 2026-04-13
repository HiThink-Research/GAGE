import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { createRealtimeInputSocket } from "./realtimeInputSocket";

class FakeWebSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;
  static instances: FakeWebSocket[] = [];

  readonly sent: string[] = [];
  readonly url: string;
  readyState = FakeWebSocket.CONNECTING;
  onopen: ((event: Event) => void) | null = null;
  onclose: ((event: Event) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent<string>) => void) | null = null;

  constructor(url: string) {
    this.url = url;
    FakeWebSocket.instances.push(this);
  }

  send(data: string): void {
    this.sent.push(data);
  }

  close(): void {
    this.readyState = FakeWebSocket.CLOSED;
    this.onclose?.(new Event("close"));
  }

  open(): void {
    this.readyState = FakeWebSocket.OPEN;
    this.onopen?.(new Event("open"));
  }

  closeFromServer(): void {
    this.readyState = FakeWebSocket.CLOSED;
    this.onclose?.(new Event("close"));
  }
}

describe("createRealtimeInputSocket", () => {
  beforeEach(() => {
    FakeWebSocket.instances = [];
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("keeps only the latest pending payload and flushes it after reconnect", async () => {
    const socket = createRealtimeInputSocket({
      url: "ws://arena.local/arena_visual/sessions/sample-1/actions/ws",
      reconnectDelayMs: 25,
      WebSocketCtor: FakeWebSocket as unknown as typeof WebSocket,
    });

    expect(FakeWebSocket.instances).toHaveLength(1);

    await socket.submit({ move: "right", metadata: { input_seq: 1 } });
    await socket.submit({ move: "right_jump", metadata: { input_seq: 2 } });
    expect(FakeWebSocket.instances[0]?.sent).toEqual([]);

    FakeWebSocket.instances[0]?.open();
    expect(FakeWebSocket.instances[0]?.sent).toEqual([
      JSON.stringify({ move: "right_jump", metadata: { input_seq: 2 } }),
    ]);

    FakeWebSocket.instances[0]?.closeFromServer();

    await socket.submit({ move: "noop", metadata: { input_seq: 3 } });
    expect(FakeWebSocket.instances).toHaveLength(2);

    await vi.advanceTimersByTimeAsync(25);
    expect(FakeWebSocket.instances).toHaveLength(2);

    FakeWebSocket.instances[1]?.open();
    expect(FakeWebSocket.instances[1]?.sent).toEqual([
      JSON.stringify({ move: "noop", metadata: { input_seq: 3 } }),
    ]);

    socket.close();
  });
});

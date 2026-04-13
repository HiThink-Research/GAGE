interface RealtimeInputSocketOptions {
  url: string;
  reconnectDelayMs?: number;
  WebSocketCtor?: typeof WebSocket;
}

export interface RealtimeInputSocket {
  submit(payload: Record<string, unknown>): Promise<void>;
  close(): void;
}

const DEFAULT_RECONNECT_DELAY_MS = 120;

interface WebSocketLike {
  readonly readyState: number;
  onopen: ((event: Event) => void) | null;
  onclose: ((event: Event) => void) | null;
  onerror: ((event: Event) => void) | null;
  onmessage: ((event: MessageEvent<string>) => void) | null;
  send(data: string): void;
  close(code?: number, reason?: string): void;
}

interface WebSocketConstructorLike {
  new (url: string): WebSocketLike;
  readonly CONNECTING: number;
  readonly OPEN: number;
}

export function createRealtimeInputSocket({
  url,
  reconnectDelayMs = DEFAULT_RECONNECT_DELAY_MS,
  WebSocketCtor = globalThis.WebSocket,
}: RealtimeInputSocketOptions): RealtimeInputSocket {
  let disposed = false;
  let pendingPayload: Record<string, unknown> | null = null;
  let socket: WebSocketLike | null = null;
  let reconnectTimer: number | null = null;

  const socketCtor = WebSocketCtor as unknown as WebSocketConstructorLike | undefined;

  const clearReconnectTimer = (): void => {
    if (reconnectTimer === null) {
      return;
    }
    window.clearTimeout(reconnectTimer);
    reconnectTimer = null;
  };

  const flushPendingPayload = (): void => {
    if (!socketCtor || socket === null || socket.readyState !== socketCtor.OPEN || pendingPayload === null) {
      return;
    }
    const payload = pendingPayload;
    pendingPayload = null;
    socket.send(JSON.stringify(payload));
  };

  const scheduleReconnect = (): void => {
    if (disposed || reconnectTimer !== null || !socketCtor) {
      return;
    }
    reconnectTimer = window.setTimeout(() => {
      reconnectTimer = null;
      connect();
    }, Math.max(0, reconnectDelayMs));
  };

  const connect = (): void => {
    if (disposed || !socketCtor) {
      return;
    }
    if (socket !== null) {
      if (socket.readyState === socketCtor.CONNECTING || socket.readyState === socketCtor.OPEN) {
        return;
      }
      socket = null;
    }
    clearReconnectTimer();

    try {
      const nextSocket = new socketCtor(url);
      socket = nextSocket;
      nextSocket.onopen = () => {
        if (socket !== nextSocket || disposed) {
          return;
        }
        flushPendingPayload();
      };
      nextSocket.onclose = () => {
        if (socket === nextSocket) {
          socket = null;
        }
        if (!disposed) {
          scheduleReconnect();
        }
      };
      nextSocket.onerror = () => {
        // Allow close/reconnect lifecycle to recover without surfacing noisy errors.
      };
      nextSocket.onmessage = () => {
        // The realtime input socket is write-heavy; receipts stay on the timeline path.
      };
    } catch {
      socket = null;
      scheduleReconnect();
    }
  };

  connect();

  return {
    async submit(payload: Record<string, unknown>): Promise<void> {
      pendingPayload = payload;
      if (!socketCtor) {
        return;
      }
      connect();
      flushPendingPayload();
    },

    close(): void {
      disposed = true;
      clearReconnectTimer();
      if (socket !== null) {
        const current = socket;
        socket = null;
        try {
          current.close(1000, "client_close");
        } catch {
          // Ignore close failures during teardown.
        }
      }
      pendingPayload = null;
    },
  };
}

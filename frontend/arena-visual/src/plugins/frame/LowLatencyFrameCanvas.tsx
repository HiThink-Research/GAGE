import { useEffect, useRef, useState } from "react";

interface LowLatencyFrameCanvasProps {
  altText: string;
  className: string;
  streamUrl: string;
  testId?: string;
}

const HEADER_DELIMITER = new Uint8Array([13, 10, 13, 10]);
const CRLF = new Uint8Array([13, 10]);
const textDecoder = new TextDecoder();
const textEncoder = new TextEncoder();

export function resolveLowLatencyStreamUrl(
  url: string | undefined,
  transport: string | undefined,
): string | null {
  if (transport !== "low_latency_channel" || !url) {
    return null;
  }
  try {
    const baseUrl =
      typeof window !== "undefined" && typeof window.location?.href === "string"
        ? window.location.href
        : "http://localhost/";
    return new URL(url, baseUrl).toString();
  } catch {
    return null;
  }
}

export function LowLatencyFrameCanvas({
  altText,
  className,
  streamUrl,
  testId = "frame-surface-canvas",
}: LowLatencyFrameCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [hasFrame, setHasFrame] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let drawing = false;
    let pendingFrame:
      | { bytes: Uint8Array<ArrayBufferLike>; mimeType: string | null }
      | null = null;
    const abortController = new AbortController();

    async function drawPendingFrames(): Promise<void> {
      if (drawing) {
        return;
      }
      drawing = true;
      try {
        while (!cancelled && pendingFrame !== null) {
          const frame = pendingFrame;
          pendingFrame = null;
          if (!canvasRef.current) {
            continue;
          }
          const drewFrame = await drawBytesToCanvas(
            canvasRef.current,
            frame.bytes,
            frame.mimeType,
          );
          if (drewFrame && !cancelled) {
            setHasFrame(true);
          }
        }
      } finally {
        drawing = false;
      }
    }

    function enqueueFrame(bytes: Uint8Array<ArrayBufferLike>, mimeType: string | null): void {
      pendingFrame = { bytes, mimeType };
      void drawPendingFrames();
    }

    async function loadStream(): Promise<void> {
      try {
        const response = await fetch(streamUrl, {
          cache: "no-store",
          signal: abortController.signal,
        });
        if (!response.ok) {
          throw new Error(`Low-latency frame fetch failed with status ${response.status}`);
        }
        const contentType = response.headers.get("Content-Type");
        if (response.body && isMultipartContentType(contentType)) {
          await readMultipartFrameStream(response.body, contentType, enqueueFrame, abortController.signal);
          return;
        }
        const blob = await response.blob();
        if (!cancelled) {
          enqueueFrame(new Uint8Array(await blob.arrayBuffer()), blob.type || contentType);
        }
      } catch (error) {
        if (cancelled || isAbortError(error)) {
          return;
        }
      }
    }

    void loadStream();
    return () => {
      cancelled = true;
      abortController.abort();
    };
  }, [streamUrl]);

  return (
    <>
      <canvas
        aria-label={altText}
        className={className}
        data-testid={testId}
        ref={canvasRef}
      />
      {!hasFrame ? <div className="frame-surface__fallback">Loading frame...</div> : null}
    </>
  );
}

async function readMultipartFrameStream(
  stream: ReadableStream<Uint8Array>,
  contentType: string | null,
  onFrame: (bytes: Uint8Array<ArrayBufferLike>, mimeType: string | null) => void,
  signal: AbortSignal,
): Promise<void> {
  const boundary = parseMultipartBoundary(contentType);
  if (!boundary) {
    return;
  }
  const reader = stream.getReader();
  const boundaryMarker = textEncoder.encode(`--${boundary}`);
  let buffer: Uint8Array<ArrayBufferLike> = new Uint8Array(0);

  try {
    while (!signal.aborted) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      if (value && value.length > 0) {
        buffer = concatBytes(buffer, cloneBytes(value));
      }
      while (true) {
        const parsed = tryParseMultipartFrame(buffer, boundaryMarker);
        if (parsed === null) {
          buffer = trimMultipartBuffer(buffer, boundaryMarker.length);
          break;
        }
        if (parsed.done) {
          return;
        }
        if (parsed.frame !== null) {
          onFrame(parsed.frame.bytes, parsed.frame.mimeType);
        }
        buffer = parsed.remaining;
      }
    }
  } finally {
    reader.releaseLock();
  }
}

function tryParseMultipartFrame(
  buffer: Uint8Array<ArrayBufferLike>,
  boundaryMarker: Uint8Array<ArrayBufferLike>,
): {
  done: boolean;
  frame: { bytes: Uint8Array<ArrayBufferLike>; mimeType: string | null } | null;
  remaining: Uint8Array<ArrayBufferLike>;
} | null {
  const boundaryIndex = indexOfBytes(buffer, boundaryMarker);
  if (boundaryIndex < 0) {
    return null;
  }
  let remaining = boundaryIndex > 0 ? buffer.slice(boundaryIndex) : buffer;
  if (remaining.length < boundaryMarker.length + 2) {
    return null;
  }

  let cursor = boundaryMarker.length;
  if (remaining[cursor] === 45 && remaining[cursor + 1] === 45) {
    return {
      done: true,
      frame: null,
      remaining: new Uint8Array(0),
    };
  }
  if (!startsWithBytes(remaining, CRLF, cursor)) {
    return null;
  }
  cursor += CRLF.length;

  const headerEnd = indexOfBytes(remaining, HEADER_DELIMITER, cursor);
  if (headerEnd < 0) {
    return null;
  }
  const headers = parsePartHeaders(textDecoder.decode(remaining.slice(cursor, headerEnd)));
  const mimeType = headers.get("content-type") ?? null;
  const bodyStart = headerEnd + HEADER_DELIMITER.length;
  const contentLength = parseContentLength(headers.get("content-length"));

  if (contentLength !== null) {
    const bodyEnd = bodyStart + contentLength;
    if (remaining.length < bodyEnd) {
      return null;
    }
    const trailingEnd =
      startsWithBytes(remaining, CRLF, bodyEnd) ? bodyEnd + CRLF.length : bodyEnd;
    return {
      done: false,
      frame: {
        bytes: remaining.slice(bodyStart, bodyEnd),
        mimeType,
      },
      remaining: remaining.slice(trailingEnd),
    };
  }

  const nextBoundaryIndex = indexOfBytes(remaining, boundaryMarker, bodyStart);
  if (nextBoundaryIndex < 0) {
    return null;
  }
  let bodyEnd = nextBoundaryIndex;
  if (bodyEnd >= 2 && remaining[bodyEnd - 2] === 13 && remaining[bodyEnd - 1] === 10) {
    bodyEnd -= 2;
  }
  return {
    done: false,
    frame: {
      bytes: remaining.slice(bodyStart, bodyEnd),
      mimeType,
    },
    remaining: remaining.slice(nextBoundaryIndex),
  };
}

function parseMultipartBoundary(contentType: string | null): string | null {
  if (!contentType) {
    return null;
  }
  const match = /boundary=([^\s;]+)/i.exec(contentType);
  if (!match) {
    return null;
  }
  return match[1]?.replace(/^"|"$/g, "") || null;
}

function parsePartHeaders(headerText: string): Map<string, string> {
  const headers = new Map<string, string>();
  for (const rawLine of headerText.split(/\r?\n/)) {
    const separatorIndex = rawLine.indexOf(":");
    if (separatorIndex <= 0) {
      continue;
    }
    const key = rawLine.slice(0, separatorIndex).trim().toLowerCase();
    const value = rawLine.slice(separatorIndex + 1).trim();
    if (key !== "" && value !== "") {
      headers.set(key, value);
    }
  }
  return headers;
}

function parseContentLength(value: string | undefined): number | null {
  if (!value) {
    return null;
  }
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : null;
}

function isMultipartContentType(contentType: string | null): boolean {
  return typeof contentType === "string" && contentType.toLowerCase().includes("multipart/");
}

function trimMultipartBuffer(
  buffer: Uint8Array<ArrayBufferLike>,
  boundaryLength: number,
): Uint8Array<ArrayBufferLike> {
  const keepLength = Math.max(boundaryLength + 8, 128);
  return buffer.length <= keepLength ? buffer : cloneBytes(buffer.slice(buffer.length - keepLength));
}

function concatBytes(
  left: Uint8Array<ArrayBufferLike>,
  right: Uint8Array<ArrayBufferLike>,
): Uint8Array<ArrayBufferLike> {
  if (left.length === 0) {
    return cloneBytes(right);
  }
  if (right.length === 0) {
    return cloneBytes(left);
  }
  const combined = new Uint8Array(left.length + right.length);
  combined.set(left);
  combined.set(right, left.length);
  return combined;
}

function cloneBytes(bytes: Uint8Array<ArrayBufferLike>): Uint8Array<ArrayBufferLike> {
  const cloned = new Uint8Array(new ArrayBuffer(bytes.length));
  cloned.set(bytes);
  return cloned;
}

function indexOfBytes(
  haystack: Uint8Array<ArrayBufferLike>,
  needle: Uint8Array<ArrayBufferLike>,
  start = 0,
): number {
  if (needle.length === 0) {
    return -1;
  }
  const lastStart = haystack.length - needle.length;
  for (let index = Math.max(0, start); index <= lastStart; index += 1) {
    let matched = true;
    for (let offset = 0; offset < needle.length; offset += 1) {
      if (haystack[index + offset] !== needle[offset]) {
        matched = false;
        break;
      }
    }
    if (matched) {
      return index;
    }
  }
  return -1;
}

function startsWithBytes(
  buffer: Uint8Array<ArrayBufferLike>,
  prefix: Uint8Array<ArrayBufferLike>,
  offset: number,
): boolean {
  if (offset + prefix.length > buffer.length) {
    return false;
  }
  for (let index = 0; index < prefix.length; index += 1) {
    if (buffer[offset + index] !== prefix[index]) {
      return false;
    }
  }
  return true;
}

async function drawBytesToCanvas(
  canvas: HTMLCanvasElement,
  bytes: Uint8Array<ArrayBufferLike>,
  mimeType: string | null,
): Promise<boolean> {
  const blob = new Blob([toArrayBuffer(bytes)], {
    type: mimeType || "application/octet-stream",
  });
  return drawBlobToCanvas(canvas, blob);
}

function toArrayBuffer(bytes: Uint8Array<ArrayBufferLike>): ArrayBuffer {
  const buffer = bytes.buffer;
  if (buffer instanceof ArrayBuffer) {
    return buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  }
  const cloned = new Uint8Array(new ArrayBuffer(bytes.byteLength));
  cloned.set(bytes);
  return cloned.buffer;
}

async function drawBlobToCanvas(
  canvas: HTMLCanvasElement,
  blob: Blob,
): Promise<boolean> {
  const context = canvas.getContext("2d");
  if (!context) {
    return false;
  }
  if (typeof createImageBitmap === "function") {
    const bitmap = await createImageBitmap(blob);
    try {
      canvas.width = Math.max(1, bitmap.width);
      canvas.height = Math.max(1, bitmap.height);
      context.clearRect(0, 0, canvas.width, canvas.height);
      context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
      return true;
    } finally {
      if (typeof bitmap.close === "function") {
        bitmap.close();
      }
    }
  }

  const image = await loadImageFromBlob(blob);
  canvas.width = Math.max(1, image.naturalWidth || image.width);
  canvas.height = Math.max(1, image.naturalHeight || image.height);
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.drawImage(image, 0, 0, canvas.width, canvas.height);
  return true;
}

function loadImageFromBlob(blob: Blob): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const objectUrl = URL.createObjectURL(blob);
    const image = new Image();
    image.onload = () => {
      URL.revokeObjectURL(objectUrl);
      resolve(image);
    };
    image.onerror = (error) => {
      URL.revokeObjectURL(objectUrl);
      reject(error);
    };
    image.src = objectUrl;
  });
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

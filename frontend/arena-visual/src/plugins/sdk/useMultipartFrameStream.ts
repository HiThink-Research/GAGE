import { useEffect, useState } from "react";

interface MultipartFrameState {
  src?: string;
  status: "idle" | "loading" | "ready" | "error";
  error?: string;
}

interface UseMultipartFrameStreamOptions {
  enabled: boolean;
  streamUrl?: string | null;
}

const HEADER_TERMINATOR = cloneBytes(new TextEncoder().encode("\r\n\r\n"));
const LINE_BREAK = cloneBytes(new TextEncoder().encode("\r\n"));
const HEADER_DECODER = new TextDecoder();

class MultipartFrameParser {
  private readonly boundary: Uint8Array<ArrayBufferLike>;
  private buffer: Uint8Array<ArrayBufferLike> = new Uint8Array(0);

  constructor(boundary: string) {
    this.boundary = cloneBytes(new TextEncoder().encode(`--${boundary}`));
  }

  push(
    chunk: Uint8Array<ArrayBufferLike>,
  ): Array<{ contentType: string | null; content: Uint8Array<ArrayBufferLike> }> {
    this.buffer = concatBytes(this.buffer, chunk);
    const frames: Array<{
      contentType: string | null;
      content: Uint8Array<ArrayBufferLike>;
    }> = [];

    while (true) {
      const boundaryIndex = indexOfSequence(this.buffer, this.boundary);
      if (boundaryIndex < 0) {
        this.buffer = trimBufferTail(this.buffer, this.boundary.length);
        return frames;
      }
      if (boundaryIndex > 0) {
        this.buffer = cloneBytes(this.buffer.slice(boundaryIndex));
      }

      let cursor = this.boundary.length;
      if (hasPrefix(this.buffer, cursor, new Uint8Array([45, 45]))) {
        return frames;
      }
      if (!hasPrefix(this.buffer, cursor, LINE_BREAK)) {
        return frames;
      }
      cursor += LINE_BREAK.length;

      const headersEnd = indexOfSequence(this.buffer, HEADER_TERMINATOR, cursor);
      if (headersEnd < 0) {
        return frames;
      }

      const headersText = HEADER_DECODER.decode(this.buffer.slice(cursor, headersEnd));
      const headers = parseHeaders(headersText);
      const contentLength = Number.parseInt(headers["content-length"] ?? "", 10);
      if (!Number.isFinite(contentLength) || contentLength < 0) {
        this.buffer = new Uint8Array(0);
        return frames;
      }

      const bodyStart = headersEnd + HEADER_TERMINATOR.length;
      const bodyEnd = bodyStart + contentLength;
      if (this.buffer.length < bodyEnd + LINE_BREAK.length) {
        return frames;
      }

      frames.push({
        contentType: headers["content-type"] ?? null,
        content: cloneBytes(this.buffer.slice(bodyStart, bodyEnd)),
      });
      this.buffer = cloneBytes(this.buffer.slice(bodyEnd + LINE_BREAK.length));
    }
  }
}

function cloneBytes(bytes: Uint8Array<ArrayBufferLike>): Uint8Array<ArrayBufferLike> {
  const cloned = new Uint8Array(new ArrayBuffer(bytes.length));
  cloned.set(bytes, 0);
  return cloned;
}

function concatBytes(
  left: Uint8Array<ArrayBufferLike>,
  right: Uint8Array<ArrayBufferLike>,
): Uint8Array<ArrayBufferLike> {
  const merged = new Uint8Array(new ArrayBuffer(left.length + right.length));
  merged.set(left, 0);
  merged.set(right, left.length);
  return merged;
}

function trimBufferTail(
  buffer: Uint8Array<ArrayBufferLike>,
  keepBytes: number,
): Uint8Array<ArrayBufferLike> {
  if (buffer.length <= keepBytes) {
    return buffer;
  }
  return cloneBytes(buffer.slice(buffer.length - keepBytes));
}

function hasPrefix(
  buffer: Uint8Array<ArrayBufferLike>,
  start: number,
  prefix: Uint8Array<ArrayBufferLike>,
): boolean {
  if (buffer.length < start + prefix.length) {
    return false;
  }
  for (let index = 0; index < prefix.length; index += 1) {
    if (buffer[start + index] !== prefix[index]) {
      return false;
    }
  }
  return true;
}

function indexOfSequence(
  source: Uint8Array<ArrayBufferLike>,
  target: Uint8Array<ArrayBufferLike>,
  fromIndex = 0,
): number {
  if (target.length === 0) {
    return fromIndex;
  }
  for (let index = fromIndex; index <= source.length - target.length; index += 1) {
    let matched = true;
    for (let offset = 0; offset < target.length; offset += 1) {
      if (source[index + offset] !== target[offset]) {
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

function toBlobPart(bytes: Uint8Array<ArrayBufferLike>): ArrayBuffer {
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
}

function parseHeaders(rawHeaders: string): Record<string, string> {
  const headers: Record<string, string> = {};
  for (const rawLine of rawHeaders.split("\r\n")) {
    const separatorIndex = rawLine.indexOf(":");
    if (separatorIndex <= 0) {
      continue;
    }
    const key = rawLine.slice(0, separatorIndex).trim().toLowerCase();
    const value = rawLine.slice(separatorIndex + 1).trim();
    if (!key || !value) {
      continue;
    }
    headers[key] = value;
  }
  return headers;
}

function parseMultipartBoundary(contentType: string | null): string | null {
  if (!contentType) {
    return null;
  }
  const match = /boundary=([^;]+)/i.exec(contentType);
  if (!match) {
    return null;
  }
  const boundary = match[1]?.trim().replace(/^"|"$/g, "");
  return boundary || null;
}

export function useMultipartFrameStream({
  enabled,
  streamUrl,
}: UseMultipartFrameStreamOptions): MultipartFrameState {
  const [state, setState] = useState<MultipartFrameState>({ status: "idle" });

  useEffect(() => {
    if (!enabled || !streamUrl) {
      setState({ status: "idle" });
      return undefined;
    }

    let disposed = false;
    let currentObjectUrl: string | undefined;
    const abortController = new AbortController();

    const load = async () => {
      setState({ status: "loading" });
      const response = await fetch(streamUrl, {
        cache: "no-store",
        signal: abortController.signal,
      });
      if (!response.ok) {
        throw new Error(`Multipart stream failed with status ${response.status}`);
      }
      const boundary = parseMultipartBoundary(response.headers.get("Content-Type"));
      if (!boundary) {
        throw new Error("Multipart stream boundary missing.");
      }
      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error("Multipart stream body is unavailable.");
      }

      const parser = new MultipartFrameParser(boundary);
      while (!disposed) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        if (!value) {
          continue;
        }
        for (const frame of parser.push(value)) {
          if (disposed) {
            return;
          }
          if (currentObjectUrl) {
            URL.revokeObjectURL(currentObjectUrl);
          }
          currentObjectUrl = URL.createObjectURL(
            new Blob([toBlobPart(frame.content)], {
              type: frame.contentType ?? "application/octet-stream",
            }),
          );
          setState({
            status: "ready",
            src: currentObjectUrl,
          });
        }
      }
    };

    void load().catch((error: unknown) => {
      if (disposed) {
        return;
      }
      if (error instanceof Error && error.name === "AbortError") {
        return;
      }
      setState({
        status: "error",
        error: error instanceof Error ? error.message : "Multipart stream failed.",
      });
    });

    return () => {
      disposed = true;
      abortController.abort();
      if (currentObjectUrl) {
        URL.revokeObjectURL(currentObjectUrl);
      }
    };
  }, [enabled, streamUrl]);

  return state;
}

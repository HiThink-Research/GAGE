import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type CSSProperties,
} from "react";
import type { VisualScene, VisualSession } from "../../gateway/types";
import {
  isRecord,
  readNumber,
  readOptionalNumber,
  readString,
} from "../../lib/sceneReaders";
import { useMediaSource } from "../sdk/useMediaSource";
import type { ArenaMediaSubscriber } from "../sdk/contracts";
import type { ActionIntent } from "../sdk/input";
import type {
  FrameActionDescriptor,
  FrameKeyboardControls,
  FrameOptimisticOffset,
} from "./contracts";
import { normalizeKeyboardKey } from "./keyboardControls";
import { LowLatencyFrameCanvas, resolveLowLatencyStreamUrl } from "./LowLatencyFrameCanvas";

interface FrameViewport {
  width: number;
  height: number;
}

interface FrameBadge {
  kind: string;
  label: string;
  value: string;
}

interface FrameBounds {
  width: number;
  height: number;
}

interface FrameMeasurementCandidate extends FrameBounds {
  top: number;
}

interface FrameSceneData {
  frame: {
    title: string;
    subtitle: string | null;
    altText: string;
    streamId: string | null;
    fit: string;
    viewport: FrameViewport | null;
  };
  status: {
    activePlayerId: string | null;
    observerPlayerId: string | null;
    tick: number;
    step: number;
    moveCount: number;
    lastMove: string | null;
    reward: number | null;
  };
  viewText: string | null;
  overlays: FrameBadge[];
}

interface FrameSurfaceProps {
  gameLabel: string;
  session: VisualSession;
  scene?: VisualScene;
  submitInput: (event: {
    playerId: string;
    actionPayload: ActionIntent["action"];
  }) => Promise<void>;
  mediaSubscribe: ArenaMediaSubscriber;
  keyboardControls?: FrameKeyboardControls;
  presentation?: "default" | "immersive";
  showStatusLine?: boolean;
  showViewText?: boolean;
}

interface ScenePrimaryMediaRef {
  transport?: string;
  url?: string | null;
}

interface KeyboardDispatchOptions {
  force?: boolean;
  phase?: "press" | "heartbeat" | "release";
  releasedHoldTicks?: number;
}

function readViewport(value: unknown): FrameViewport | null {
  if (!isRecord(value)) {
    return null;
  }
  const width = readNumber(value.width);
  const height = readNumber(value.height);
  if (width <= 0 || height <= 0) {
    return null;
  }
  return { width, height };
}

function readOverlays(scene?: VisualScene): FrameBadge[] {
  if (!scene?.overlays) {
    return [];
  }
  return scene.overlays
    .filter(isRecord)
    .map((item) => ({
      kind: readString(item.kind) ?? "badge",
      label: readString(item.label) ?? "Info",
      value: readString(item.value) ?? "",
    }))
    .filter((item) => item.value !== "");
}

export function readFrameScene(scene?: VisualScene): FrameSceneData | null {
  if (!scene || scene.kind !== "frame" || !isRecord(scene.body)) {
    return null;
  }

  const frame = scene.body.frame;
  const status = scene.body.status;
  const view = scene.body.view;
  if (!isRecord(frame) || !isRecord(status)) {
    return null;
  }

  return {
    frame: {
      title: readString(frame.title) ?? "Frame view",
      subtitle: readString(frame.subtitle),
      altText: readString(frame.altText) ?? "Game frame",
      streamId: readString(frame.streamId),
      fit: readString(frame.fit) ?? "contain",
      viewport: readViewport(frame.viewport),
    },
    status: {
      activePlayerId: readString(status.activePlayerId),
      observerPlayerId: readString(status.observerPlayerId),
      tick: readNumber(status.tick),
      step: readNumber(status.step),
      moveCount: readNumber(status.moveCount),
      lastMove: readString(status.lastMove),
      reward: readOptionalNumber(status.reward),
    },
    viewText: isRecord(view) ? readString(view.text) : null,
    overlays: readOverlays(scene),
  };
}

export function readFrameActions(scene?: VisualScene): FrameActionDescriptor[] {
  if (!scene) {
    return [];
  }
  return scene.legalActions
    .filter(isRecord)
    .map((action) => {
      const actionId =
        readString(action.id) ?? readString(action.text) ?? readString(action.label);
      if (!actionId) {
        return null;
      }
      const label = readString(action.label) ?? readString(action.text) ?? actionId;
      const payload: Record<string, unknown> = {
        id: actionId,
        move: actionId,
      };
      for (const [key, value] of Object.entries(action)) {
        if (key === "id" || key === "label" || key === "text") {
          continue;
        }
        payload[key] = value;
      }
      return {
        id: actionId,
        label,
        payload,
      };
    })
    .filter((item): item is FrameActionDescriptor => item !== null);
}

export function resolveFrameActorId(
  session: VisualSession,
  scene: VisualScene | undefined,
  frameScene: FrameSceneData,
): string | null {
  if (session.observer.observerKind === "player" && session.observer.observerId.trim() !== "") {
    return session.observer.observerId;
  }
  return (
    frameScene.status.activePlayerId ??
    frameScene.status.observerPlayerId ??
    scene?.activePlayerId ??
    session.scheduling.activeActorId ??
    null
  );
}

export function formatFrameActorLabel(
  session: VisualSession,
  resolvedActorId: string | null,
): string {
  if (session.observer.observerKind === "player" && session.observer.observerId.trim() !== "") {
    return `Observer: ${session.observer.observerId}`;
  }
  if (resolvedActorId) {
    return `Active player: ${resolvedActorId}`;
  }
  return "Active player: waiting";
}

function formatStatusLine(frameScene: FrameSceneData): string {
  const parts = [`Tick ${frameScene.status.tick}`];
  if (frameScene.status.moveCount > 0) {
    parts.push(`Move ${frameScene.status.moveCount}`);
  }
  if (frameScene.status.reward !== null) {
    parts.push(`Reward ${frameScene.status.reward}`);
  }
  if (frameScene.status.lastMove) {
    parts.push(`Last ${frameScene.status.lastMove}`);
  }
  return parts.join(" · ");
}

function resolveImageClassName(fit: string): string {
  return fit === "cover"
    ? "frame-surface__image frame-surface__image--cover"
    : "frame-surface__image frame-surface__image--contain";
}

function resolveCanvasClassName(fit: string): string {
  return fit === "cover"
    ? "frame-surface__canvas frame-surface__canvas--cover"
    : "frame-surface__canvas frame-surface__canvas--contain";
}

function isAbsoluteMediaUrl(url: string): boolean {
  return /^(https?:\/\/|data:|blob:)/i.test(url);
}

function resolveDirectFrameImageSrc(ref: ScenePrimaryMediaRef | null | undefined): string | null {
  if (!ref) {
    return null;
  }
  if (ref.transport === "low_latency_channel" || ref.transport === "binary_stream") {
    return null;
  }
  if (typeof ref.url !== "string" || ref.url.trim() === "") {
    return null;
  }
  return isAbsoluteMediaUrl(ref.url) ? ref.url : null;
}

export function fitViewportWithinBounds(
  viewport: FrameViewport,
  bounds: FrameBounds,
): FrameBounds | null {
  if (
    viewport.width <= 0 ||
    viewport.height <= 0 ||
    bounds.width <= 0 ||
    bounds.height <= 0
  ) {
    return null;
  }

  const scale = Math.min(bounds.width / viewport.width, bounds.height / viewport.height);
  if (!Number.isFinite(scale) || scale <= 0) {
    return null;
  }

  return {
    width: viewport.width * scale,
    height: viewport.height * scale,
  };
}

export function resolveImmersiveViewportBounds(
  viewport: FrameViewport,
  candidates: FrameMeasurementCandidate[],
  options: {
    windowHeight: number;
    fullscreenActive: boolean;
  },
): FrameBounds | null {
  const fullscreenBottomInset = options.fullscreenActive ? 0 : 32;
  let bestBounds: FrameBounds | null = null;
  let bestArea = 0;

  for (const candidate of candidates) {
    const stageTop = Math.max(0, candidate.top);
    const visibleHeight = Math.max(
      0,
      options.windowHeight - stageTop - fullscreenBottomInset,
    );
    const boundedHeight = Math.min(candidate.height, visibleHeight);
    const nextBounds = fitViewportWithinBounds(viewport, {
      width: candidate.width,
      height: boundedHeight,
    });
    if (!nextBounds) {
      continue;
    }
    const nextArea = nextBounds.width * nextBounds.height;
    if (nextArea <= bestArea) {
      continue;
    }
    bestBounds = nextBounds;
    bestArea = nextArea;
  }

  return bestBounds;
}

function readMeasurementCandidate(element: HTMLElement | null): FrameMeasurementCandidate | null {
  if (!element) {
    return null;
  }
  const rect = element.getBoundingClientRect();
  return {
    width: rect.width,
    height: rect.height,
    top: rect.top,
  };
}

function collectImmersiveMeasurementCandidates(
  element: HTMLElement,
): FrameMeasurementCandidate[] {
  const candidates: FrameMeasurementCandidate[] = [];
  const seen = new Set<HTMLElement>();

  const appendCandidate = (candidateElement: HTMLElement | null) => {
    if (!candidateElement || seen.has(candidateElement)) {
      return;
    }
    seen.add(candidateElement);
    const candidate = readMeasurementCandidate(candidateElement);
    if (!candidate) {
      return;
    }
    candidates.push(candidate);
  };

  appendCandidate(element);
  appendCandidate(element.parentElement);
  appendCandidate(element.closest(".session-stage__surface"));
  appendCandidate(element.closest(".session-stage"));

  return candidates;
}

function resolveViewportStyle(
  viewport: FrameViewport | null,
  presentation: "default" | "immersive",
  fittedViewport: FrameBounds | null = null,
): CSSProperties {
  if (presentation === "immersive") {
    if (viewport && fittedViewport) {
      return {
        width: `${fittedViewport.width}px`,
        height: `${fittedViewport.height}px`,
        maxWidth: "100%",
        maxHeight: "100%",
        aspectRatio: `${viewport.width} / ${viewport.height}`,
      };
    }
    if (!viewport) {
      return {
        width: "100%",
        maxWidth: "100%",
      };
    }
    return {
      width: "auto",
      maxWidth: "100%",
      aspectRatio: `${viewport.width} / ${viewport.height}`,
    };
  }
  if (!viewport) {
    return {
      width: "min(100%, 26rem)",
      maxWidth: "100%",
    };
  }

  const scaledWidth = Math.min(Math.max(Math.round(viewport.width * 2.25), 320), 560);
  return {
    width: "min(100%, 26rem)",
    maxWidth: `${scaledWidth}px`,
    aspectRatio: `${viewport.width} / ${viewport.height}`,
  };
}

function resolveImmersiveCanvasStyle(): CSSProperties {
  return {
    width: "100%",
    height: "100%",
    maxHeight: "none",
  };
}

function resolveImmersiveImageStyle(fit: string): CSSProperties {
  return {
    width: "100%",
    height: "100%",
    maxHeight: "none",
    objectFit: fit === "cover" ? "cover" : "contain",
  };
}

function clampHoldTicks(value: number, controls: FrameKeyboardControls): number {
  const min = Math.max(1, Math.round(controls.holdTicksMin ?? 1));
  const max = Math.max(min, Math.round(controls.holdTicksMax ?? value));
  return Math.max(min, Math.min(max, Math.round(value)));
}

function resolveReleasedHoldTicks(durationMs: number, controls: FrameKeyboardControls): number {
  const tickMs = Math.max(1, Number(controls.holdTickMs ?? 16));
  return clampHoldTicks(durationMs / tickMs, controls);
}

function resolveKeyboardHoldTicks(
  controls: FrameKeyboardControls,
  options: KeyboardDispatchOptions,
): number | null {
  if (options.phase === "release" && options.releasedHoldTicks !== undefined) {
    return clampHoldTicks(options.releasedHoldTicks, controls);
  }
  if (options.phase === "heartbeat" && controls.heartbeatHoldTicks !== undefined) {
    return clampHoldTicks(controls.heartbeatHoldTicks, controls);
  }
  if ((options.phase === "press" || options.phase === undefined) && controls.initialHoldTicks !== undefined) {
    return clampHoldTicks(controls.initialHoldTicks, controls);
  }
  return null;
}

function mergeFrameStyle(
  baseStyle: CSSProperties | undefined,
  optimisticStyle: CSSProperties | undefined,
): CSSProperties | undefined {
  if (!baseStyle && !optimisticStyle) {
    return undefined;
  }
  return {
    ...(baseStyle ?? {}),
    ...(optimisticStyle ?? {}),
  };
}

function resolveOptimisticFrameStyle(
  controls: FrameKeyboardControls | undefined,
  offset: FrameOptimisticOffset,
): CSSProperties | undefined {
  if (!controls?.resolveOptimisticOffset) {
    return undefined;
  }
  return {
    transform: `translate3d(${offset.x}px, ${offset.y}px, 0)`,
    transition: "transform 80ms ease-out",
    willChange: "transform",
  };
}

export function FrameSurface({
  gameLabel,
  session,
  scene,
  submitInput,
  mediaSubscribe,
  keyboardControls,
  presentation = "default",
  showStatusLine = true,
  showViewText = true,
}: FrameSurfaceProps) {
  const rootRef = useRef<HTMLElement | null>(null);
  const isImmersive = presentation === "immersive";
  const frameScene = readFrameScene(scene);
  const actionDescriptors = readFrameActions(scene);
  const [immersiveViewportBounds, setImmersiveViewportBounds] = useState<FrameBounds | null>(null);
  const [observedViewport, setObservedViewport] = useState<FrameViewport | null>(null);
  const pressedKeysRef = useRef<Set<string>>(new Set());
  const pressedKeyStartedAtRef = useRef<Map<string, number>>(new Map());
  const lastKeyboardDispatchRef = useRef<string | null>(null);
  const lastKeyboardActionTsByIdRef = useRef<Map<string, number>>(new Map());
  const keyboardInputSequenceRef = useRef(0);
  const [optimisticOffset, setOptimisticOffset] = useState<FrameOptimisticOffset>({ x: 0, y: 0 });
  const primaryMediaRef = scene?.media?.primary;
  const primaryMediaId = primaryMediaRef?.mediaId;
  const mediaState = useMediaSource({
    sessionId: session.sessionId,
    mediaId: primaryMediaId ?? "",
    subscribe: mediaSubscribe,
  });
  const resolvedActorId = frameScene ? resolveFrameActorId(session, scene, frameScene) : null;
  const effectiveViewport = frameScene?.frame.viewport ?? observedViewport;
  const canSubmitActions =
    frameScene !== null && session.scheduling.acceptsHumanIntent && resolvedActorId !== null;
  const keyboardSceneToken =
    scene?.sceneId ??
    (frameScene
      ? `${session.sessionId}:${frameScene.status.tick}:${frameScene.status.moveCount}:${frameScene.status.lastMove ?? ""}`
      : session.sessionId);
  const keyboardDispatchContextRef = useRef({
    actionDescriptors,
    canSubmitActions,
    frameScene,
    keyboardControls,
    keyboardSceneToken,
    resolvedActorId,
    submitInput,
  });
  keyboardDispatchContextRef.current = {
    actionDescriptors,
    canSubmitActions,
    frameScene,
    keyboardControls,
    keyboardSceneToken,
    resolvedActorId,
    submitInput,
  };

  function dispatchKeyboardAction(
    pressedKeys: ReadonlySet<string>,
    options: KeyboardDispatchOptions = {},
  ): void {
    const context = keyboardDispatchContextRef.current;
    if (
      !context.keyboardControls ||
      !context.frameScene ||
      !context.canSubmitActions ||
      !context.resolvedActorId
    ) {
      return;
    }
    const actionDescriptor = context.keyboardControls.resolveAction(
      context.actionDescriptors,
      pressedKeys,
    );
    if (!actionDescriptor) {
      if (pressedKeys.size === 0) {
        lastKeyboardDispatchRef.current = null;
      }
      if (context.keyboardControls.resolveOptimisticOffset) {
        setOptimisticOffset({ x: 0, y: 0 });
      }
      return;
    }
    const nowMs = performance.now();
    const throttleMs = context.keyboardControls.resolveActionThrottleMs?.(actionDescriptor) ?? 0;
    if (throttleMs > 0) {
      const lastSubmittedAt = lastKeyboardActionTsByIdRef.current.get(actionDescriptor.id);
      if (
        lastSubmittedAt !== undefined &&
        nowMs - lastSubmittedAt < throttleMs
      ) {
        return;
      }
    }
    const pressedKeysToken = [...pressedKeys].sort().join(",");
    const dispatchToken = `${context.keyboardSceneToken}:${context.resolvedActorId}:${actionDescriptor.id}:${pressedKeysToken}`;
    if (!options.force && lastKeyboardDispatchRef.current === dispatchToken) {
      return;
    }
    lastKeyboardDispatchRef.current = dispatchToken;
    if (throttleMs > 0) {
      lastKeyboardActionTsByIdRef.current.set(actionDescriptor.id, nowMs);
    }
    keyboardInputSequenceRef.current += 1;
    const timedHoldTicks = resolveKeyboardHoldTicks(context.keyboardControls, options);
    const actionPayload = isRecord(actionDescriptor.payload)
      ? {
          ...actionDescriptor.payload,
          ...(timedHoldTicks !== null ? { hold_ticks: timedHoldTicks } : {}),
          metadata: {
            ...(isRecord(actionDescriptor.payload.metadata) ? actionDescriptor.payload.metadata : {}),
            input_seq: keyboardInputSequenceRef.current,
            input_client_ts_ms: Date.now(),
            realtime_input: true,
          },
        }
      : actionDescriptor.payload;
    if (context.keyboardControls.resolveOptimisticOffset) {
      setOptimisticOffset(context.keyboardControls.resolveOptimisticOffset(pressedKeys));
    }
    void context.submitInput({
      playerId: context.resolvedActorId,
      actionPayload,
    })
      .catch(() => {});
  }

  useEffect(() => {
    setObservedViewport(null);
  }, [primaryMediaId]);

  useEffect(() => {
    setOptimisticOffset({ x: 0, y: 0 });
  }, [keyboardSceneToken]);

  useEffect(() => {
    if (!keyboardControls) {
      pressedKeysRef.current = new Set();
      pressedKeyStartedAtRef.current = new Map();
      lastKeyboardDispatchRef.current = null;
      setOptimisticOffset({ x: 0, y: 0 });
      return;
    }
    const controls = keyboardControls;
    const watchedKeys = new Set(
      controls.watchedKeys.map((key) => normalizeKeyboardKey(key)),
    );

    function resetKeyboardState(): void {
      const hadPressedKeys = pressedKeysRef.current.size > 0;
      pressedKeysRef.current = new Set();
      pressedKeyStartedAtRef.current = new Map();
      lastKeyboardDispatchRef.current = null;
      setOptimisticOffset({ x: 0, y: 0 });
      if (hadPressedKeys) {
        dispatchKeyboardAction(new Set(), {
          force: true,
          phase: "release",
        });
      }
    }

    function handleKeyDown(event: KeyboardEvent) {
      const normalizedKey = normalizeKeyboardKey(event.key);
      if (!watchedKeys.has(normalizedKey) || isEditingTarget(event.target)) {
        return;
      }
      const nextPressedKeys = new Set(pressedKeysRef.current);
      const sizeBefore = nextPressedKeys.size;
      nextPressedKeys.add(normalizedKey);
      pressedKeysRef.current = nextPressedKeys;
      event.preventDefault();
      if (nextPressedKeys.size !== sizeBefore) {
        pressedKeyStartedAtRef.current.set(normalizedKey, performance.now());
        dispatchKeyboardAction(nextPressedKeys, { phase: "press" });
      }
    }

    function handleKeyUp(event: KeyboardEvent) {
      const normalizedKey = normalizeKeyboardKey(event.key);
      if (!watchedKeys.has(normalizedKey) || isEditingTarget(event.target)) {
        return;
      }
      const nextPressedKeys = new Set(pressedKeysRef.current);
      const hadKey = nextPressedKeys.delete(normalizedKey);
      pressedKeysRef.current = nextPressedKeys;
      event.preventDefault();
      if (hadKey) {
        const startedAt = pressedKeyStartedAtRef.current.get(normalizedKey);
        pressedKeyStartedAtRef.current.delete(normalizedKey);
        const releasedHoldTicks =
          startedAt === undefined
            ? undefined
            : resolveReleasedHoldTicks(performance.now() - startedAt, controls);
        if (nextPressedKeys.size === 0) {
          dispatchKeyboardAction(nextPressedKeys, {
            phase: "release",
            releasedHoldTicks,
          });
        } else {
          dispatchKeyboardAction(nextPressedKeys, {
            phase: "press",
          });
        }
      }
    }

    function handleWindowBlur() {
      resetKeyboardState();
    }

    function handleVisibilityChange() {
      if (document.hidden) {
        resetKeyboardState();
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    window.addEventListener("blur", handleWindowBlur);
    document.addEventListener("visibilitychange", handleVisibilityChange);
    const heartbeatMs = controls.heartbeatMs;
    const heartbeatTimer =
      heartbeatMs !== undefined && heartbeatMs > 0
        ? window.setInterval(() => {
            if (pressedKeysRef.current.size === 0) {
              return;
            }
            dispatchKeyboardAction(new Set(pressedKeysRef.current), {
              force: true,
              phase: "heartbeat",
            });
          }, Math.max(1, heartbeatMs))
        : null;
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
      window.removeEventListener("blur", handleWindowBlur);
      document.removeEventListener("visibilitychange", handleVisibilityChange);
      if (heartbeatTimer !== null) {
        window.clearInterval(heartbeatTimer);
      }
    };
  }, [actionDescriptors, canSubmitActions, keyboardControls, keyboardSceneToken, resolvedActorId, submitInput]);

  useLayoutEffect(() => {
    if (!isImmersive || !effectiveViewport) {
      setImmersiveViewportBounds(null);
      return;
    }
    const viewport = effectiveViewport;

    const element = rootRef.current;
    if (!element) {
      setImmersiveViewportBounds(null);
      return;
    }

    let frameId = 0;

    const measure = () => {
      const windowHeight =
        window.innerHeight ||
        document.documentElement.clientHeight ||
        element.getBoundingClientRect().height;
      const nextBounds = resolveImmersiveViewportBounds(
        viewport,
        collectImmersiveMeasurementCandidates(element),
        {
          windowHeight,
          fullscreenActive: Boolean(document.fullscreenElement),
        },
      );

      setImmersiveViewportBounds((currentBounds) => {
        if (!nextBounds) {
          return null;
        }
        if (
          currentBounds &&
          Math.abs(currentBounds.width - nextBounds.width) < 0.5 &&
          Math.abs(currentBounds.height - nextBounds.height) < 0.5
        ) {
          return currentBounds;
        }
        return nextBounds;
      });
    };

    const scheduleMeasure = () => {
      window.cancelAnimationFrame(frameId);
      frameId = window.requestAnimationFrame(measure);
    };

    scheduleMeasure();

    let resizeObserver: ResizeObserver | null = null;
    if (typeof ResizeObserver !== "undefined") {
      resizeObserver = new ResizeObserver(() => {
        scheduleMeasure();
      });
      resizeObserver.observe(element);
      if (element.parentElement) {
        resizeObserver.observe(element.parentElement);
      }
    }

    window.addEventListener("resize", scheduleMeasure);
    document.addEventListener("fullscreenchange", scheduleMeasure);

    return () => {
      window.cancelAnimationFrame(frameId);
      resizeObserver?.disconnect();
      window.removeEventListener("resize", scheduleMeasure);
      document.removeEventListener("fullscreenchange", scheduleMeasure);
    };
  }, [effectiveViewport?.height, effectiveViewport?.width, isImmersive]);

  if (!frameScene) {
    return (
      <section className="plugin-stage-card">
        <p className="eyebrow">{gameLabel}</p>
        <h2>Frame unavailable</h2>
        <p className="plugin-stage-card__copy">Waiting for frame scene data and media sources.</p>
      </section>
    );
  }

  const actorLabel = formatFrameActorLabel(session, resolvedActorId);
  const imageClassName = resolveImageClassName(frameScene.frame.fit);
  const canvasClassName = resolveCanvasClassName(frameScene.frame.fit);
  const directImageSrc = resolveDirectFrameImageSrc(primaryMediaRef);
  const imageSrc =
    directImageSrc ??
    (typeof mediaState?.src === "string"
      ? mediaState.src
      : null);
  const lowLatencyStreamUrl = resolveLowLatencyStreamUrl(
    mediaState?.ref?.url ?? primaryMediaRef?.url,
    mediaState?.ref?.transport ?? primaryMediaRef?.transport,
  );
  const viewportStyle = resolveViewportStyle(
    effectiveViewport,
    presentation,
    immersiveViewportBounds,
  );
  const immersiveCanvasStyle = isImmersive ? resolveImmersiveCanvasStyle() : undefined;
  const immersiveImageStyle = isImmersive
    ? resolveImmersiveImageStyle(frameScene.frame.fit)
    : undefined;
  const optimisticFrameStyle = resolveOptimisticFrameStyle(keyboardControls, optimisticOffset);

  return (
    <section
      className={[
        "frame-surface",
        isImmersive ? "frame-surface--immersive" : "",
      ]
        .filter((value) => value !== "")
        .join(" ")}
      data-testid="frame-surface-root"
      ref={rootRef}
    >
      {!isImmersive ? (
        <>
          <div className="frame-surface__header">
            <p className="eyebrow">Frame</p>
            <p className="frame-surface__actor-label">{actorLabel}</p>
          </div>
          <h2 className="frame-surface__title">{frameScene.frame.title}</h2>
          {frameScene.frame.subtitle ? (
            <p className="frame-surface__subtitle">{frameScene.frame.subtitle}</p>
          ) : null}
        </>
      ) : null}

      <div
        className="frame-surface__viewport"
        data-testid="frame-surface-viewport"
        style={viewportStyle}
      >
        {lowLatencyStreamUrl ? (
          <LowLatencyFrameCanvas
            altText={frameScene.frame.altText}
            className={canvasClassName}
            onFrameSizeChange={(size) => {
              setObservedViewport((current) => {
                if (
                  current &&
                  current.width === size.width &&
                  current.height === size.height
                ) {
                  return current;
                }
                return {
                  width: size.width,
                  height: size.height,
                };
              });
            }}
            streamUrl={lowLatencyStreamUrl}
            style={mergeFrameStyle(immersiveCanvasStyle, optimisticFrameStyle)}
          />
        ) : imageSrc ? (
          <img
            alt={frameScene.frame.altText}
            className={imageClassName}
            data-testid="frame-surface-image"
            src={imageSrc}
            onLoad={(event) => {
              const target = event.currentTarget;
              const width = target.naturalWidth || target.width;
              const height = target.naturalHeight || target.height;
              if (width <= 0 || height <= 0) {
                return;
              }
              setObservedViewport((current) => {
                if (
                  current &&
                  current.width === width &&
                  current.height === height
                ) {
                  return current;
                }
                return { width, height };
              });
            }}
            style={mergeFrameStyle(immersiveImageStyle, optimisticFrameStyle)}
          />
        ) : (
          <div className="frame-surface__fallback">Loading frame...</div>
        )}
      </div>

      {!isImmersive ? (
        <>
          {showStatusLine ? (
            <p className="frame-surface__status-line" data-testid="frame-status-line">
              {formatStatusLine(frameScene)}
            </p>
          ) : null}
          {showViewText && frameScene.viewText ? (
            <p className="frame-surface__view-text">{frameScene.viewText}</p>
          ) : null}
          {keyboardControls ? (
            <p className="frame-surface__keyboard-hint" data-testid="frame-keyboard-hint">
              {keyboardControls.hint}
            </p>
          ) : null}

          {actionDescriptors.length > 0 ? (
            <div className="frame-surface__actions">
              {actionDescriptors.map((actionDescriptor) => (
                <button
                  className="frame-surface__action-chip"
                  disabled={!canSubmitActions}
                  key={actionDescriptor.id}
                  onClick={() => {
                    if (!canSubmitActions || !resolvedActorId) {
                      return;
                    }
                    void submitInput({
                      playerId: resolvedActorId,
                      actionPayload: actionDescriptor.payload,
                    });
                  }}
                  type="button"
                >
                  {actionDescriptor.label}
                </button>
              ))}
            </div>
          ) : null}
        </>
      ) : null}
    </section>
  );
}

function isEditingTarget(target: EventTarget | null): boolean {
  if (!target || typeof target !== "object") {
    return false;
  }
  const candidate = target as {
    tagName?: string;
    isContentEditable?: boolean;
  };
  const tagName = typeof candidate.tagName === "string" ? candidate.tagName.toUpperCase() : "";
  return (
    candidate.isContentEditable === true ||
    tagName === "INPUT" ||
    tagName === "TEXTAREA" ||
    tagName === "SELECT" ||
    tagName === "BUTTON"
  );
}

import { useEffect, useRef, type CSSProperties } from "react";
import type { VisualScene, VisualSession } from "../../gateway/types";
import { useMediaSource } from "../sdk/useMediaSource";
import type { ArenaMediaSubscriber } from "../sdk/contracts";
import type { ActionIntent } from "../sdk/input";
import type { FrameActionDescriptor, FrameKeyboardControls } from "./contracts";
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
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function readString(value: unknown): string | null {
  return typeof value === "string" && value.trim() !== "" ? value : null;
}

function readNumber(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function readNullableNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
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
      reward: readNullableNumber(status.reward),
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

function resolveViewportStyle(
  viewport: FrameViewport | null,
  presentation: "default" | "immersive",
): CSSProperties {
  if (presentation === "immersive") {
    if (!viewport) {
      return {
        width: "100%",
        maxWidth: "100%",
      };
    }
    return {
      width: "100%",
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

export function FrameSurface({
  gameLabel,
  session,
  scene,
  submitInput,
  mediaSubscribe,
  keyboardControls,
  presentation = "default",
}: FrameSurfaceProps) {
  const isImmersive = presentation === "immersive";
  const frameScene = readFrameScene(scene);
  const actionDescriptors = readFrameActions(scene);
  const pressedKeysRef = useRef<Set<string>>(new Set());
  const lastKeyboardDispatchRef = useRef<string | null>(null);
  const keyboardInputSequenceRef = useRef(0);
  const primaryMediaId = scene?.media?.primary?.mediaId;
  const mediaState = useMediaSource({
    sessionId: session.sessionId,
    mediaId: primaryMediaId ?? "",
    subscribe: mediaSubscribe,
  });
  const resolvedActorId = frameScene ? resolveFrameActorId(session, scene, frameScene) : null;
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

  function dispatchKeyboardAction(pressedKeys: ReadonlySet<string>): void {
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
      return;
    }
    const pressedKeysToken = [...pressedKeys].sort().join(",");
    const dispatchToken = `${context.keyboardSceneToken}:${context.resolvedActorId}:${actionDescriptor.id}:${pressedKeysToken}`;
    if (lastKeyboardDispatchRef.current === dispatchToken) {
      return;
    }
    lastKeyboardDispatchRef.current = dispatchToken;
    keyboardInputSequenceRef.current += 1;
    const actionPayload = isRecord(actionDescriptor.payload)
      ? {
          ...actionDescriptor.payload,
          metadata: {
            ...(isRecord(actionDescriptor.payload.metadata) ? actionDescriptor.payload.metadata : {}),
            input_seq: keyboardInputSequenceRef.current,
            realtime_input: true,
          },
        }
      : actionDescriptor.payload;
    void context.submitInput({
      playerId: context.resolvedActorId,
      actionPayload,
    })
      .catch(() => {});
  }

  useEffect(() => {
    if (!keyboardControls) {
      pressedKeysRef.current = new Set();
      lastKeyboardDispatchRef.current = null;
      return;
    }
    const watchedKeys = new Set(
      keyboardControls.watchedKeys.map((key) => normalizeKeyboardKey(key)),
    );

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
        dispatchKeyboardAction(nextPressedKeys);
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
        dispatchKeyboardAction(nextPressedKeys);
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, [actionDescriptors, canSubmitActions, keyboardControls, keyboardSceneToken, resolvedActorId, submitInput]);

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
  const imageSrc = typeof mediaState?.src === "string" ? mediaState.src : null;
  const lowLatencyStreamUrl = resolveLowLatencyStreamUrl(mediaState?.ref?.url, mediaState?.ref?.transport);
  const viewportStyle = resolveViewportStyle(frameScene.frame.viewport, presentation);

  return (
    <section
      className={[
        "frame-surface",
        isImmersive ? "frame-surface--immersive" : "",
      ]
        .filter((value) => value !== "")
        .join(" ")}
      data-testid="frame-surface-root"
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
            streamUrl={lowLatencyStreamUrl}
          />
        ) : imageSrc ? (
          <img
            alt={frameScene.frame.altText}
            className={imageClassName}
            data-testid="frame-surface-image"
            src={imageSrc}
          />
        ) : (
          <div className="frame-surface__fallback">Loading frame...</div>
        )}

        {frameScene.overlays.length > 0 ? (
          <div className="frame-surface__overlay-strip">
            {frameScene.overlays.map((overlay) => (
              <div
                className="frame-surface__overlay-badge"
                key={`${overlay.kind}:${overlay.label}:${overlay.value}`}
              >
                <span>{overlay.label}</span>
                <strong>{overlay.value}</strong>
              </div>
            ))}
          </div>
        ) : null}
      </div>

      {!isImmersive ? (
        <>
          <p className="frame-surface__status-line" data-testid="frame-status-line">
            {formatStatusLine(frameScene)}
          </p>
          {frameScene.viewText ? <p className="frame-surface__view-text">{frameScene.viewText}</p> : null}
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

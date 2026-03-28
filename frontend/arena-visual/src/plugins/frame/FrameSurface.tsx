import type { VisualScene, VisualSession } from "../../gateway/types";
import { useMediaSource } from "../sdk/useMediaSource";
import type { ArenaMediaSubscriber } from "../sdk/contracts";

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

interface FrameActionDescriptor {
  id: string;
  label: string;
  payload: Record<string, unknown>;
}

interface FrameSurfaceProps {
  gameLabel: string;
  session: VisualSession;
  scene?: VisualScene;
  submitAction: (payload: Record<string, unknown>) => Promise<void>;
  mediaSubscribe: ArenaMediaSubscriber;
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

export function FrameSurface({
  gameLabel,
  session,
  scene,
  submitAction,
  mediaSubscribe,
}: FrameSurfaceProps) {
  const frameScene = readFrameScene(scene);

  if (!frameScene) {
    return (
      <section className="plugin-stage-card">
        <p className="eyebrow">{gameLabel}</p>
        <h2>Frame unavailable</h2>
        <p className="plugin-stage-card__copy">Waiting for frame scene data and media sources.</p>
      </section>
    );
  }

  const actionDescriptors = readFrameActions(scene);
  const resolvedActorId = resolveFrameActorId(session, scene, frameScene);
  const canSubmitActions = session.scheduling.acceptsHumanIntent && resolvedActorId !== null;
  const actorLabel = formatFrameActorLabel(session, resolvedActorId);
  const primaryMediaId = scene?.media?.primary?.mediaId;
  const mediaState = useMediaSource({
    sessionId: session.sessionId,
    mediaId: primaryMediaId ?? "",
    subscribe: mediaSubscribe,
  });
  const imageClassName = resolveImageClassName(frameScene.frame.fit);
  const imageSrc = typeof mediaState?.src === "string" ? mediaState.src : null;

  return (
    <section className="frame-surface">
      <div className="frame-surface__header">
        <p className="eyebrow">Frame</p>
        <p className="frame-surface__actor-label">{actorLabel}</p>
      </div>
      <h2 className="frame-surface__title">{frameScene.frame.title}</h2>
      {frameScene.frame.subtitle ? (
        <p className="frame-surface__subtitle">{frameScene.frame.subtitle}</p>
      ) : null}

      <div className="frame-surface__viewport" data-testid="frame-surface-viewport">
        {imageSrc ? (
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

      <p className="frame-surface__status-line" data-testid="frame-status-line">
        {formatStatusLine(frameScene)}
      </p>
      {frameScene.viewText ? <p className="frame-surface__view-text">{frameScene.viewText}</p> : null}

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
                void submitAction({
                  playerId: resolvedActorId,
                  action: actionDescriptor.payload,
                });
              }}
              type="button"
            >
              {actionDescriptor.label}
            </button>
          ))}
        </div>
      ) : null}
    </section>
  );
}

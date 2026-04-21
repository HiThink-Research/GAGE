import { useRef } from "react";
import type { VisualScene, VisualSession } from "../../gateway/types";
import type { ArenaPluginRenderProps } from "../sdk/contracts";
import { useMediaSource } from "../sdk/useMediaSource";
import { LowLatencyFrameCanvas, resolveLowLatencyStreamUrl } from "../frame/LowLatencyFrameCanvas";

type OpenRAActionIntent = {
  playerId: string;
  actionPayload: Record<string, unknown>;
};

interface OpenRAFrameViewport {
  width: number;
  height: number;
}

interface OpenRAMapBounds {
  x: number | null;
  y: number | null;
  width: number | null;
  height: number | null;
}

interface OpenRAMapSize {
  width: number | null;
  height: number | null;
}

interface OpenRAFrameData {
  title: string;
  subtitle: string | null;
  altText: string;
  streamId: string | null;
  fit: string;
  viewport: OpenRAFrameViewport | null;
}

interface OpenRAMapData {
  id: string | null;
  modId: string | null;
  title: string | null;
  gridSize: OpenRAMapSize | null;
  bounds: OpenRAMapBounds | null;
  imageSize: OpenRAMapSize | null;
  previewSource: string | null;
}

interface OpenRASelectionData {
  unitIds: string[];
  primaryUnitId: string | null;
}

interface OpenRAEconomyData {
  credits: number | null;
  incomePerMinute: number | null;
  powerProduced: number | null;
  powerUsed: number | null;
}

interface OpenRAObjectiveData {
  id: string;
  label: string;
  status: string;
}

interface OpenRAUnitData {
  id: string;
  owner: string | null;
  label: string;
  kind: string | null;
  hp: number | null;
  status: string | null;
  position: { x: number | null; y: number | null } | null;
  selected: boolean;
}

interface OpenRAProductionItemData {
  id: string;
  label: string;
  progress: number | null;
}

interface OpenRAProductionQueueData {
  buildingId: string;
  label: string;
  items: OpenRAProductionItemData[];
}

interface OpenRAActionData {
  id: string;
  label: string;
  text: string;
  payload: Record<string, unknown>;
}

interface OpenRARTSData {
  frame: OpenRAFrameData;
  map: OpenRAMapData;
  selection: OpenRASelectionData;
  economy: OpenRAEconomyData;
  objectives: OpenRAObjectiveData[];
  units: OpenRAUnitData[];
  production: OpenRAProductionQueueData[];
  legalActions: OpenRAActionData[];
}

interface OpenRASceneData {
  frame: OpenRAFrameData;
  status: {
    activePlayerId: string | null;
    observerPlayerId: string | null;
    tick: number;
    step: number;
    moveCount: number;
    lastMove: string | null;
    reward: number | null;
  };
  rts: OpenRARTSData;
  viewText: string | null;
  overlays: OpenRAOverlayData[];
}

interface OpenRAOverlayData {
  kind: string;
  label: string;
  value: string;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function readString(value: unknown): string | null {
  return typeof value === "string" && value.trim() !== "" ? value : null;
}

function readNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function readBoolean(value: unknown): boolean {
  return value === true;
}

function readStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .map((item) => readString(item))
    .filter((item): item is string => item !== null);
}

function readFrameData(source: Record<string, unknown>): OpenRAFrameData {
  const frameSource = isRecord(source.frame) ? source.frame : ({} as Record<string, unknown>);
  return {
    title: readString(frameSource.title) ?? "OpenRA RTS",
    subtitle: readString(frameSource.subtitle),
    altText: readString(frameSource.altText) ?? "OpenRA RTS frame",
    streamId: readString(frameSource.streamId),
    fit: readString(frameSource.fit) ?? "contain",
    viewport: readViewport(frameSource.viewport),
  };
}

function readViewport(value: unknown): OpenRAFrameViewport | null {
  if (!isRecord(value)) {
    return null;
  }
  const width = readNumber(value.width);
  const height = readNumber(value.height);
  if (width === null || height === null || width <= 0 || height <= 0) {
    return null;
  }
  return {
    width,
    height,
  };
}

function readRTSData(source: Record<string, unknown>): OpenRARTSData {
  const rtsSource = isRecord(source.rts) ? source.rts : ({} as Record<string, unknown>);
  const selectionSource = isRecord(rtsSource.selection)
    ? rtsSource.selection
    : ({} as Record<string, unknown>);
  const economySource = isRecord(rtsSource.economy)
    ? rtsSource.economy
    : ({} as Record<string, unknown>);
  return {
    frame: readFrameData(source),
    map: readMapData(rtsSource.map),
    selection: {
      unitIds: readStringArray(selectionSource.unitIds ?? selectionSource.unit_ids),
      primaryUnitId:
        readString(selectionSource.primaryUnitId ?? selectionSource.primary_unit_id),
    },
    economy: {
      credits: readNumber(economySource.credits),
      incomePerMinute: readNumber(
        economySource.incomePerMinute ?? economySource.income_per_minute,
      ),
      powerProduced: readNumber(
        isRecord(economySource.power)
          ? economySource.power.produced
          : economySource.power_produced,
      ),
      powerUsed: readNumber(
        isRecord(economySource.power) ? economySource.power.used : economySource.power_used,
      ),
    },
    objectives: readObjectives(rtsSource.objectives),
    units: readUnits(rtsSource.units),
    production: readProductionQueues(rtsSource.production),
    legalActions: readLegalActions(source.legalActions ?? source.legal_actions),
  };
}

function readMapData(value: unknown): OpenRAMapData {
  const payload = isRecord(value) ? value : ({} as Record<string, unknown>);
  const gridSizeSource = isRecord(payload.gridSize)
    ? payload.gridSize
    : isRecord(payload.map_size)
      ? payload.map_size
      : ({} as Record<string, unknown>);
  const boundsSource = isRecord(payload.bounds)
    ? payload.bounds
    : ({} as Record<string, unknown>);
  const imageSizeSource = isRecord(payload.imageSize)
    ? payload.imageSize
    : isRecord(payload.image_size)
      ? payload.image_size
      : ({} as Record<string, unknown>);
  const readSize = (source: Record<string, unknown>): OpenRAMapSize | null => {
    const width = readNumber(source.width);
    const height = readNumber(source.height);
    if (width === null && height === null) {
      return null;
    }
    return { width, height };
  };

  return {
    id: readString(payload.id),
    modId: readString(payload.modId ?? payload.mod_id),
    title: readString(payload.title),
    gridSize: readSize(gridSizeSource),
    bounds: {
      x: readNumber(boundsSource.x),
      y: readNumber(boundsSource.y),
      width: readNumber(boundsSource.width),
      height: readNumber(boundsSource.height),
    },
    imageSize: readSize(imageSizeSource),
    previewSource: readString(payload.previewSource ?? payload.preview_source),
  };
}

function readObjectives(value: unknown): OpenRAObjectiveData[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .filter(isRecord)
    .map((item, index) => ({
      id: readString(item.id) ?? `objective-${index + 1}`,
      label: readString(item.label) ?? readString(item.name) ?? `Objective ${index + 1}`,
      status: readString(item.status) ?? "unknown",
    }));
}

function readUnits(value: unknown): OpenRAUnitData[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .filter(isRecord)
    .map((item, index) => {
      const position = isRecord(item.position) ? item.position : ({} as Record<string, unknown>);
      return {
        id: readString(item.id) ?? `unit-${index + 1}`,
        owner: readString(item.owner),
        label: readString(item.label) ?? readString(item.name) ?? `Unit ${index + 1}`,
        kind: readString(item.kind),
        hp: readNumber(item.hp),
        status: readString(item.status),
        position:
          readNumber(position.x) !== null || readNumber(position.y) !== null
            ? { x: readNumber(position.x), y: readNumber(position.y) }
            : null,
        selected: readBoolean(item.selected),
      };
    });
}

function readProductionQueues(value: unknown): OpenRAProductionQueueData[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .filter(isRecord)
    .map((item, index) => {
      const queueItems = Array.isArray(item.items) ? item.items : [];
      return {
        buildingId: readString(item.buildingId ?? item.building_id) ?? `queue-${index + 1}`,
        label: readString(item.label) ?? readString(item.name) ?? `Queue ${index + 1}`,
        items: queueItems
          .filter(isRecord)
          .map((queueItem, itemIndex) => ({
            id: readString(queueItem.id) ?? `item-${itemIndex + 1}`,
            label: readString(queueItem.label) ?? `Item ${itemIndex + 1}`,
            progress: readNumber(queueItem.progress),
          })),
      };
    });
}

function readLegalActions(value: unknown): OpenRAActionData[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .filter(isRecord)
    .map((item, index) => {
      const payload: Record<string, unknown> = {};
      for (const [key, rawValue] of Object.entries(item)) {
        if (key === "id" || key === "label" || key === "text" || key === "payloadSchema") {
          continue;
        }
        payload[key] = rawValue;
      }
      const actionPayload = isRecord(item.payloadSchema) ? item.payloadSchema : payload;
      return {
        id: readString(item.id) ?? `action-${index + 1}`,
        label: readString(item.label) ?? readString(item.text) ?? `Action ${index + 1}`,
        text: readString(item.text) ?? readString(item.label) ?? `Action ${index + 1}`,
        payload: actionPayload,
      };
    });
}

function readScene(scene?: VisualScene): OpenRASceneData | null {
  if (!scene || scene.kind !== "rts" || !isRecord(scene.body)) {
    return null;
  }

  const body = scene.body;
  const rtsData = readRTSData(body);
  const legalActions = readLegalActions(scene.legalActions ?? body.legalActions ?? body.legal_actions);
  const statusSource = isRecord(body.status) ? body.status : ({} as Record<string, unknown>);
  const viewSource = isRecord(body.view) ? body.view : ({} as Record<string, unknown>);

  return {
    frame: rtsData.frame,
    status: {
      activePlayerId: readString(statusSource.activePlayerId ?? body.activePlayerId),
      observerPlayerId: readString(statusSource.observerPlayerId ?? body.observerPlayerId),
      tick: readNumber(statusSource.tick) ?? 0,
      step: readNumber(statusSource.step) ?? 0,
      moveCount: readNumber(statusSource.moveCount ?? statusSource.move_count) ?? 0,
      lastMove: readString(statusSource.lastMove ?? statusSource.last_move),
      reward: readNumber(statusSource.reward),
    },
    rts: {
      ...rtsData,
      legalActions,
    },
    viewText: readString(viewSource.text),
    overlays: readOverlays(scene.overlays),
  };
}

function readOverlays(value: unknown): OpenRAOverlayData[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .filter(isRecord)
    .map((item, index) => ({
      kind: readString(item.kind) ?? "badge",
      label: readString(item.label) ?? `Overlay ${index + 1}`,
      value: readString(item.value) ?? "",
    }))
    .filter((item) => item.value !== "");
}

function resolveActorId(session: VisualSession, scene?: VisualScene): string | null {
  if (
    session.observer.observerKind === "player" &&
    session.observer.observerId.trim() !== ""
  ) {
    return session.observer.observerId;
  }

  if (scene && isRecord(scene.body)) {
    const status = isRecord(scene.body.status)
      ? scene.body.status
      : ({} as Record<string, unknown>);
    const activePlayerId = readString(status.activePlayerId ?? scene.activePlayerId);
    if (activePlayerId) {
      return activePlayerId;
    }
  }

  return session.scheduling.activeActorId ?? null;
}

function formatProgress(progress: number | null): string {
  if (progress === null) {
    return "pending";
  }
  const percentage = Math.max(0, Math.min(100, Math.round(progress * 100)));
  return `${percentage}%`;
}

function formatPosition(position: { x: number | null; y: number | null } | null): string {
  if (!position) {
    return "unknown";
  }
  const x = position.x === null ? "?" : position.x;
  const y = position.y === null ? "?" : position.y;
  return `(${x}, ${y})`;
}

function renderValue(value: number | string | null | undefined): string {
  if (value === null || value === undefined) {
    return "n/a";
  }
  return String(value);
}

function renderPayloadPreview(payload: Record<string, unknown>): string {
  const preview = JSON.stringify(payload);
  if (preview.length <= 84) {
    return preview;
  }
  return `${preview.slice(0, 81)}...`;
}

function readPointerButton(button: number): string {
  switch (button) {
    case 1:
      return "middle";
    case 2:
      return "right";
    default:
      return "left";
  }
}

function readPointerButtonsMask(buttons: number): string[] {
  const resolved: string[] = [];
  if ((buttons & 1) === 1) {
    resolved.push("left");
  }
  if ((buttons & 2) === 2) {
    resolved.push("right");
  }
  if ((buttons & 4) === 4) {
    resolved.push("middle");
  }
  return resolved;
}

function readModifierList(event: {
  altKey?: boolean;
  ctrlKey?: boolean;
  metaKey?: boolean;
  shiftKey?: boolean;
}): string[] {
  const modifiers: string[] = [];
  if (event.shiftKey) {
    modifiers.push("shift");
  }
  if (event.ctrlKey) {
    modifiers.push("ctrl");
  }
  if (event.altKey) {
    modifiers.push("alt");
  }
  if (event.metaKey) {
    modifiers.push("meta");
  }
  return modifiers;
}

function resolveNativeSurfacePoint(
  element: HTMLElement,
  frame: OpenRAFrameData,
  clientX: number,
  clientY: number,
): { x: number; y: number; viewport: OpenRAFrameViewport } | null {
  const rect = element.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) {
    return null;
  }
  const viewport = frame.viewport ?? { width: rect.width, height: rect.height };
  const localX = Math.max(0, Math.min(rect.width, clientX - rect.left));
  const localY = Math.max(0, Math.min(rect.height, clientY - rect.top));
  return {
    x: Math.round((localX / rect.width) * viewport.width),
    y: Math.round((localY / rect.height) * viewport.height),
    viewport,
  };
}

function isDirectControlAction(action: OpenRAActionData): boolean {
  return !["bridge_input", "noop"].includes(action.id);
}

function clampPercentage(value: number): number {
  return Math.max(0, Math.min(100, value));
}

function resolveUnitMarkerStyle(
  unit: OpenRAUnitData,
  map: OpenRAMapData,
): { left: string; top: string } | null {
  if (!unit.position || unit.position.x === null || unit.position.y === null) {
    return null;
  }
  const bounds = map.bounds;
  const gridSize = map.gridSize;
  const originX = bounds?.x ?? 0;
  const originY = bounds?.y ?? 0;
  const spanWidth = bounds?.width ?? gridSize?.width ?? null;
  const spanHeight = bounds?.height ?? gridSize?.height ?? null;
  if (spanWidth === null || spanHeight === null || spanWidth <= 0 || spanHeight <= 0) {
    return null;
  }

  const left = clampPercentage(((unit.position.x - originX + 0.5) / spanWidth) * 100);
  const top = clampPercentage(((unit.position.y - originY + 0.5) / spanHeight) * 100);
  return {
    left: `${left}%`,
    top: `${top}%`,
  };
}

function describeMapPreview(map: OpenRAMapData): string | null {
  const title = map.title;
  const source = map.previewSource;
  if (!title && !source) {
    return null;
  }
  if (title && source === "reference_map_preview") {
    return `Map preview · ${title}`;
  }
  return title ?? source;
}

export function OpenRAPlugin({
  session,
  scene,
  submitInput,
  mediaSubscribe,
}: ArenaPluginRenderProps<OpenRAActionIntent>) {
  const openRAScene = readScene(scene);
  const nativeSurfaceRef = useRef<HTMLDivElement | null>(null);
  const inputSequenceRef = useRef(0);
  const primaryMediaId = scene?.media?.primary?.mediaId;
  const mediaState = useMediaSource({
    sessionId: session.sessionId,
    mediaId: primaryMediaId ?? "",
    subscribe: mediaSubscribe,
  });
  const mediaSrc = typeof mediaState?.src === "string" ? mediaState.src : null;
  const lowLatencyStreamUrl = resolveLowLatencyStreamUrl(
    mediaState?.ref?.url,
    mediaState?.ref?.transport,
  );

  if (!openRAScene) {
    return (
      <section className="plugin-stage-card">
        <p className="eyebrow">OpenRA</p>
        <h2>RTS scene unavailable</h2>
        <p className="plugin-stage-card__copy">Waiting for structured RTS scene data.</p>
      </section>
    );
  }

  const resolvedActorId = resolveActorId(session, scene);
  const canSubmitActions = session.scheduling.acceptsHumanIntent && resolvedActorId !== null;
  const imageSrc = mediaSrc;
  const mapPreviewCaption = describeMapPreview(openRAScene.rts.map);
  const nativeControlsEnabled =
    canSubmitActions &&
    openRAScene.rts.legalActions.some((action) => action.id === "bridge_input");
  const isNativeRuntime = openRAScene.rts.map.previewSource === "native_runtime";
  const directControlActions = openRAScene.rts.legalActions.filter(isDirectControlAction);
  const hasEconomyStats = [
    openRAScene.rts.economy.credits,
    openRAScene.rts.economy.incomePerMinute,
    openRAScene.rts.economy.powerProduced,
    openRAScene.rts.economy.powerUsed,
  ].some((value) => value !== null);
  const hasSelectionSummary =
    openRAScene.rts.selection.primaryUnitId !== null ||
    openRAScene.rts.selection.unitIds.length > 0;
  const showEconomyCard = !isNativeRuntime || hasEconomyStats;
  const showSelectionCard = !isNativeRuntime || hasSelectionSummary;
  const showObjectivesCard = !isNativeRuntime || openRAScene.rts.objectives.length > 0;
  const showUnitsCard = !isNativeRuntime || openRAScene.rts.units.length > 0;
  const showProductionCard = !isNativeRuntime || openRAScene.rts.production.length > 0;
  const showSidePanel =
    showEconomyCard ||
    showSelectionCard ||
    showObjectivesCard ||
    showUnitsCard ||
    showProductionCard;
  const showActionsCard = directControlActions.length > 0 || !isNativeRuntime;
  const mapAspectRatio =
    openRAScene.rts.map.imageSize?.width !== null &&
    openRAScene.rts.map.imageSize?.width !== undefined &&
    openRAScene.rts.map.imageSize?.height !== null &&
    openRAScene.rts.map.imageSize?.height !== undefined &&
    openRAScene.rts.map.imageSize.height > 0
      ? `${openRAScene.rts.map.imageSize.width} / ${openRAScene.rts.map.imageSize.height}`
      : undefined;

  const submitBridgeInput = (payload: Record<string, unknown>) => {
    if (!nativeControlsEnabled || !resolvedActorId) {
      return;
    }
    inputSequenceRef.current += 1;
    void submitInput({
      playerId: resolvedActorId,
      actionPayload: {
        move: "bridge_input",
        payload,
        metadata: {
          input_seq: inputSequenceRef.current,
          realtime_input: true,
        },
      },
    });
  };

  return (
    <section
      className={[
        "openra-stage",
        isNativeRuntime ? "openra-stage--native-runtime" : "",
        showSidePanel ? "" : "openra-stage--focus-map",
      ]
        .filter((value) => value !== "")
        .join(" ")}
      data-testid="openra-stage"
    >
      <header className="openra-stage__header">
        <div>
          <p className="eyebrow">OpenRA RTS</p>
          <h2>{openRAScene.frame.title}</h2>
          {openRAScene.frame.subtitle ? (
            <p className="openra-stage__subtitle">{openRAScene.frame.subtitle}</p>
          ) : null}
        </div>
        <p className="openra-stage__actor">
          {resolvedActorId ? `Observer: ${resolvedActorId}` : "Observer: waiting"}
        </p>
      </header>

      <div className="openra-stage__grid">
        <section className="openra-stage__map-panel" aria-label="OpenRA map view">
          <div
            className="openra-stage__viewport"
            data-testid="openra-native-surface"
            onContextMenu={(event) => {
              if (!nativeControlsEnabled) {
                return;
              }
              event.preventDefault();
            }}
            onKeyDown={(event) => {
              if (!nativeControlsEnabled) {
                return;
              }
              submitBridgeInput({
                event_type: "key_down",
                key: event.key,
                modifiers: readModifierList(event),
              });
            }}
            onKeyUp={(event) => {
              if (!nativeControlsEnabled) {
                return;
              }
              submitBridgeInput({
                event_type: "key_up",
                key: event.key,
                modifiers: readModifierList(event),
              });
            }}
            onMouseDown={(event) => {
              if (!nativeControlsEnabled || !nativeSurfaceRef.current) {
                return;
              }
              const point = resolveNativeSurfacePoint(
                nativeSurfaceRef.current,
                openRAScene.frame,
                event.clientX,
                event.clientY,
              );
              if (!point) {
                return;
              }
              nativeSurfaceRef.current.focus();
              submitBridgeInput({
                event_type: "mouse_down",
                button: readPointerButton(event.button),
                buttons: readPointerButtonsMask(event.buttons || 1),
                x: point.x,
                y: point.y,
                viewport: point.viewport,
                modifiers: readModifierList(event),
              });
            }}
            onMouseMove={(event) => {
              if (!nativeControlsEnabled || !nativeSurfaceRef.current || event.buttons === 0) {
                return;
              }
              const point = resolveNativeSurfacePoint(
                nativeSurfaceRef.current,
                openRAScene.frame,
                event.clientX,
                event.clientY,
              );
              if (!point) {
                return;
              }
              submitBridgeInput({
                event_type: "mouse_move",
                button: readPointerButton(event.button),
                buttons: readPointerButtonsMask(event.buttons),
                x: point.x,
                y: point.y,
                viewport: point.viewport,
                modifiers: readModifierList(event),
              });
            }}
            onMouseUp={(event) => {
              if (!nativeControlsEnabled || !nativeSurfaceRef.current) {
                return;
              }
              const point = resolveNativeSurfacePoint(
                nativeSurfaceRef.current,
                openRAScene.frame,
                event.clientX,
                event.clientY,
              );
              if (!point) {
                return;
              }
              submitBridgeInput({
                event_type: "mouse_up",
                button: readPointerButton(event.button),
                buttons: readPointerButtonsMask(event.buttons),
                x: point.x,
                y: point.y,
                viewport: point.viewport,
                modifiers: readModifierList(event),
              });
            }}
            onWheel={(event) => {
              if (!nativeControlsEnabled || !nativeSurfaceRef.current) {
                return;
              }
              event.preventDefault();
              const point = resolveNativeSurfacePoint(
                nativeSurfaceRef.current,
                openRAScene.frame,
                event.clientX,
                event.clientY,
              );
              if (!point) {
                return;
              }
              submitBridgeInput({
                event_type: "mouse_scroll",
                button: "none",
                buttons: [],
                delta_x: Math.round(event.deltaX),
                delta_y: Math.round(event.deltaY),
                x: point.x,
                y: point.y,
                viewport: point.viewport,
                modifiers: readModifierList(event),
              });
            }}
            ref={nativeSurfaceRef}
            style={mapAspectRatio ? { aspectRatio: mapAspectRatio } : undefined}
            tabIndex={nativeControlsEnabled ? 0 : undefined}
          >
            <div className="openra-stage__map-layer">
              {lowLatencyStreamUrl ? (
                <LowLatencyFrameCanvas
                  altText={openRAScene.frame.altText}
                  className={[
                    "openra-stage__canvas",
                    isNativeRuntime ? "openra-stage__canvas--native" : "",
                  ]
                    .filter((value) => value !== "")
                    .join(" ")}
                  streamUrl={lowLatencyStreamUrl}
                  testId="openra-map-canvas"
                />
              ) : imageSrc ? (
                <img
                  alt={openRAScene.frame.altText}
                  className={[
                    "openra-stage__image",
                    isNativeRuntime ? "openra-stage__image--native" : "",
                  ]
                    .filter((value) => value !== "")
                    .join(" ")}
                  data-testid="openra-map-image"
                  draggable={false}
                  src={imageSrc}
                />
              ) : (
                <div className="openra-stage__fallback">Loading OpenRA frame...</div>
              )}
              <div className="openra-stage__map-markers" data-testid="openra-map-markers">
                {openRAScene.rts.units.map((unit) => {
                  const markerStyle = resolveUnitMarkerStyle(unit, openRAScene.rts.map);
                  if (!markerStyle) {
                    return null;
                  }
                  return (
                    <div
                      aria-label={`${unit.label} at ${formatPosition(unit.position)}`}
                      className={[
                        "openra-stage__map-marker",
                        unit.selected ? "is-selected" : "",
                        unit.owner === openRAScene.status.activePlayerId ? "is-friendly" : "is-hostile",
                      ]
                        .filter((value) => value !== "")
                        .join(" ")}
                      data-testid={`openra-map-marker-${unit.id}`}
                      key={unit.id}
                      style={markerStyle}
                      title={`${unit.label} ${formatPosition(unit.position)}`}
                    >
                      <span>{unit.label.slice(0, 1)}</span>
                    </div>
                  );
                })}
              </div>
            </div>
            {isNativeRuntime ? (
              <div className="openra-stage__native-guide" data-testid="openra-native-guide">
                <p className="eyebrow">
                  {nativeControlsEnabled ? "Native Control" : "Native Demo"}
                </p>
                <p className="openra-stage__native-guide-copy">
                  {nativeControlsEnabled
                    ? "Click the battlefield to focus input. Left click selects, right click commands, mouse wheel zooms, and arrow or WASD keys pan the camera."
                    : "This is the live native OpenRA engine feed. The dummy driver is issuing real mouse orders so units move and the battlefield changes in real time."}
                </p>
              </div>
            ) : null}
            {openRAScene.overlays.length > 0 ? (
              <div className="openra-stage__overlay-strip">
                {openRAScene.overlays.map((overlay) => (
                  <div
                    className="openra-stage__overlay-badge"
                    key={`${overlay.kind}:${overlay.label}:${overlay.value}`}
                  >
                    <span>{overlay.label}</span>
                    <strong>{overlay.value}</strong>
                  </div>
                ))}
              </div>
            ) : null}
            <div className="openra-stage__status-strip">
              <span>Tick {openRAScene.status.tick}</span>
              <span>Step {openRAScene.status.step}</span>
              <span>Moves {openRAScene.status.moveCount}</span>
              {openRAScene.status.lastMove ? <span>Last {openRAScene.status.lastMove}</span> : null}
            </div>
          </div>
          {mapPreviewCaption ? (
            <p className="openra-stage__view-text">{mapPreviewCaption}</p>
          ) : null}
          {openRAScene.viewText ? (
            <p className="openra-stage__view-text">{openRAScene.viewText}</p>
          ) : null}
          {nativeControlsEnabled ? (
            <p className="openra-stage__view-text" data-testid="openra-native-controls-hint">
              Direct controls: left click selects, right click commands, mouse wheel zooms,
              and arrow or WASD keys pan the camera.
            </p>
          ) : null}
        </section>

        {showSidePanel ? (
          <aside className="openra-stage__side-panel">
            {showEconomyCard ? (
              <section className="openra-stage__card" data-testid="openra-economy">
                <h3>Economy</h3>
                <dl className="openra-stage__definition-list">
                  <div>
                    <dt>Credits</dt>
                    <dd>{renderValue(openRAScene.rts.economy.credits)}</dd>
                  </div>
                  <div>
                    <dt>Income / min</dt>
                    <dd>{renderValue(openRAScene.rts.economy.incomePerMinute)}</dd>
                  </div>
                  <div>
                    <dt>Power produced</dt>
                    <dd>{renderValue(openRAScene.rts.economy.powerProduced)}</dd>
                  </div>
                  <div>
                    <dt>Power used</dt>
                    <dd>{renderValue(openRAScene.rts.economy.powerUsed)}</dd>
                  </div>
                </dl>
              </section>
            ) : null}

            {showSelectionCard ? (
              <section className="openra-stage__card" data-testid="openra-selection">
                <h3>Selection</h3>
                <p className="openra-stage__copy">
                  Primary unit: {openRAScene.rts.selection.primaryUnitId ?? "none"}
                </p>
                <p className="openra-stage__copy">
                  Units:{" "}
                  {openRAScene.rts.selection.unitIds.length > 0
                    ? openRAScene.rts.selection.unitIds.join(", ")
                    : "none"}
                </p>
              </section>
            ) : null}

            {showObjectivesCard ? (
              <section className="openra-stage__card" data-testid="openra-objectives">
                <h3>Objectives</h3>
                {openRAScene.rts.objectives.length > 0 ? (
                  <ul className="openra-stage__list">
                    {openRAScene.rts.objectives.map((objective) => (
                      <li key={objective.id}>
                        <strong>{objective.label}</strong>
                        <span>{objective.status}</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="openra-stage__copy">No active objectives.</p>
                )}
              </section>
            ) : null}

            {showUnitsCard ? (
              <section className="openra-stage__card" data-testid="openra-units">
                <h3>Units</h3>
                {openRAScene.rts.units.length > 0 ? (
                  <ul className="openra-stage__list openra-stage__list--dense">
                    {openRAScene.rts.units.map((unit) => (
                      <li key={unit.id} className="openra-stage__unit">
                        <div className="openra-stage__unit-header">
                          <strong>{unit.label}</strong>
                          {unit.selected ? <span>selected</span> : null}
                        </div>
                        <p className="openra-stage__copy">
                          {unit.owner ?? "unknown"} {unit.kind ? `· ${unit.kind}` : ""}
                        </p>
                        <p className="openra-stage__copy">
                          HP {renderValue(unit.hp)} · {unit.status ?? "unknown"} ·{" "}
                          {formatPosition(unit.position)}
                        </p>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="openra-stage__copy">No visible units.</p>
                )}
              </section>
            ) : null}

            {showProductionCard ? (
              <section className="openra-stage__card" data-testid="openra-production">
                <h3>Production</h3>
                {openRAScene.rts.production.length > 0 ? (
                  <ul className="openra-stage__list">
                    {openRAScene.rts.production.map((queue) => (
                      <li key={queue.buildingId}>
                        <strong>{queue.label}</strong>
                        {queue.items.length > 0 ? (
                          <ul className="openra-stage__nested-list">
                            {queue.items.map((item) => (
                              <li key={item.id}>
                                <span>{item.label}</span>
                                <span>{formatProgress(item.progress)}</span>
                              </li>
                            ))}
                          </ul>
                        ) : (
                          <p className="openra-stage__copy">Queue idle.</p>
                        )}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="openra-stage__copy">No queued production.</p>
                )}
              </section>
            ) : null}
          </aside>
        ) : null}
      </div>

      {showActionsCard ? (
        <section className="openra-stage__card" data-testid="openra-actions">
          <h3>Legal Actions</h3>
          {directControlActions.length > 0 ? (
            <div className="openra-stage__actions">
              {directControlActions.map((action) => (
                <button
                  key={action.id}
                  className="openra-stage__action-chip"
                  disabled={!canSubmitActions}
                  onClick={() => {
                    if (!canSubmitActions || !resolvedActorId) {
                      return;
                    }
                    void submitInput({
                      playerId: resolvedActorId,
                      actionPayload: {
                        move: action.id,
                        payload: action.payload,
                      },
                    });
                  }}
                  type="button"
                >
                  <span>{action.label}</span>
                  <small>{renderPayloadPreview(action.payload)}</small>
                </button>
              ))}
            </div>
          ) : nativeControlsEnabled ? (
            <p className="openra-stage__copy">
              Direct frame controls are active. Use the game viewport instead of action chips.
            </p>
          ) : (
            <p className="openra-stage__copy">No legal actions available.</p>
          )}
        </section>
      ) : null}
    </section>
  );
}

import { useEffect, useState } from "react";

import type { TableDiscardLane, TableMeldGroup, TableSceneData, TableSeat } from "../table/TableLayout";
import { MahjongActionTray } from "./MahjongActionTray";
import { resolveMahjongTileAsset } from "./mahjongTileAssets";
import {
  isMahjongTileCode,
  normalizeTileCode,
  resolveMahjongDiscardLanes,
  resolveMahjongMeldGroups,
  splitMahjongSeatHand,
  uniqueMahjongActionTexts,
} from "./mahjongScene";

import "./mahjong.css";

type SeatAnchor = "bottom" | "right" | "top" | "left";

interface MahjongTableProps {
  tableScene: TableSceneData;
  actionTexts: string[];
  canSubmitActions: boolean;
  onSubmitAction: (actionText: string) => void;
}

interface MahjongTileProps {
  tile: string;
  compact?: boolean;
  hidden?: boolean;
  emphasized?: boolean;
  selected?: boolean;
  disabled?: boolean;
  ariaLabel?: string;
  onClick?: () => void;
  onDoubleClick?: () => void;
}

const MAHJONG_CALL_ACTIONS = new Set(["pong", "chow", "kong", "gong", "hu"]);

function titleCase(value: string): string {
  return value
    .split(/[_\s]+/)
    .filter((part) => part !== "")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1).toLowerCase())
    .join(" ");
}

function formatResultSummary(tableScene: TableSceneData): { title: string; detail: string[] } | null {
  const { winner, result, resultReason, remainingTiles } = tableScene.status;
  if (!result) {
    return null;
  }
  const title = winner ? `${resolvePlayerLabel(tableScene, winner)} ${titleCase(result)}` : titleCase(result);
  const detail = [
    resultReason ? titleCase(resultReason) : null,
    remainingTiles !== null ? `${remainingTiles} tiles left in wall` : null,
  ].filter((entry): entry is string => entry !== null);
  return { title, detail };
}

function rotateSeatIds(seatIds: string[], bottomSeatId: string): string[] {
  const anchorIndex = seatIds.indexOf(bottomSeatId);
  if (anchorIndex < 0) {
    return seatIds;
  }
  return seatIds.slice(anchorIndex).concat(seatIds.slice(0, anchorIndex));
}

function resolveAnchoredSeats(tableScene: TableSceneData): Array<{ anchor: SeatAnchor; seat: TableSeat }> {
  const seatsById = new Map(tableScene.table.seats.map((seat) => [seat.seatId, seat]));
  const observerSeat =
    tableScene.table.seats.find((seat) => seat.isObserver) ??
    tableScene.table.seats.find((seat) => seat.hand.isVisible) ??
    tableScene.table.seats[0];
  const canonicalOrder = ["east", "south", "west", "north"];
  const knownSeatIds = canonicalOrder.filter((seatId) => seatsById.has(seatId));
  const orderedSeatIds =
    knownSeatIds.length === tableScene.table.seats.length
      ? knownSeatIds
      : tableScene.table.seats.map((seat) => seat.seatId);
  const rotatedSeatIds = rotateSeatIds(orderedSeatIds, observerSeat?.seatId ?? orderedSeatIds[0]);
  const anchors: SeatAnchor[] = ["bottom", "right", "top", "left"];

  return anchors
    .map((anchor, index) => {
      const seatId = rotatedSeatIds[index];
      const seat = seatId ? seatsById.get(seatId) ?? null : null;
      return seat ? { anchor, seat } : null;
    })
    .filter((entry): entry is { anchor: SeatAnchor; seat: TableSeat } => entry !== null);
}

function resolveLatestSeatChatMap(tableScene: TableSceneData): Map<string, string> {
  const latestByPlayerId = new Map<string, string>();
  tableScene.panels.chatLog.forEach((entry) => {
    latestByPlayerId.set(entry.playerId, entry.text);
  });
  return latestByPlayerId;
}

function resolveSeatByPlayerId(tableScene: TableSceneData, playerId: string | null): TableSeat | null {
  if (!playerId) {
    return null;
  }
  return tableScene.table.seats.find((seat) => seat.playerId === playerId) ?? null;
}

function resolvePlayerLabel(tableScene: TableSceneData, playerId: string | null): string {
  if (!playerId) {
    return "Unknown";
  }
  return resolveSeatByPlayerId(tableScene, playerId)?.playerName ?? titleCase(playerId);
}

function renderMaskedHand(maskedCount: number, compact: boolean) {
  const previewCount = Math.min(Math.max(maskedCount, 3), compact ? 5 : 7);
  return (
    <div className="mahjong-hand mahjong-hand--masked">
      <div className="mahjong-hand__rack">
        {Array.from({ length: previewCount }, (_, index) => (
          <MahjongTile key={`masked-${index}`} tile="Back" compact={compact} hidden={true} />
        ))}
      </div>
      <p className="mahjong-hand__caption">
        {maskedCount > 0 ? `Hidden hand · ${maskedCount} tiles` : "Hidden hand"}
      </p>
    </div>
  );
}

function MahjongTile({
  tile,
  compact = false,
  hidden = false,
  emphasized = false,
  selected = false,
  disabled = false,
  ariaLabel,
  onClick,
  onDoubleClick,
}: MahjongTileProps) {
  const resolvedTile = hidden ? "Back" : normalizeTileCode(tile);
  const src = resolveMahjongTileAsset(resolvedTile);
  const className = [
    "mahjong-tile",
    compact ? "is-compact" : "",
    hidden ? "is-hidden" : "",
    emphasized ? "is-emphasized" : "",
    selected ? "is-selected" : "",
    onClick ? "is-interactive" : "",
  ]
    .filter((value) => value !== "")
    .join(" ");
  const content = src ? <img src={src} alt={hidden ? "" : resolvedTile} /> : <span>{hidden ? "🀫" : resolvedTile}</span>;

  if (onClick) {
    return (
      <button
        type="button"
        className={className}
        aria-label={ariaLabel ?? `Select ${resolvedTile}`}
        aria-pressed={selected}
        disabled={disabled}
        onClick={onClick}
        onDoubleClick={onDoubleClick}
      >
        {content}
      </button>
    );
  }

  return (
    <div className={className} aria-hidden={hidden ? "true" : undefined}>
      {content}
      {!hidden ? <span className="mahjong-visually-hidden">{resolvedTile}</span> : null}
    </div>
  );
}

function renderMelds(groups: TableMeldGroup[], compact: boolean) {
  if (groups.length === 0) {
    return null;
  }
  return (
    <div className={`mahjong-melds ${compact ? "is-compact" : ""}`}>
      {groups.map((group, groupIndex) => (
        <div key={`${group.label}-${groupIndex}`} className="mahjong-meld">
          <div className="mahjong-meld__tiles">
            {group.tiles.length > 0 ? (
              group.tiles.map((tileCode, tileIndex) => (
                <MahjongTile
                  key={`${group.label}-${tileCode}-${tileIndex}`}
                  tile={tileCode}
                  compact={compact}
                />
              ))
            ) : (
              <span className="mahjong-meld__label">{group.label}</span>
            )}
          </div>
          <p className="mahjong-meld__caption">{group.label}</p>
        </div>
      ))}
    </div>
  );
}

function renderSeatHand({
  seat,
  compact,
  canSubmitActions,
  tileActionSet,
  selectedTile,
  onSelectTile,
  onConfirmTile,
}: {
  seat: TableSeat;
  compact: boolean;
  canSubmitActions: boolean;
  tileActionSet: Set<string>;
  selectedTile: string | null;
  onSelectTile: (tile: string) => void;
  onConfirmTile: (tile: string) => void;
}) {
  if (!seat.hand.isVisible) {
    return renderMaskedHand(seat.hand.maskedCount, compact);
  }

  const handState = splitMahjongSeatHand(seat);
  const interactive = canSubmitActions && !compact;

  return (
    <div className="mahjong-hand mahjong-hand--visible">
      <div className="mahjong-hand__rack">
        {handState.mainTiles.map((tileCode, index) => {
          const isLegal = interactive && tileActionSet.has(tileCode);
          return (
            <MahjongTile
              key={`${seat.playerId}-${tileCode}-${index}`}
              tile={tileCode}
              compact={compact}
              selected={selectedTile === tileCode}
              disabled={!isLegal}
              ariaLabel={`Select ${tileCode}`}
              onClick={
                isLegal
                  ? () => {
                      onSelectTile(tileCode);
                    }
                  : undefined
              }
              onDoubleClick={
                isLegal
                  ? () => {
                      onConfirmTile(tileCode);
                    }
                  : undefined
              }
            />
          );
        })}
        {handState.drawTile ? (
          <div className="mahjong-hand__draw-slot" data-testid="mahjong-draw-slot">
            <MahjongTile
              tile={handState.drawTile}
              compact={compact}
              emphasized={tileActionSet.has(handState.drawTile)}
              selected={selectedTile === handState.drawTile}
              disabled={!interactive || !tileActionSet.has(handState.drawTile)}
              ariaLabel={`Select ${handState.drawTile}`}
              onClick={
                interactive && tileActionSet.has(handState.drawTile)
                  ? () => {
                      onSelectTile(handState.drawTile ?? "");
                    }
                  : undefined
              }
              onDoubleClick={
                interactive && tileActionSet.has(handState.drawTile)
                  ? () => {
                      onConfirmTile(handState.drawTile ?? "");
                    }
                  : undefined
              }
            />
          </div>
        ) : null}
      </div>
    </div>
  );
}

function MahjongSeat({
  seat,
  anchor,
  canSubmitActions,
  tileActionSet,
  selectedTile,
  chatBubbleText,
  onSelectTile,
  onConfirmTile,
}: {
  seat: TableSeat;
  anchor: SeatAnchor;
  canSubmitActions: boolean;
  tileActionSet: Set<string>;
  selectedTile: string | null;
  chatBubbleText?: string;
  onSelectTile: (tile: string) => void;
  onConfirmTile: (tile: string) => void;
}) {
  const compact = anchor !== "bottom";
  const className = [
    "mahjong-seat",
    `mahjong-seat--${anchor}`,
    seat.isActive ? "is-active" : "",
    seat.isObserver ? "is-observer" : "",
  ]
    .filter((value) => value !== "")
    .join(" ");
  const meldGroups = resolveMahjongMeldGroups(seat);

  return (
    <article
      className={className}
      aria-label={`Mahjong seat ${seat.seatId}`}
      data-testid={`mahjong-seat-${anchor}`}
    >
      <header className="mahjong-seat__header">
        <div>
          <p className="mahjong-seat__name">{seat.playerName}</p>
          <p className="mahjong-seat__meta">{seat.seatId}</p>
        </div>
        <div className="mahjong-seat__chips">
          {seat.isObserver ? <span className="mahjong-seat__marker">you</span> : null}
          {seat.isActive ? <span className="mahjong-seat__badge">turn</span> : null}
        </div>
      </header>

      {chatBubbleText ? (
        <p className="mahjong-seat__bubble" data-testid={`mahjong-seat-${anchor}-bubble`}>
          {chatBubbleText}
        </p>
      ) : null}

      {renderMelds(meldGroups, compact)}

      <div className="mahjong-seat__hand" data-testid={`mahjong-seat-${anchor}-hand`}>
        {renderSeatHand({
          seat,
          compact,
          canSubmitActions: canSubmitActions && anchor === "bottom",
          tileActionSet,
          selectedTile: anchor === "bottom" ? selectedTile : null,
          onSelectTile,
          onConfirmTile,
        })}
      </div>
    </article>
  );
}

function resolveLastDiscardIndex(
  lane: TableDiscardLane,
  lastDiscard: TableSceneData["status"]["lastDiscard"],
): number {
  if (!lastDiscard?.tile || lastDiscard.playerId !== lane.playerId) {
    return -1;
  }
  const normalizedTile = normalizeTileCode(lastDiscard.tile);
  return lane.cards.reduce((resolvedIndex, tileCode, index) => {
    return normalizeTileCode(tileCode) === normalizedTile ? index : resolvedIndex;
  }, -1);
}

function MahjongDiscardLane({
  lane,
  lastDiscard,
}: {
  lane: TableDiscardLane;
  lastDiscard: TableSceneData["status"]["lastDiscard"];
}) {
  const emphasizedIndex = resolveLastDiscardIndex(lane, lastDiscard);

  return (
    <div className="mahjong-discard-lane" data-testid={`mahjong-discard-lane-${lane.playerId}`}>
      <p className="mahjong-discard-lane__label">{lane.seatId}</p>
      <div className="mahjong-discard-lane__tiles">
        {lane.cards.length > 0 ? (
          lane.cards.map((tileCode, index) => {
            const isLastDiscard = index === emphasizedIndex;
            const discardStyle =
              isLastDiscard && lastDiscard
                ? lastDiscard.isTsumogiri
                  ? "is-tsumogiri"
                  : "is-tedashi"
                : "";

            return (
              <div
                key={`${lane.playerId}-${tileCode}-${index}`}
                className={["mahjong-discard-lane__entry", discardStyle].filter(Boolean).join(" ")}
              >
                <MahjongTile tile={tileCode} compact={true} emphasized={isLastDiscard} />
                {isLastDiscard && lastDiscard ? (
                  <span
                    className="mahjong-discard-lane__marker"
                    aria-label={lastDiscard.isTsumogiri ? "Tsumogiri discard" : "Tedashi discard"}
                  >
                    {lastDiscard.isTsumogiri ? "Tsumo" : "Tedashi"}
                  </span>
                ) : null}
              </div>
            );
          })
        ) : (
          <span className="mahjong-discards__placeholder">No discards</span>
        )}
      </div>
    </div>
  );
}

function isMahjongCallAction(actionText: string): boolean {
  return MAHJONG_CALL_ACTIONS.has(actionText.trim().toLowerCase());
}

function MahjongCallPanel({
  callActions,
  canSubmitActions,
  onSubmitAction,
}: {
  callActions: string[];
  canSubmitActions: boolean;
  onSubmitAction: (actionText: string) => void;
}) {
  if (callActions.length === 0) {
    return null;
  }

  return (
    <div className="mahjong-call-panel" data-testid="mahjong-call-panel">
      <p className="mahjong-call-panel__eyebrow">Call options</p>
      <div className="mahjong-call-panel__actions">
        {callActions.map((actionText) => (
          <button
            key={actionText}
            type="button"
            className="mahjong-action mahjong-action--call"
            aria-label={`Play ${actionText}`}
            disabled={!canSubmitActions}
            onClick={() => {
              onSubmitAction(actionText);
            }}
          >
            <span className="mahjong-action__label">{actionText}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

function MahjongResultBanner({ tableScene }: { tableScene: TableSceneData }) {
  const summary = formatResultSummary(tableScene);
  if (!summary) {
    return null;
  }

  return (
    <section className="mahjong-result-banner" data-testid="mahjong-result-banner">
      <p className="mahjong-result-banner__eyebrow">Hand result</p>
      <h3 className="mahjong-result-banner__title">{summary.title}</h3>
      {summary.detail.length > 0 ? (
        <p className="mahjong-result-banner__detail">{summary.detail.join(" · ")}</p>
      ) : null}
    </section>
  );
}

export function MahjongTable({
  tableScene,
  actionTexts,
  canSubmitActions,
  onSubmitAction,
}: MahjongTableProps) {
  const [selectedTile, setSelectedTile] = useState<string | null>(null);
  const anchoredSeats = resolveAnchoredSeats(tableScene);
  const tileActionSet = new Set(actionTexts.filter(isMahjongTileCode).map(normalizeTileCode));
  const uniqueActions = uniqueMahjongActionTexts(actionTexts);
  const callActions = uniqueActions.filter(isMahjongCallAction);
  const trayActions = uniqueActions.filter(
    (actionText) => !isMahjongTileCode(actionText) && !isMahjongCallAction(actionText),
  );
  const latestSeatChats = resolveLatestSeatChatMap(tableScene);
  const lastDiscard = tableScene.status.lastDiscard;
  const discardLanes = resolveMahjongDiscardLanes(tableScene);
  const viewingLabel = resolvePlayerLabel(
    tableScene,
    tableScene.status.privateViewPlayerId ?? tableScene.status.observerPlayerId,
  );
  const turnLabel = tableScene.status.activePlayerId
    ? resolvePlayerLabel(tableScene, tableScene.status.activePlayerId)
    : null;
  const lastDiscardLabel =
    lastDiscard?.tile && lastDiscard.playerId
      ? `${resolvePlayerLabel(tableScene, lastDiscard.playerId)} ${normalizeTileCode(lastDiscard.tile)}`
      : null;
  const resultSummary = formatResultSummary(tableScene);

  useEffect(() => {
    if (!selectedTile) {
      return;
    }
    if (!canSubmitActions || !tileActionSet.has(selectedTile)) {
      setSelectedTile(null);
    }
  }, [canSubmitActions, selectedTile, tileActionSet]);

  const handleSelectTile = (tile: string) => {
    const normalized = normalizeTileCode(tile);
    setSelectedTile((current) => (current === normalized ? null : normalized));
  };

  const handleConfirmTile = (tile: string) => {
    const normalized = normalizeTileCode(tile);
    onSubmitAction(normalized);
    setSelectedTile(null);
  };

  return (
    <section className="mahjong-stage" data-testid="mahjong-stage">
      <div className="mahjong-stage__status" data-testid="mahjong-stage-status">
        {tableScene.status.moveCount > 0 ? (
          <span className="mahjong-stage__status-chip">Hand {tableScene.status.moveCount}</span>
        ) : null}
        {turnLabel ? (
          <span className="mahjong-stage__status-chip">Turn {turnLabel}</span>
        ) : null}
        {viewingLabel ? (
          <span className="mahjong-stage__status-chip">Viewing {viewingLabel}</span>
        ) : null}
        {lastDiscardLabel && !resultSummary ? (
          <span className="mahjong-stage__status-chip">Last discard {lastDiscardLabel}</span>
        ) : null}
        {tableScene.status.remainingTiles !== null ? (
          <span className="mahjong-stage__status-chip">Wall {tableScene.status.remainingTiles}</span>
        ) : null}
        {resultSummary ? (
          <span className="mahjong-stage__status-chip is-result">{resultSummary.title}</span>
        ) : null}
      </div>

      <div className="mahjong-stage__table">
        {anchoredSeats.map(({ anchor, seat }) => (
          <MahjongSeat
            key={`${anchor}-${seat.playerId}`}
            seat={seat}
            anchor={anchor}
            canSubmitActions={canSubmitActions}
            tileActionSet={tileActionSet}
            selectedTile={selectedTile}
            chatBubbleText={latestSeatChats.get(seat.playerId)}
            onSelectTile={handleSelectTile}
            onConfirmTile={handleConfirmTile}
          />
        ))}

        <section className="mahjong-discards">
          <div className="mahjong-discards__header">
            <div>
              <p className="mahjong-discards__label">{tableScene.table.center.label}</p>
              <p className="mahjong-discards__subcopy">
                {tableScene.table.center.history.length > 0
                  ? `${tableScene.table.center.history.length} public actions tracked`
                  : "Live table snapshot"}
              </p>
            </div>
            {lastDiscardLabel ? (
              <div className="mahjong-discards__spotlight">
                <span className="mahjong-discards__spotlight-label">Latest</span>
                <strong className="mahjong-discards__spotlight-value">{lastDiscardLabel}</strong>
              </div>
            ) : null}
          </div>

          <div className="mahjong-discards__pool" data-testid="mahjong-discard-pool">
            {tableScene.table.center.cards.length > 0 ? (
              tableScene.table.center.cards.map((tileCode, index) => (
                <MahjongTile
                  key={`discard-${tileCode}-${index}`}
                  tile={tileCode}
                  compact={true}
                  emphasized={index === tableScene.table.center.cards.length - 1}
                />
              ))
            ) : (
              <p className="mahjong-discards__placeholder">No discards yet</p>
            )}
          </div>

          {tableScene.table.center.history.length > 0 ? (
            <div className="mahjong-history" data-testid="mahjong-history">
              {tableScene.table.center.history.map((entry, index) => (
                <span key={`${entry}-${index}`} className="mahjong-history__chip">
                  {entry}
                </span>
              ))}
            </div>
          ) : null}

          <MahjongResultBanner tableScene={tableScene} />

          {discardLanes.length > 0 ? (
            <div className="mahjong-discard-lanes">
              {discardLanes.map((lane) => (
                <MahjongDiscardLane
                  key={`${lane.playerId}-${lane.seatId}`}
                  lane={lane}
                  lastDiscard={lastDiscard}
                />
              ))}
            </div>
          ) : null}

          <MahjongCallPanel
            callActions={callActions}
            canSubmitActions={canSubmitActions}
            onSubmitAction={onSubmitAction}
          />
        </section>
      </div>

      <MahjongActionTray
        selectedTile={selectedTile}
        trayActions={trayActions}
        canSubmitActions={canSubmitActions}
        onClearSelection={() => {
          setSelectedTile(null);
        }}
        onConfirmTile={handleConfirmTile}
        onSubmitAction={onSubmitAction}
      />
    </section>
  );
}

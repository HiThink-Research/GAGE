import type { TableSceneData, TableSeat } from "../table/TableLayout";

import "./mahjong.css";

const mahjongTileAssetModules = import.meta.glob(
  "../../../../rlcard-showdown/src/assets/mahjong/*.svg",
  {
    eager: true,
    import: "default",
  },
) as Record<string, string>;

const mahjongTileAssets = Object.fromEntries(
  Object.entries(mahjongTileAssetModules).map(([path, url]) => {
    const fileName = path.split("/").pop() ?? path;
    return [fileName.replace(".svg", ""), url];
  }),
);

type SeatAnchor = "bottom" | "right" | "top" | "left";

interface MahjongTableProps {
  tableScene: TableSceneData;
  actionTexts: string[];
  canSubmitActions: boolean;
  onSubmitAction: (actionText: string) => void;
}

interface ParsedMeld {
  label: string;
  tiles: string[];
}

interface MahjongTileProps {
  tile: string;
  compact?: boolean;
  hidden?: boolean;
  emphasized?: boolean;
  disabled?: boolean;
  ariaLabel?: string;
  onClick?: () => void;
}

interface SeatRenderState {
  mainTiles: string[];
  drawTile: string | null;
}

function isMahjongTileCode(value: string): boolean {
  const normalized = value.trim();
  return /^[BCD][1-9]$/i.test(normalized) || /^(East|South|West|North|Green|Red|White)$/i.test(normalized);
}

function normalizeTileCode(value: string): string {
  const trimmed = value.trim();
  if (/^[BCD][1-9]$/i.test(trimmed)) {
    return `${trimmed.charAt(0).toUpperCase()}${trimmed.slice(1)}`;
  }
  if (/^(East|South|West|North|Green|Red|White)$/i.test(trimmed)) {
    return trimmed.charAt(0).toUpperCase() + trimmed.slice(1).toLowerCase();
  }
  return trimmed;
}

function parseMeldNote(note: string): ParsedMeld {
  const tiles = Array.from(
    note.matchAll(/\b([BCD][1-9]|East|South|West|North|Green|Red|White)\b/gi),
    (match) => normalizeTileCode(match[1] ?? match[0]),
  );
  const prefix = note.trim().split(/\s+/)[0]?.toLowerCase() ?? "";

  if (tiles.length === 1 && (prefix === "pong" || prefix === "peng")) {
    return {
      label: note,
      tiles: [tiles[0], tiles[0], tiles[0]],
    };
  }
  if (tiles.length === 1 && (prefix === "kong" || prefix === "gang")) {
    return {
      label: note,
      tiles: [tiles[0], tiles[0], tiles[0], tiles[0]],
    };
  }
  if (tiles.length >= 3 && (prefix === "chow" || prefix === "chi")) {
    return {
      label: note,
      tiles: tiles.slice(0, 3),
    };
  }
  return {
    label: note,
    tiles,
  };
}

function splitSeatHand(cards: string[]): SeatRenderState {
  if (cards.length > 1 && cards.length % 3 === 2) {
    return {
      mainTiles: cards.slice(0, -1),
      drawTile: cards.at(-1) ?? null,
    };
  }
  return {
    mainTiles: cards,
    drawTile: null,
  };
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

function renderMaskedHand(maskedCount: number, compact: boolean) {
  const previewCount = Math.min(Math.max(maskedCount, 3), compact ? 5 : 7);
  return (
    <div className="mahjong-hand mahjong-hand--masked">
      <div className="mahjong-hand__rack">
        {Array.from({ length: previewCount }, (_, index) => (
          <MahjongTile
            key={`masked-${index}`}
            tile="Back"
            compact={compact}
            hidden={true}
          />
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
  disabled = false,
  ariaLabel,
  onClick,
}: MahjongTileProps) {
  const resolvedTile = hidden ? "Back" : normalizeTileCode(tile);
  const src = mahjongTileAssets[resolvedTile] ?? null;
  const className = [
    "mahjong-tile",
    compact ? "is-compact" : "",
    hidden ? "is-hidden" : "",
    emphasized ? "is-emphasized" : "",
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
        aria-label={ariaLabel ?? `Play ${resolvedTile}`}
        disabled={disabled}
        onClick={onClick}
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

function renderMelds(notes: string[], compact: boolean) {
  if (notes.length === 0) {
    return null;
  }
  return (
    <div className={`mahjong-melds ${compact ? "is-compact" : ""}`}>
      {notes.map((note) => {
        const meld = parseMeldNote(note);
        return (
          <div key={note} className="mahjong-meld">
            <div className="mahjong-meld__tiles">
              {meld.tiles.length > 0 ? (
                meld.tiles.map((tileCode, index) => (
                  <MahjongTile
                    key={`${note}-${tileCode}-${index}`}
                    tile={tileCode}
                    compact={compact}
                  />
                ))
              ) : (
                <span className="mahjong-meld__label">{note}</span>
              )}
            </div>
            <p className="mahjong-meld__caption">{meld.label}</p>
          </div>
        );
      })}
    </div>
  );
}

function renderVisibleHand(
  seat: TableSeat,
  compact: boolean,
  canSubmitActions: boolean,
  tileActionSet: Set<string>,
  onSubmitAction: (actionText: string) => void,
) {
  const handState = splitSeatHand(seat.hand.cards.map(normalizeTileCode));
  const interactive = canSubmitActions && !compact;

  return (
    <div className="mahjong-hand mahjong-hand--visible">
      <div className="mahjong-hand__rack">
        {handState.mainTiles.map((tileCode, index) => {
          const normalizedTile = normalizeTileCode(tileCode);
          const isLegal = interactive && tileActionSet.has(normalizedTile);
          return (
            <MahjongTile
              key={`${seat.playerId}-${normalizedTile}-${index}`}
              tile={normalizedTile}
              compact={compact}
              disabled={!isLegal}
              ariaLabel={`Play ${normalizedTile}`}
              onClick={
                isLegal
                  ? () => {
                      onSubmitAction(normalizedTile);
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
              emphasized={interactive && tileActionSet.has(normalizeTileCode(handState.drawTile))}
              disabled={!interactive || !tileActionSet.has(normalizeTileCode(handState.drawTile))}
              ariaLabel={`Play ${normalizeTileCode(handState.drawTile)}`}
              onClick={
                interactive && tileActionSet.has(normalizeTileCode(handState.drawTile))
                  ? () => {
                      onSubmitAction(normalizeTileCode(handState.drawTile ?? ""));
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

function renderSeatHand(
  seat: TableSeat,
  compact: boolean,
  canSubmitActions: boolean,
  tileActionSet: Set<string>,
  onSubmitAction: (actionText: string) => void,
) {
  if (!seat.hand.isVisible) {
    return renderMaskedHand(seat.hand.maskedCount, compact);
  }
  return renderVisibleHand(seat, compact, canSubmitActions, tileActionSet, onSubmitAction);
}

function MahjongSeat({
  seat,
  anchor,
  canSubmitActions,
  tileActionSet,
  onSubmitAction,
}: {
  seat: TableSeat;
  anchor: SeatAnchor;
  canSubmitActions: boolean;
  tileActionSet: Set<string>;
  onSubmitAction: (actionText: string) => void;
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

      {renderMelds(seat.publicNotes, compact)}

      <div
        className="mahjong-seat__hand"
        data-testid={`mahjong-seat-${anchor}-hand`}
      >
        {renderSeatHand(
          seat,
          compact,
          canSubmitActions && anchor === "bottom",
          tileActionSet,
          onSubmitAction,
        )}
      </div>
    </article>
  );
}

function uniqueActionTexts(actionTexts: string[]): string[] {
  const seen = new Set<string>();
  return actionTexts.filter((actionText) => {
    const normalized = normalizeTileCode(actionText);
    if (seen.has(normalized)) {
      return false;
    }
    seen.add(normalized);
    return true;
  });
}

export function MahjongTable({
  tableScene,
  actionTexts,
  canSubmitActions,
  onSubmitAction,
}: MahjongTableProps) {
  const anchoredSeats = resolveAnchoredSeats(tableScene);
  const bottomSeat = anchoredSeats.find((entry) => entry.anchor === "bottom")?.seat ?? null;
  const tileActionSet = new Set(
    actionTexts.filter(isMahjongTileCode).map(normalizeTileCode),
  );
  const bottomVisibleTiles = new Set(
    bottomSeat?.hand.isVisible ? bottomSeat.hand.cards.map(normalizeTileCode) : [],
  );
  const trayActions = uniqueActionTexts(actionTexts).filter((actionText) => {
    const normalized = normalizeTileCode(actionText);
    return !isMahjongTileCode(normalized) || !bottomVisibleTiles.has(normalized);
  });
  const lastDiscardIndex =
    tableScene.table.center.cards.length > 0 ? tableScene.table.center.cards.length - 1 : -1;

  return (
    <section className="mahjong-stage" data-testid="mahjong-stage">
      <div className="mahjong-stage__table">
        {anchoredSeats.map(({ anchor, seat }) => (
          <MahjongSeat
            key={`${anchor}-${seat.playerId}`}
            seat={seat}
            anchor={anchor}
            canSubmitActions={canSubmitActions}
            tileActionSet={tileActionSet}
            onSubmitAction={onSubmitAction}
          />
        ))}

        <section className="mahjong-discards">
          <p className="mahjong-discards__label">{tableScene.table.center.label}</p>
          <div className="mahjong-discards__pool" data-testid="mahjong-discard-pool">
            {tableScene.table.center.cards.length > 0 ? (
              tableScene.table.center.cards.map((tileCode, index) => (
                <MahjongTile
                  key={`discard-${tileCode}-${index}`}
                  tile={tileCode}
                  compact={true}
                  emphasized={index === lastDiscardIndex}
                />
              ))
            ) : (
              <p className="mahjong-discards__placeholder">No discards yet</p>
            )}
          </div>
        </section>
      </div>

      {trayActions.length > 0 ? (
        <div className="mahjong-actions">
          {trayActions.map((actionText) => (
            <button
              key={actionText}
              type="button"
              className="mahjong-action"
              aria-label={`Play ${actionText}`}
              disabled={!canSubmitActions}
              onClick={() => {
                onSubmitAction(normalizeTileCode(actionText));
              }}
            >
              {isMahjongTileCode(actionText) ? (
                <MahjongTile tile={actionText} compact={true} />
              ) : (
                <span className="mahjong-action__label">{actionText}</span>
              )}
            </button>
          ))}
        </div>
      ) : null}
    </section>
  );
}

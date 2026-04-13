import type { VisualScene, VisualSession } from "../../gateway/types";
import {
  isRecord,
  readBoolean,
  readNumber,
  readOptionalNumber,
  readString,
  readStringArray,
} from "../../lib/sceneReaders";
import { resolveMahjongTileAsset } from "../mahjong/mahjongTileAssets";

export interface TableHand {
  isVisible: boolean;
  cards: string[];
  maskedCount: number;
  drawTile?: string | null;
}

export interface TableMeldGroup {
  type: string | null;
  label: string;
  tiles: string[];
}

export interface TableDiscardLane {
  seatId: string;
  playerId: string;
  cards: string[];
}

export interface TableSeat {
  seatId: string;
  playerId: string;
  playerName: string;
  role: string | null;
  isActive: boolean;
  isObserver: boolean;
  playedCards: string[];
  publicNotes: string[];
  meldGroups: TableMeldGroup[];
  drawTile: string | null;
  hand: TableHand;
}

interface TableChatEntry {
  playerId: string;
  text: string;
}

interface TablePanelEvent {
  label: string;
  detail?: string;
}

export interface TableSceneData {
  table: {
    layout: string;
    seats: TableSeat[];
    center: {
      label: string;
      cards: string[];
      history: string[];
      discardLanes: TableDiscardLane[];
    };
  };
  status: {
    activePlayerId: string | null;
    observerPlayerId: string | null;
    privateViewPlayerId: string | null;
    moveCount: number;
    lastMove: string | null;
    lastDiscard: {
      playerId: string | null;
      tile: string | null;
      isTsumogiri: boolean;
    } | null;
    winner: string | null;
    result: string | null;
    resultReason: string | null;
    remainingTiles: number | null;
    landlordId: string | null;
  };
  panels: {
    chatLog: TableChatEntry[];
    events: TablePanelEvent[];
    trace: string[];
  };
}

export type TableVariant = "doudizhu" | "mahjong";

interface TableLayoutProps {
  variant: TableVariant;
  gameLabel: string;
  actorLabel: string;
  tableScene: TableSceneData;
  actionTexts: string[];
  canSubmitActions: boolean;
  onSubmitAction: (actionText: string) => void;
}

function readHand(value: unknown): TableHand {
  if (!isRecord(value)) {
    return {
      isVisible: false,
      cards: [],
      maskedCount: 0,
      drawTile: null,
    };
  }
  return {
    isVisible: readBoolean(value.isVisible),
    cards: readStringArray(value.cards),
    maskedCount: Math.max(0, Math.floor(readNumber(value.maskedCount))),
    drawTile: readString(value.drawTile),
  };
}

function readMeldGroups(value: unknown): TableMeldGroup[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .filter(isRecord)
    .map((group) => ({
      type: readString(group.type),
      label: readString(group.label) ?? "",
      tiles: readStringArray(group.tiles),
    }))
    .filter((group) => group.label !== "" || group.tiles.length > 0);
}

function readDiscardLanes(value: unknown): TableDiscardLane[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .filter(isRecord)
    .map((lane, index) => ({
      seatId: readString(lane.seatId) ?? `seat-${index + 1}`,
      playerId: readString(lane.playerId) ?? `player-${index + 1}`,
      cards: readStringArray(lane.cards),
    }));
}

function readLastDiscard(value: unknown): TableSceneData["status"]["lastDiscard"] {
  if (!isRecord(value)) {
    return null;
  }
  return {
    playerId: readString(value.playerId),
    tile: readString(value.tile),
    isTsumogiri: readBoolean(value.isTsumogiri),
  };
}

function readChatLog(value: unknown): TableChatEntry[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .filter(isRecord)
    .map((entry) => ({
      playerId: readString(entry.playerId) ?? "system",
      text: readString(entry.text) ?? "",
    }))
    .filter((entry) => entry.text !== "");
}

function readPanelEvents(value: unknown): TablePanelEvent[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .filter(isRecord)
    .map((entry) => ({
      label: readString(entry.label) ?? "Event",
      detail: readString(entry.detail) ?? undefined,
    }));
}

export function readTableScene(scene?: VisualScene): TableSceneData | null {
  if (!scene || scene.kind !== "table" || !isRecord(scene.body)) {
    return null;
  }

  const body = scene.body;
  const table = body.table;
  if (!isRecord(table) || !Array.isArray(table.seats) || !isRecord(table.center)) {
    return null;
  }

  const seats: TableSeat[] = table.seats
    .filter(isRecord)
    .map((seat, index) => ({
      seatId: readString(seat.seatId) ?? `seat-${index + 1}`,
      playerId: readString(seat.playerId) ?? `player-${index + 1}`,
      playerName: readString(seat.playerName) ?? `Player ${index + 1}`,
      role: readString(seat.role),
      isActive: readBoolean(seat.isActive),
      isObserver: readBoolean(seat.isObserver),
      playedCards: readStringArray(seat.playedCards),
      publicNotes: readStringArray(seat.publicNotes),
      meldGroups: readMeldGroups(seat.meldGroups),
      drawTile: readString(seat.drawTile),
      hand: readHand(seat.hand),
    }));

  const center = table.center;
  const status = isRecord(body.status) ? body.status : {};
  const panels = isRecord(body.panels) ? body.panels : {};

  return {
    table: {
      layout: readString(table.layout) ?? "table",
      seats,
      center: {
        label: readString(center.label) ?? "Center",
        cards: readStringArray(center.cards),
        history: readStringArray(center.history),
        discardLanes: readDiscardLanes(center.discardLanes),
      },
    },
    status: {
      activePlayerId: readString(status.activePlayerId),
      observerPlayerId: readString(status.observerPlayerId),
      privateViewPlayerId: readString(status.privateViewPlayerId),
      moveCount: readNumber(status.moveCount),
      lastMove: readString(status.lastMove),
      lastDiscard: readLastDiscard(status.lastDiscard),
      winner: readString(status.winner),
      result: readString(status.result),
      resultReason: readString(status.resultReason),
      remainingTiles: readOptionalNumber(status.remainingTiles),
      landlordId: readString(status.landlordId),
    },
    panels: {
      chatLog: readChatLog(panels.chatLog),
      events: readPanelEvents(panels.events),
      trace: readStringArray(panels.trace),
    },
  };
}

export function readTableActionTexts(scene?: VisualScene): string[] {
  if (!scene) {
    return [];
  }

  return scene.legalActions
    .filter(isRecord)
    .map((action) => readString(action.text) ?? readString(action.id) ?? readString(action.label))
    .filter((action): action is string => action !== null);
}

export function resolveTableActorId(
  session: VisualSession,
  scene: VisualScene | undefined,
  tableScene: TableSceneData,
): string | null {
  if (
    session.observer.observerKind === "player" &&
    typeof session.observer.observerId === "string" &&
    session.observer.observerId.trim() !== ""
  ) {
    return session.observer.observerId;
  }

  return (
    tableScene.status.activePlayerId ??
    (typeof scene?.activePlayerId === "string" ? scene.activePlayerId : null) ??
    session.scheduling.activeActorId ??
    null
  );
}

export function formatTableActorLabel(
  session: VisualSession,
  resolvedActorId: string | null,
): string {
  if (
    session.observer.observerKind === "player" &&
    typeof session.observer.observerId === "string" &&
    session.observer.observerId.trim() !== ""
  ) {
    return `Observer: ${session.observer.observerId}`;
  }

  if (resolvedActorId) {
    return `Active player: ${resolvedActorId}`;
  }

  return "Active player: waiting";
}

function resolveSeatAnchor(layout: string, seatId: string, index: number): "top" | "right" | "bottom" | "left" {
  const normalizedSeatId = seatId.trim().toLowerCase();
  if (layout === "three-seat") {
    if (normalizedSeatId === "bottom") {
      return "bottom";
    }
    if (normalizedSeatId === "left") {
      return "left";
    }
    if (normalizedSeatId === "right") {
      return "right";
    }
    return ["bottom", "left", "right"][index % 3] as "bottom" | "left" | "right";
  }

  if (normalizedSeatId === "south" || normalizedSeatId === "bottom") {
    return "bottom";
  }
  if (normalizedSeatId === "west" || normalizedSeatId === "left") {
    return "left";
  }
  if (normalizedSeatId === "north" || normalizedSeatId === "top") {
    return "top";
  }
  if (normalizedSeatId === "east" || normalizedSeatId === "right") {
    return "right";
  }
  return ["bottom", "right", "top", "left"][index % 4] as "bottom" | "right" | "top" | "left";
}

function isMahjongTileCode(value: string): boolean {
  const normalized = value.trim();
  return /^[BCD][1-9]$/i.test(normalized) || /^(East|South|West|North|Green|Red|White)$/i.test(normalized);
}

function isDoudizhuJoker(card: string): boolean {
  return card === "BJ" || card === "RJ" || card === "B" || card === "R";
}

function isDoudizhuRed(card: string): boolean {
  return card.startsWith("H") || card.startsWith("D") || card === "RJ" || card === "R";
}

function resolveDoudizhuCardLabel(card: string): string {
  if (card === "BJ") {
    return "Black Joker";
  }
  if (card === "RJ") {
    return "Red Joker";
  }
  return card;
}

function resolveDoudizhuRank(card: string): string {
  if (card.length === 1) {
    return card.toUpperCase();
  }
  if (isDoudizhuJoker(card)) {
    return card.startsWith("R") ? "RJ" : "BJ";
  }
  return card.slice(-1).toUpperCase();
}

function resolveDoudizhuSuit(card: string): string {
  if (card.length === 1 || isDoudizhuJoker(card)) {
    return "";
  }
  const suit = card.charAt(0).toUpperCase();
  if (suit === "H") {
    return "♥";
  }
  if (suit === "D") {
    return "♦";
  }
  if (suit === "S") {
    return "♠";
  }
  if (suit === "C") {
    return "♣";
  }
  return "";
}

function DoudizhuCard({
  card,
  faceDown = false,
}: {
  card: string;
  faceDown?: boolean;
}) {
  if (faceDown) {
    return <div className="table-card table-card--back" aria-hidden="true" />;
  }

  const rank = resolveDoudizhuRank(card);
  const suit = resolveDoudizhuSuit(card);
  const className = [
    "table-card",
    isDoudizhuJoker(card) ? "table-card--joker" : "",
    isDoudizhuRed(card) ? "table-card--red" : "table-card--black",
  ]
    .filter((name) => name !== "")
    .join(" ");

  return (
    <div className={className} aria-label={`Doudizhu card ${resolveDoudizhuCardLabel(card)}`}>
      <span className="table-card__rank">{rank}</span>
      <span className="table-card__suit">{suit || "•"}</span>
    </div>
  );
}

function MahjongTile({
  tile,
  hidden = false,
}: {
  tile: string;
  hidden?: boolean;
}) {
  const assetKey = hidden ? "Back" : tile;
  const src = resolveMahjongTileAsset(assetKey);

  return (
    <div
      className={`table-tile ${hidden ? "table-tile--back" : ""}`}
      aria-hidden={hidden ? "true" : undefined}
    >
      {src ? (
        <>
          <img src={src} alt={tile} />
          <span className="table-visually-hidden">{tile}</span>
        </>
      ) : (
        <span>{hidden ? "🀫" : tile}</span>
      )}
    </div>
  );
}

function renderMaskedStack(
  variant: TableVariant,
  maskedCount: number,
) {
  const previewCount = Math.min(Math.max(maskedCount, 1), 3);
  return (
    <div className="table-stack table-stack--masked">
      {Array.from({ length: previewCount }, (_, index) =>
        variant === "doudizhu" ? (
          <DoudizhuCard key={`back-${index}`} card="Back" faceDown={true} />
        ) : (
          <MahjongTile key={`back-${index}`} tile="Back" hidden={true} />
        ),
      )}
      <p className="table-stack__caption">
        {maskedCount > 0 ? `Hidden hand · ${maskedCount} cards` : "Hidden hand"}
      </p>
    </div>
  );
}

function renderVisibleStack(
  variant: TableVariant,
  cards: string[],
) {
  if (cards.length === 0) {
    return <p className="table-stack__caption">No cards</p>;
  }

  return (
    <div className={`table-stack table-stack--${variant}`}>
      {cards.map((card, index) =>
        variant === "doudizhu" ? (
          <DoudizhuCard key={`${card}-${index}`} card={card} />
        ) : (
          <MahjongTile key={`${card}-${index}`} tile={card} />
        ),
      )}
    </div>
  );
}

function renderSeatHand(seat: TableSeat, variant: TableVariant) {
  if (!seat.hand.isVisible) {
    return renderMaskedStack(variant, seat.hand.maskedCount);
  }
  return renderVisibleStack(variant, seat.hand.cards);
}

function renderSeatPlayed(seat: TableSeat, variant: TableVariant) {
  if (seat.playedCards.length === 0) {
    return <p className="table-seat__placeholder">No public cards</p>;
  }
  return renderVisibleStack(variant, seat.playedCards);
}

function renderCenterCards(cards: string[], variant: TableVariant) {
  if (cards.length === 0) {
    return <p className="table-seat__placeholder">No center cards</p>;
  }
  return renderVisibleStack(variant, cards);
}

function renderActionVisual(actionText: string, variant: TableVariant) {
  if (variant === "mahjong" && isMahjongTileCode(actionText)) {
    return <MahjongTile tile={actionText} />;
  }
  if (variant === "doudizhu" && actionText.toLowerCase() !== "pass") {
    return <DoudizhuCard card={actionText} />;
  }
  return <span className="table-layout__action-text">{actionText}</span>;
}

export function TableLayout({
  variant,
  gameLabel,
  actorLabel,
  tableScene,
  actionTexts,
  canSubmitActions,
  onSubmitAction,
}: TableLayoutProps) {
  const surfaceClassName = [
    "table-layout-surface",
    `table-layout-surface--${variant}`,
    `table-layout-surface--${tableScene.table.layout}`,
  ]
    .filter((name) => name !== "")
    .join(" ");
  const boardClassName = [
    "table-layout__board",
    `table-layout__board--${variant}`,
    `table-layout__board--${tableScene.table.layout}`,
  ]
    .filter((name) => name !== "")
    .join(" ");

  return (
    <section className={surfaceClassName}>
      <div className="table-layout__header">
        <div>
          <p className="eyebrow">Table</p>
          <h2 className="table-layout__title">{gameLabel} table</h2>
        </div>
        <p className="table-layout__actor-label">{actorLabel}</p>
      </div>

      <div className="table-layout__layout">
        <div className={boardClassName}>
          {tableScene.table.seats.map((seat, index) => {
            const anchor = resolveSeatAnchor(tableScene.table.layout, seat.seatId, index);
            const seatClassName = [
              "table-seat",
              `table-seat--${variant}`,
              `table-seat--${anchor}`,
              seat.isActive ? "table-seat--active" : "",
              seat.isObserver ? "table-seat--observer" : "",
            ]
              .filter((name) => name !== "")
              .join(" ");

            return (
              <article
                key={seat.playerId}
                className={seatClassName}
                aria-label={`Table seat ${seat.seatId}`}
              >
                <header className="table-seat__header">
                  <div>
                    <p className="table-seat__name">{seat.playerName}</p>
                    <p className="table-seat__meta">{seat.seatId}</p>
                  </div>
                  {seat.role ? <span className="table-seat__badge">{seat.role}</span> : null}
                </header>

                <div className="table-seat__zone" data-testid={`seat-${seat.playerId}-hand`}>
                  {renderSeatHand(seat, variant)}
                </div>

                <div className="table-seat__zone table-seat__zone--played">
                  {renderSeatPlayed(seat, variant)}
                </div>

                <div className="table-seat__notes" data-testid={`seat-${seat.playerId}-notes`}>
                  {seat.publicNotes.length > 0 ? (
                    <div className="table-seat__note-list">
                      {seat.publicNotes.map((note) => (
                        <span key={note} className="table-seat__note-chip">
                          {note}
                        </span>
                      ))}
                    </div>
                  ) : (
                    "No notes"
                  )}
                </div>
              </article>
            );
          })}

          <section className={`table-layout__center table-layout__center--${variant}`}>
            <p className="table-layout__center-label">{tableScene.table.center.label}</p>
            <div className="table-layout__center-cards" data-testid="table-center-cards">
              {renderCenterCards(tableScene.table.center.cards, variant)}
            </div>
            <div className="table-layout__center-meta">
              {tableScene.status.landlordId ? (
                <span className="table-layout__pill">Landlord: {tableScene.status.landlordId}</span>
              ) : null}
              {tableScene.status.lastMove ? (
                <span className="table-layout__pill">Last move: {tableScene.status.lastMove}</span>
              ) : null}
            </div>
          </section>
        </div>

        <aside className="table-layout__sidebar">
          <section className="table-layout__summary">
            <p className="eyebrow">Snapshot</p>
            <dl className="table-layout__summary-list">
              <div>
                <dt>Move count</dt>
                <dd>{tableScene.status.moveCount}</dd>
              </div>
              <div>
                <dt>Active player</dt>
                <dd>{tableScene.status.activePlayerId ?? "Waiting"}</dd>
              </div>
              <div>
                <dt>Observer</dt>
                <dd>{tableScene.status.observerPlayerId ?? "Spectator"}</dd>
              </div>
            </dl>
          </section>

          <section className="table-layout__panel">
            <p className="table-layout__panel-title">History</p>
            <div className="table-layout__panel-body">
              {tableScene.table.center.history.length > 0 ? (
                tableScene.table.center.history.map((entry) => (
                  <p key={entry}>{entry}</p>
                ))
              ) : (
                <p>No history yet</p>
              )}
            </div>
          </section>

          <section className="table-layout__panel">
            <p className="table-layout__panel-title">Table talk</p>
            <div className="table-layout__panel-body">
              {tableScene.panels.chatLog.length > 0 ? (
                tableScene.panels.chatLog.map((entry, index) => (
                  <p key={`${entry.playerId}-${index}`}>
                    <strong>{entry.playerId}</strong>: {entry.text}
                  </p>
                ))
              ) : (
                <p>No table talk</p>
              )}
            </div>
          </section>
        </aside>
      </div>

      <div className="table-layout__actions">
        {actionTexts.map((actionText) => (
          <button
            key={actionText}
            type="button"
            className={`table-layout__action-chip table-layout__action-chip--${variant}`}
            aria-label={`Play ${actionText}`}
            disabled={!canSubmitActions}
            onClick={() => {
              onSubmitAction(actionText);
            }}
          >
            {renderActionVisual(actionText, variant)}
          </button>
        ))}
      </div>
    </section>
  );
}

import type { VisualScene, VisualSession } from "../../gateway/types";

export interface TableHand {
  isVisible: boolean;
  cards: string[];
  maskedCount: number;
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
  hand: TableHand;
}

export interface TableSceneData {
  table: {
    layout: string;
    seats: TableSeat[];
    center: {
      label: string;
      cards: string[];
      history: string[];
    };
  };
  status: {
    activePlayerId: string | null;
    observerPlayerId: string | null;
    moveCount: number;
    lastMove: string | null;
    landlordId: string | null;
  };
}

interface TableLayoutProps {
  gameLabel: string;
  actorLabel: string;
  tableScene: TableSceneData;
  actionTexts: string[];
  canSubmitActions: boolean;
  onSubmitAction: (actionText: string) => void;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function readString(value: unknown): string | null {
  return typeof value === "string" && value.trim() !== "" ? value : null;
}

function readBoolean(value: unknown): boolean {
  return value === true;
}

function readNumber(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function readStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .map(readString)
    .filter((item): item is string => item !== null);
}

function readHand(value: unknown): TableHand {
  if (!isRecord(value)) {
    return {
      isVisible: false,
      cards: [],
      maskedCount: 0,
    };
  }
  return {
    isVisible: readBoolean(value.isVisible),
    cards: readStringArray(value.cards),
    maskedCount: Math.max(0, Math.floor(readNumber(value.maskedCount))),
  };
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
      hand: readHand(seat.hand),
    }));

  const center = table.center;
  const status = isRecord(body.status) ? body.status : {};

  return {
    table: {
      layout: readString(table.layout) ?? "table",
      seats,
      center: {
        label: readString(center.label) ?? "Center",
        cards: readStringArray(center.cards),
        history: readStringArray(center.history),
      },
    },
    status: {
      activePlayerId: readString(status.activePlayerId),
      observerPlayerId: readString(status.observerPlayerId),
      moveCount: readNumber(status.moveCount),
      lastMove: readString(status.lastMove),
      landlordId: readString(status.landlordId),
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

function renderHandSummary(hand: TableHand): string {
  if (hand.isVisible && hand.cards.length > 0) {
    return hand.cards.join(" ");
  }
  if (!hand.isVisible && hand.maskedCount > 0) {
    return `Hidden hand · ${hand.maskedCount} cards`;
  }
  if (!hand.isVisible) {
    return "Hidden hand";
  }
  return "No cards";
}

export function TableLayout({
  gameLabel,
  actorLabel,
  tableScene,
  actionTexts,
  canSubmitActions,
  onSubmitAction,
}: TableLayoutProps) {
  const tableClassName = [
    "table-layout-surface",
    `table-layout-surface--${tableScene.table.layout}`,
  ].join(" ");

  return (
    <section className={tableClassName}>
      <div className="table-layout__header">
        <p className="eyebrow">Table</p>
        <p className="table-layout__actor-label">{actorLabel}</p>
      </div>
      <h2 className="table-layout__title">{gameLabel} table</h2>

      <div className="table-layout__grid">
        {tableScene.table.seats.map((seat) => {
          const seatClassName = [
            "table-seat",
            seat.isActive ? "table-seat--active" : "",
            seat.isObserver ? "table-seat--observer" : "",
          ]
            .filter((name) => name !== "")
            .join(" ");

          return (
            <article key={seat.playerId} className={seatClassName}>
              <header className="table-seat__header">
                <div>
                  <p className="table-seat__name">{seat.playerName}</p>
                  <p className="table-seat__meta">{seat.seatId}</p>
                </div>
                {seat.role ? <span className="table-seat__badge">{seat.role}</span> : null}
              </header>

              <div
                className="table-seat__hand"
                data-testid={`seat-${seat.playerId}-hand`}
              >
                {renderHandSummary(seat.hand)}
              </div>

              <div className="table-seat__played">
                {seat.playedCards.length > 0 ? seat.playedCards.join(" ") : "No public cards"}
              </div>

              <div
                className="table-seat__notes"
                data-testid={`seat-${seat.playerId}-notes`}
              >
                {seat.publicNotes.length > 0 ? seat.publicNotes.join(" · ") : "No notes"}
              </div>
            </article>
          );
        })}
      </div>

      <section className="table-layout__center">
        <p className="table-layout__center-label">{tableScene.table.center.label}</p>
        <div className="table-layout__center-cards" data-testid="table-center-cards">
          {tableScene.table.center.cards.length > 0
            ? tableScene.table.center.cards.join(" ")
            : "No center cards"}
        </div>
        {tableScene.table.center.history.length > 0 ? (
          <div className="table-layout__history">
            {tableScene.table.center.history.join(" · ")}
          </div>
        ) : null}
      </section>

      <div className="table-layout__actions">
        {actionTexts.map((actionText) => (
          <button
            key={actionText}
            type="button"
            className="table-layout__action-chip"
            aria-label={`Play ${actionText}`}
            disabled={!canSubmitActions}
            onClick={() => {
              onSubmitAction(actionText);
            }}
          >
            {actionText}
          </button>
        ))}
      </div>
    </section>
  );
}

import type { TableSceneData, TableSeat } from "../table/TableLayout";

import "./doudizhu.css";

interface DoudizhuTableProps {
  tableScene: TableSceneData;
  actionTexts: string[];
  canSubmitActions: boolean;
  onSubmitAction: (actionText: string) => void;
}

function resolveSeatById(tableScene: TableSceneData, seatId: string): TableSeat | null {
  return tableScene.table.seats.find((seat) => seat.seatId === seatId) ?? null;
}

function isDoudizhuJoker(card: string): boolean {
  return (
    card === "BJ" ||
    card === "RJ" ||
    card === "B" ||
    card === "R" ||
    card === "BlackJoker" ||
    card === "RedJoker"
  );
}

function isDoudizhuRed(card: string): boolean {
  return (
    card.startsWith("H") ||
    card.startsWith("D") ||
    card === "RJ" ||
    card === "R" ||
    card === "RedJoker"
  );
}

function resolveDoudizhuCardLabel(card: string): string {
  if (card === "BJ" || card === "B" || card === "BlackJoker") {
    return "BlackJoker";
  }
  if (card === "RJ" || card === "R" || card === "RedJoker") {
    return "RedJoker";
  }
  return card;
}

function resolveDoudizhuRank(card: string): string {
  if (card.length === 1) {
    return card.toUpperCase();
  }
  if (isDoudizhuJoker(card)) {
    return resolveDoudizhuCardLabel(card) === "RedJoker" ? "RJ" : "BJ";
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
  compact = false,
}: {
  card: string;
  faceDown?: boolean;
  compact?: boolean;
}) {
  if (faceDown) {
    return <div className={`doudizhu-card doudizhu-card--back ${compact ? "is-compact" : ""}`} aria-hidden="true" />;
  }

  const className = [
    "doudizhu-card",
    compact ? "is-compact" : "",
    isDoudizhuJoker(card) ? "doudizhu-card--joker" : "",
    isDoudizhuRed(card) ? "doudizhu-card--red" : "doudizhu-card--black",
  ]
    .filter((value) => value !== "")
    .join(" ");

  return (
    <div className={className} aria-label={`Doudizhu card ${resolveDoudizhuCardLabel(card)}`}>
      <span className="doudizhu-card__rank">{resolveDoudizhuRank(card)}</span>
      <span className="doudizhu-card__suit">{resolveDoudizhuSuit(card) || "•"}</span>
    </div>
  );
}

function renderMaskedHand(maskedCount: number, compact: boolean) {
  const previewCount = Math.min(Math.max(maskedCount, 1), compact ? 6 : 8);
  return (
    <div className="doudizhu-hand doudizhu-hand--masked">
      <div className="doudizhu-hand__cards">
        {Array.from({ length: previewCount }, (_, index) => (
          <DoudizhuCard key={`masked-${index}`} card="Back" faceDown={true} compact={compact} />
        ))}
      </div>
      <p className="doudizhu-hand__caption">
        {maskedCount > 0 ? `Hidden hand · ${maskedCount} cards` : "Hidden hand"}
      </p>
    </div>
  );
}

function renderVisibleCards(cards: string[], compact: boolean) {
  if (cards.length === 0) {
    return <p className="doudizhu-hand__caption">No cards</p>;
  }
  return (
    <div className={`doudizhu-hand__cards ${compact ? "is-compact" : ""}`}>
      {cards.map((card, index) => (
        <DoudizhuCard key={`${card}-${index}`} card={card} compact={compact} />
      ))}
    </div>
  );
}

function renderSeatHand(seat: TableSeat, compact: boolean) {
  if (!seat.hand.isVisible) {
    return renderMaskedHand(seat.hand.maskedCount, compact);
  }
  return (
    <div className="doudizhu-hand">
      {renderVisibleCards(seat.hand.cards, compact)}
    </div>
  );
}

function renderSeatPlayed(seat: TableSeat) {
  if (seat.playedCards.length === 0) {
    return <p className="doudizhu-seat__placeholder">No public cards</p>;
  }
  return (
    <div className="doudizhu-played">
      {seat.playedCards.map((card, index) => (
        <DoudizhuCard key={`${seat.playerId}-${card}-${index}`} card={card} compact={true} />
      ))}
    </div>
  );
}

function renderActionVisual(actionText: string) {
  if (actionText.toLowerCase() === "pass") {
    return <span className="doudizhu-action__label">pass</span>;
  }
  return <DoudizhuCard card={actionText} compact={true} />;
}

function DoudizhuSeat({
  seat,
  anchor,
}: {
  seat: TableSeat;
  anchor: "bottom" | "left" | "right";
}) {
  const compact = anchor !== "bottom";
  const className = [
    "doudizhu-seat",
    `doudizhu-seat--${anchor}`,
    seat.isActive ? "is-active" : "",
    seat.isObserver ? "is-observer" : "",
  ]
    .filter((value) => value !== "")
    .join(" ");

  return (
    <article
      className={className}
      aria-label={`Doudizhu seat ${anchor}`}
      data-testid={`doudizhu-seat-${anchor}`}
    >
      <header className="doudizhu-seat__header">
        <div>
          <p className="doudizhu-seat__name">{seat.playerName}</p>
          <p className="doudizhu-seat__meta">{seat.playerId}</p>
        </div>
        <div className="doudizhu-seat__chips">
          {seat.role ? <span className="doudizhu-seat__badge">{seat.role}</span> : null}
          {seat.isObserver ? <span className="doudizhu-seat__marker">you</span> : null}
        </div>
      </header>

      <div className="doudizhu-seat__played">{renderSeatPlayed(seat)}</div>

      <div data-testid={`doudizhu-seat-${anchor}-hand`} className="doudizhu-seat__hand">
        {renderSeatHand(seat, compact)}
      </div>
    </article>
  );
}

export function DoudizhuTable({
  tableScene,
  actionTexts,
  canSubmitActions,
  onSubmitAction,
}: DoudizhuTableProps) {
  const bottomSeat = resolveSeatById(tableScene, "bottom");
  const leftSeat = resolveSeatById(tableScene, "left");
  const rightSeat = resolveSeatById(tableScene, "right");

  if (!bottomSeat || !leftSeat || !rightSeat) {
    return null;
  }

  return (
    <section className="doudizhu-stage" data-testid="doudizhu-stage">
      <div className="doudizhu-stage__table">
        <DoudizhuSeat seat={leftSeat} anchor="left" />
        <DoudizhuSeat seat={rightSeat} anchor="right" />
        <DoudizhuSeat seat={bottomSeat} anchor="bottom" />

        <section className="doudizhu-center">
          <p className="doudizhu-center__eyebrow">{tableScene.table.center.label}</p>
          <div className="doudizhu-center__cards" data-testid="doudizhu-center-cards">
            {tableScene.table.center.cards.length > 0 ? (
              tableScene.table.center.cards.map((card, index) => (
                <DoudizhuCard key={`center-${card}-${index}`} card={card} compact={true} />
              ))
            ) : (
              <p className="doudizhu-seat__placeholder">No center cards</p>
            )}
          </div>
        </section>
      </div>

      <div className="doudizhu-actions">
        {actionTexts.map((actionText) => (
          <button
            key={actionText}
            type="button"
            className="doudizhu-action"
            aria-label={`Play ${actionText}`}
            disabled={!canSubmitActions}
            onClick={() => {
              onSubmitAction(actionText);
            }}
          >
            {renderActionVisual(actionText)}
          </button>
        ))}
      </div>
    </section>
  );
}

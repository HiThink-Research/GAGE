import type { ReactNode } from "react";

import type { TableSceneData, TableSeat } from "../table/TableLayout";
import { DoudizhuActionComposer } from "./DoudizhuActionComposer";
import { DoudizhuCardVisual } from "./doudizhuCards";

import "./doudizhu.css";

const portraitAssetModules = import.meta.glob(
  "../../../../rlcard-showdown/src/assets/images/Portrait/*.{png,jpg,jpeg}",
  {
    eager: true,
    import: "default",
  },
) as Record<string, string>;

const portraitAssets = Object.fromEntries(
  Object.entries(portraitAssetModules).map(([path, url]) => {
    const fileName = path.split("/").pop() ?? path;
    return [fileName.replace(/\.(png|jpg|jpeg)$/i, ""), url];
  }),
);

interface DoudizhuTableProps {
  tableScene: TableSceneData;
  actionTexts: string[];
  canSubmitActions: boolean;
  onSubmitAction: (actionText: string) => void;
}

function resolveLatestSeatChatMap(tableScene: TableSceneData): Map<string, string> {
  const latestByPlayerId = new Map<string, string>();
  tableScene.panels.chatLog.forEach((entry) => {
    latestByPlayerId.set(entry.playerId, entry.text);
  });
  return latestByPlayerId;
}

function resolveSeatById(tableScene: TableSceneData, seatId: string): TableSeat | null {
  return tableScene.table.seats.find((seat) => seat.seatId === seatId) ?? null;
}

function resolvePortrait(role: string | null): string | null {
  if (role === "landlord") {
    return portraitAssets.Landlord_wName ?? portraitAssets.Landlord ?? null;
  }
  if (role === "peasant") {
    return portraitAssets.Peasant_wName ?? portraitAssets.Pleasant ?? null;
  }
  return portraitAssets.Player ?? null;
}

function renderMaskedHand(maskedCount: number, compact: boolean) {
  const previewCount = Math.min(Math.max(maskedCount, 1), compact ? 6 : 8);
  return (
    <div className="doudizhu-hand doudizhu-hand--masked">
      <div className="doudizhu-hand__cards">
        {Array.from({ length: previewCount }, (_, index) => (
          <DoudizhuCardVisual key={`masked-${index}`} card="Back" faceDown={true} compact={compact} />
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
        <DoudizhuCardVisual key={`${card}-${index}`} card={card} compact={compact} />
      ))}
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
        <DoudizhuCardVisual key={`${seat.playerId}-${card}-${index}`} card={card} compact={true} />
      ))}
    </div>
  );
}

function DoudizhuSeat({
  seat,
  anchor,
  handContent,
  chatBubbleText,
}: {
  seat: TableSeat;
  anchor: "bottom" | "left" | "right";
  handContent?: ReactNode;
  chatBubbleText?: string;
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
  const portrait = resolvePortrait(seat.role);

  return (
    <article
      className={className}
      aria-label={`Doudizhu seat ${anchor}`}
      data-testid={`doudizhu-seat-${anchor}`}
    >
      <header className="doudizhu-seat__header">
        <div className="doudizhu-seat__identity">
          {portrait ? (
            <img
              className="doudizhu-seat__portrait"
              src={portrait}
              alt={seat.role ?? "player"}
            />
          ) : null}
          <div>
            <p className="doudizhu-seat__name">{seat.playerName}</p>
            <p className="doudizhu-seat__meta">{seat.playerId}</p>
          </div>
        </div>
        <div className="doudizhu-seat__chips">
          {seat.role ? <span className="doudizhu-seat__badge">{seat.role}</span> : null}
          {seat.isObserver ? <span className="doudizhu-seat__marker">you</span> : null}
        </div>
      </header>

      {chatBubbleText ? (
        <p className="doudizhu-seat__bubble" data-testid={`doudizhu-seat-${anchor}-bubble`}>
          {chatBubbleText}
        </p>
      ) : null}

      <div className="doudizhu-seat__played">{renderSeatPlayed(seat)}</div>

      <div data-testid={`doudizhu-seat-${anchor}-hand`} className="doudizhu-seat__hand">
        {handContent ??
          (seat.hand.isVisible
            ? (
                <div className="doudizhu-hand">
                  {renderVisibleCards(seat.hand.cards, compact)}
                </div>
              )
            : renderMaskedHand(seat.hand.maskedCount, compact))}
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
  const latestSeatChats = resolveLatestSeatChatMap(tableScene);

  if (!bottomSeat || !leftSeat || !rightSeat) {
    return null;
  }

  const bottomHandContent =
    bottomSeat.hand.isVisible && canSubmitActions ? (
      <DoudizhuActionComposer
        handCards={bottomSeat.hand.cards}
        actionTexts={actionTexts}
        canSubmitActions={canSubmitActions}
        onSubmitAction={onSubmitAction}
      />
    ) : bottomSeat.hand.isVisible ? (
      <div className="doudizhu-hand">{renderVisibleCards(bottomSeat.hand.cards, false)}</div>
    ) : (
      renderMaskedHand(bottomSeat.hand.maskedCount, false)
    );

  return (
    <section className="doudizhu-stage" data-testid="doudizhu-stage">
      <div className="doudizhu-stage__table">
        <DoudizhuSeat
          seat={leftSeat}
          anchor="left"
          chatBubbleText={latestSeatChats.get(leftSeat.playerId)}
        />
        <DoudizhuSeat
          seat={rightSeat}
          anchor="right"
          chatBubbleText={latestSeatChats.get(rightSeat.playerId)}
        />
        <DoudizhuSeat
          seat={bottomSeat}
          anchor="bottom"
          handContent={bottomHandContent}
          chatBubbleText={latestSeatChats.get(bottomSeat.playerId)}
        />

        <section className="doudizhu-center">
          <p className="doudizhu-center__eyebrow">{tableScene.table.center.label}</p>
          <div className="doudizhu-center__cards" data-testid="doudizhu-center-cards">
            {tableScene.table.center.cards.length > 0 ? (
              tableScene.table.center.cards.map((card, index) => (
                <DoudizhuCardVisual key={`center-${card}-${index}`} card={card} compact={true} />
              ))
            ) : (
              <p className="doudizhu-seat__placeholder">No center cards</p>
            )}
          </div>
          {tableScene.table.center.history.length > 0 ? (
            <div className="doudizhu-center__history">
              {tableScene.table.center.history.map((entry) => (
                <span key={entry} className="doudizhu-center__history-chip">
                  {entry}
                </span>
              ))}
            </div>
          ) : null}
        </section>
      </div>
    </section>
  );
}

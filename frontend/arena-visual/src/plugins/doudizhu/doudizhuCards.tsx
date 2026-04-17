import { memo, type MouseEventHandler } from "react";

import { cx } from "../../lib/cx";

const DOUDIZHU_ACTION_TOKEN_PATTERN = /[3456789TJQKA2BR]/g;

const cardAssetModules = import.meta.glob("./assets/cards/*.{png,jpg,jpeg}", {
  eager: true,
  import: "default",
}) as Record<string, string>;

const cardAssets = Object.fromEntries(
  Object.entries(cardAssetModules).map(([path, url]) => {
    const fileName = path.split("/").pop() ?? path;
    return [fileName.replace(/\.(png|jpg|jpeg)$/i, ""), url];
  }),
);

function buildTokenCounts(tokens: string[]): Map<string, number> {
  const counts = new Map<string, number>();
  tokens.forEach((token) => {
    counts.set(token, (counts.get(token) ?? 0) + 1);
  });
  return counts;
}

function haveSameCounts(left: string[], right: string[]): boolean {
  if (left.length !== right.length) {
    return false;
  }
  const leftCounts = buildTokenCounts(left);
  const rightCounts = buildTokenCounts(right);
  if (leftCounts.size !== rightCounts.size) {
    return false;
  }
  return Array.from(leftCounts.entries()).every(([token, count]) => rightCounts.get(token) === count);
}

export function isDoudizhuJoker(card: string): boolean {
  return (
    card === "BJ" ||
    card === "RJ" ||
    card === "B" ||
    card === "R" ||
    card === "BlackJoker" ||
    card === "RedJoker"
  );
}

export function isDoudizhuRed(card: string): boolean {
  return (
    card.startsWith("H") ||
    card.startsWith("D") ||
    card === "RJ" ||
    card === "R" ||
    card === "RedJoker"
  );
}

export function resolveDoudizhuCardLabel(card: string): string {
  if (card === "BJ" || card === "B" || card === "BlackJoker") {
    return "BlackJoker";
  }
  if (card === "RJ" || card === "R" || card === "RedJoker") {
    return "RedJoker";
  }
  return card;
}

function normalizeSuitfulCard(card: string): string {
  const normalized = resolveDoudizhuCardLabel(card).trim();
  if (normalized === "BlackJoker") {
    return "BJ";
  }
  if (normalized === "RedJoker") {
    return "RJ";
  }
  if (/^[SHDC](10|[3456789TJQKA2])$/i.test(normalized)) {
    const suit = normalized.charAt(0).toUpperCase();
    const rank = normalized.slice(1).toUpperCase();
    return `${suit}${rank === "10" || rank === "T" ? "10" : rank}`;
  }
  if (/^(10|[3456789TJQKA2])[SHDC]$/i.test(normalized)) {
    const suit = normalized.slice(-1).toUpperCase();
    const rank = normalized.slice(0, -1).toUpperCase();
    return `${suit}${rank === "10" || rank === "T" ? "10" : rank}`;
  }
  return normalized;
}

export function resolveDoudizhuCardToken(card: string): string | null {
  const normalized = normalizeSuitfulCard(card);
  if (normalized === "10") {
    return "T";
  }
  if (normalized === "BJ") {
    return "B";
  }
  if (normalized === "RJ") {
    return "R";
  }
  if (/^[3456789JQKA2]$/i.test(normalized)) {
    return normalized.toUpperCase();
  }
  if (/^[SHDC](10|[3456789TJQKA2])$/i.test(normalized)) {
    const rank = normalized.slice(1).toUpperCase();
    return rank === "10" ? "T" : rank;
  }
  return null;
}

export function tokenizeDoudizhuAction(actionText: string): string[] {
  return (actionText.toUpperCase().match(DOUDIZHU_ACTION_TOKEN_PATTERN) ?? []).map((token) => token.toUpperCase());
}

export function matchLegalActionForSelection(actionTexts: string[], selectedCards: string[]): string | null {
  const selectedTokens = selectedCards
    .map(resolveDoudizhuCardToken)
    .filter((token): token is string => token !== null);

  if (selectedTokens.length === 0) {
    return null;
  }

  const exactOrder = actionTexts.find((actionText) => {
    const actionTokens = tokenizeDoudizhuAction(actionText);
    return actionTokens.join("") === selectedTokens.join("");
  });
  if (exactOrder) {
    return exactOrder;
  }

  return (
    actionTexts.find((actionText) => {
      if (actionText.trim().toLowerCase() === "pass") {
        return false;
      }
      return haveSameCounts(tokenizeDoudizhuAction(actionText), selectedTokens);
    }) ?? null
  );
}

export function selectHandIndexesForAction(handCards: string[], actionText: string): number[] {
  const required = buildTokenCounts(tokenizeDoudizhuAction(actionText));
  if (required.size === 0) {
    return [];
  }

  const selectedIndexes: number[] = [];
  handCards.forEach((card, index) => {
    const token = resolveDoudizhuCardToken(card);
    if (!token) {
      return;
    }
    const remaining = required.get(token) ?? 0;
    if (remaining <= 0) {
      return;
    }
    selectedIndexes.push(index);
    if (remaining === 1) {
      required.delete(token);
      return;
    }
    required.set(token, remaining - 1);
  });

  return required.size === 0 ? selectedIndexes : [];
}

export function resolveHintAction(actionTexts: string[], handCards: string[]): string | null {
  return (
    actionTexts.find((actionText) => {
      if (actionText.trim().toLowerCase() === "pass") {
        return false;
      }
      return selectHandIndexesForAction(handCards, actionText).length > 0;
    }) ?? null
  );
}

export function resolveDoudizhuRank(card: string): string {
  const normalized = normalizeSuitfulCard(card);
  if (normalized === "BJ") {
    return "BJ";
  }
  if (normalized === "RJ") {
    return "RJ";
  }
  if (/^[SHDC](10|[3456789TJQKA2])$/i.test(normalized)) {
    const rank = normalized.slice(1).toUpperCase();
    return rank === "T" ? "10" : rank;
  }
  if (normalized === "10") {
    return "10";
  }
  return normalized;
}

export function resolveDoudizhuSuit(card: string): string {
  const normalized = normalizeSuitfulCard(card);
  if (!/^[SHDC](10|[3456789TJQKA2])$/i.test(normalized)) {
    return "";
  }
  const suit = normalized.charAt(0).toUpperCase();
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

function resolveDoudizhuCardAssetKey(card: string, faceDown: boolean): string | null {
  if (faceDown) {
    return "pokerback";
  }
  const normalized = normalizeSuitfulCard(card);
  if (normalized === "BJ") {
    return "Poker_Joker_B";
  }
  if (normalized === "RJ") {
    return "Poker_Joker_R";
  }
  if (/^[SHDC](10|[3456789TJQKA2])$/i.test(normalized)) {
    return `Poker_${normalized.charAt(0).toUpperCase()}${normalized.slice(1).toUpperCase()}`;
  }
  return null;
}

function resolveDoudizhuCardAsset(card: string, faceDown: boolean): string | null {
  const assetKey = resolveDoudizhuCardAssetKey(card, faceDown);
  if (!assetKey) {
    return null;
  }
  return cardAssets[assetKey] ?? null;
}

interface DoudizhuCardVisualProps {
  card: string;
  faceDown?: boolean;
  compact?: boolean;
  selected?: boolean;
  disabled?: boolean;
  ariaLabel?: string;
  cardIndex?: number;
  onClick?: MouseEventHandler<HTMLButtonElement>;
}

export const DoudizhuCardVisual = memo(function DoudizhuCardVisual({
  card,
  faceDown = false,
  compact = false,
  selected = false,
  disabled = false,
  ariaLabel,
  cardIndex,
  onClick,
}: DoudizhuCardVisualProps) {
  const label = resolveDoudizhuCardLabel(card);
  const assetSrc = resolveDoudizhuCardAsset(card, faceDown);
  const className = cx(
    "doudizhu-card",
    compact && "is-compact",
    faceDown && "doudizhu-card--back",
    isDoudizhuJoker(card) && "doudizhu-card--joker",
    isDoudizhuRed(card) ? "doudizhu-card--red" : "doudizhu-card--black",
    onClick && "is-interactive",
    selected && "is-selected",
  );
  const content = assetSrc ? (
    <>
      <img className="doudizhu-card__image" src={assetSrc} alt="" aria-hidden="true" />
      {!faceDown ? (
        <span className="doudizhu-card__sr-label">
          {resolveDoudizhuRank(card)}
          {resolveDoudizhuSuit(card)}
        </span>
      ) : null}
    </>
  ) : (
    <>
      <span className="doudizhu-card__rank">{resolveDoudizhuRank(card)}</span>
      <span className="doudizhu-card__suit">{resolveDoudizhuSuit(card)}</span>
    </>
  );

  if (faceDown) {
    return <div className={className} aria-hidden="true">{content}</div>;
  }

  if (onClick) {
    return (
      <button
        type="button"
        className={className}
        aria-label={ariaLabel ?? `Select card ${label}`}
        aria-pressed={selected}
        disabled={disabled}
        data-card-index={cardIndex}
        onClick={onClick}
      >
        {content}
      </button>
    );
  }

  return (
    <div className={className} aria-label={`Doudizhu card ${label}`}>
      {content}
    </div>
  );
});

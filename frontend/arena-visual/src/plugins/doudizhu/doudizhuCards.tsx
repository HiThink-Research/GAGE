const DOUDIZHU_ACTION_TOKEN_PATTERN = /[3456789TJQKA2BR]/g;

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

export function resolveDoudizhuCardToken(card: string): string | null {
  const normalized = resolveDoudizhuCardLabel(card).trim();
  if (normalized === "10") {
    return "T";
  }
  if (normalized === "BlackJoker") {
    return "B";
  }
  if (normalized === "RedJoker") {
    return "R";
  }
  if (/^[3456789JQKA2]$/i.test(normalized)) {
    return normalized.toUpperCase();
  }
  if (/^[SHDC][3456789TJQKA2]$/i.test(normalized)) {
    return normalized.slice(-1).toUpperCase();
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
  const token = resolveDoudizhuCardToken(card);
  if (token === "T") {
    return "10";
  }
  if (token === "B") {
    return "BJ";
  }
  if (token === "R") {
    return "RJ";
  }
  return token ?? resolveDoudizhuCardLabel(card);
}

export function resolveDoudizhuSuit(card: string): string {
  const normalized = resolveDoudizhuCardLabel(card);
  if (normalized.length === 1 || isDoudizhuJoker(normalized)) {
    return "";
  }
  if (normalized === "10") {
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

interface DoudizhuCardVisualProps {
  card: string;
  faceDown?: boolean;
  compact?: boolean;
  selected?: boolean;
  disabled?: boolean;
  ariaLabel?: string;
  onClick?: () => void;
}

export function DoudizhuCardVisual({
  card,
  faceDown = false,
  compact = false,
  selected = false,
  disabled = false,
  ariaLabel,
  onClick,
}: DoudizhuCardVisualProps) {
  if (faceDown) {
    return (
      <div
        className={`doudizhu-card doudizhu-card--back ${compact ? "is-compact" : ""}`}
        aria-hidden="true"
      />
    );
  }

  const label = resolveDoudizhuCardLabel(card);
  const className = [
    "doudizhu-card",
    compact ? "is-compact" : "",
    isDoudizhuJoker(card) ? "doudizhu-card--joker" : "",
    isDoudizhuRed(card) ? "doudizhu-card--red" : "doudizhu-card--black",
    onClick ? "is-interactive" : "",
    selected ? "is-selected" : "",
  ]
    .filter((value) => value !== "")
    .join(" ");
  const content = (
    <>
      <span className="doudizhu-card__rank">{resolveDoudizhuRank(card)}</span>
      <span className="doudizhu-card__suit">{resolveDoudizhuSuit(card) || "•"}</span>
    </>
  );

  if (onClick) {
    return (
      <button
        type="button"
        className={className}
        aria-label={ariaLabel ?? `Select card ${label}`}
        aria-pressed={selected}
        disabled={disabled}
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
}

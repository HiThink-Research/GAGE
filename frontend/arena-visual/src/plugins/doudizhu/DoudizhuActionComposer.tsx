import { useCallback, useEffect, useMemo, useState, type MouseEvent } from "react";

import {
  DoudizhuCardVisual,
  matchLegalActionForSelection,
  resolveDoudizhuCardLabel,
  resolveHintAction,
  selectHandIndexesForAction,
} from "./doudizhuCards";

interface DoudizhuActionComposerProps {
  handCards: string[];
  actionTexts: string[];
  canSubmitActions: boolean;
  onSubmitAction: (actionText: string) => void;
}

export function DoudizhuActionComposer({
  handCards,
  actionTexts,
  canSubmitActions,
  onSubmitAction,
}: DoudizhuActionComposerProps) {
  const [selectedIndexes, setSelectedIndexes] = useState<number[]>([]);
  const [showFallbackActions, setShowFallbackActions] = useState(false);
  const handCardsKey = useMemo(() => handCards.join("\u0001"), [handCards]);

  useEffect(() => {
    if (!canSubmitActions) {
      setSelectedIndexes([]);
      setShowFallbackActions(false);
      return;
    }
    setSelectedIndexes([]);
    setShowFallbackActions(false);
  }, [canSubmitActions, handCardsKey]);

  const selectedCards = useMemo(
    () => selectedIndexes.map((index) => handCards[index]).filter((card): card is string => typeof card === "string"),
    [handCards, selectedIndexes],
  );
  const matchedAction = useMemo(
    () => matchLegalActionForSelection(actionTexts, selectedCards),
    [actionTexts, selectedCards],
  );
  const hintAction = useMemo(() => resolveHintAction(actionTexts, handCards), [actionTexts, handCards]);
  const canPass = canSubmitActions && actionTexts.some((actionText) => actionText.trim().toLowerCase() === "pass");
  const fallbackActions = useMemo(
    () => actionTexts.filter((actionText) => actionText.trim() !== "" && actionText.trim().toLowerCase() !== "pass"),
    [actionTexts],
  );

  const toggleCard = useCallback((index: number) => {
    setSelectedIndexes((current) =>
      current.includes(index) ? current.filter((value) => value !== index) : current.concat(index),
    );
  }, []);
  const handleCardClick = useCallback((event: MouseEvent<HTMLButtonElement>) => {
    const rawIndex = event.currentTarget.dataset.cardIndex;
    if (rawIndex === undefined) {
      return;
    }
    const index = Number.parseInt(rawIndex, 10);
    if (!Number.isInteger(index)) {
      return;
    }
    toggleCard(index);
  }, [toggleCard]);

  return (
    <div className="doudizhu-composer">
      <div className="doudizhu-hand__cards doudizhu-hand__cards--interactive">
        {handCards.map((card, index) => (
          <DoudizhuCardVisual
            key={`${card}-${index}`}
            card={card}
            selected={selectedIndexes.includes(index)}
            disabled={!canSubmitActions}
            ariaLabel={`Select card ${resolveDoudizhuCardLabel(card)}`}
            cardIndex={index}
            onClick={canSubmitActions ? handleCardClick : undefined}
          />
        ))}
      </div>

      <div className="doudizhu-composer__toolbar">
        <p className="doudizhu-composer__summary">
          {selectedCards.length > 0
            ? `Selected · ${selectedCards.map(resolveDoudizhuCardLabel).join(" ")}`
            : "Select cards to compose a legal move."}
        </p>

        <div className="doudizhu-actions">
          <button
            type="button"
            className="doudizhu-action doudizhu-action--ghost"
            aria-label="Hint"
            disabled={!canSubmitActions || !hintAction}
            onClick={() => {
              if (!hintAction) {
                return;
              }
              setSelectedIndexes(selectHandIndexesForAction(handCards, hintAction));
            }}
          >
            <span className="doudizhu-action__label">Hint</span>
          </button>

          <button
            type="button"
            className="doudizhu-action doudizhu-action--ghost"
            aria-label="Clear selected cards"
            disabled={selectedIndexes.length === 0}
            onClick={() => {
              setSelectedIndexes([]);
            }}
          >
            <span className="doudizhu-action__label">Clear</span>
          </button>

          {canPass ? (
            <button
              type="button"
              className="doudizhu-action"
              aria-label="Pass"
              disabled={!canSubmitActions}
              onClick={() => {
                onSubmitAction("pass");
                setSelectedIndexes([]);
              }}
            >
              <span className="doudizhu-action__label">Pass</span>
            </button>
          ) : null}

          <button
            type="button"
            className="doudizhu-action doudizhu-action--primary"
            aria-label={matchedAction ? `Play ${matchedAction}` : "Play Selected"}
            disabled={!canSubmitActions || !matchedAction}
            onClick={() => {
              if (!matchedAction) {
                return;
              }
              onSubmitAction(matchedAction);
              setSelectedIndexes([]);
            }}
          >
            <span className="doudizhu-action__label">
              {matchedAction ? `Play ${matchedAction}` : "Play Selected"}
            </span>
          </button>
        </div>

        {fallbackActions.length > 0 ? (
          <div className="doudizhu-fallback">
            <button
              type="button"
              className="doudizhu-fallback__toggle"
              aria-expanded={showFallbackActions}
              aria-label={showFallbackActions ? "Hide legal actions" : "Show legal actions"}
              disabled={!canSubmitActions}
              onClick={() => {
                setShowFallbackActions((current) => !current);
              }}
            >
              {showFallbackActions
                ? "Hide legal actions"
                : `Show legal actions (${fallbackActions.length})`}
            </button>
            {showFallbackActions ? (
              <>
                <p className="doudizhu-fallback__label">Direct legal actions</p>
                <div className="doudizhu-actions doudizhu-actions--fallback">
                  {fallbackActions.map((actionText) => (
                    <button
                      key={`fallback-${actionText}`}
                      type="button"
                      className="doudizhu-action doudizhu-action--fallback"
                      aria-label={`Play legal ${actionText}`}
                      disabled={!canSubmitActions}
                      onClick={() => {
                        onSubmitAction(actionText);
                        setSelectedIndexes([]);
                      }}
                    >
                      <span className="doudizhu-action__label">{actionText}</span>
                    </button>
                  ))}
                </div>
              </>
            ) : null}
          </div>
        ) : null}
      </div>
    </div>
  );
}

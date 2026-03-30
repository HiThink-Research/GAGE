interface MahjongActionTrayProps {
  selectedTile: string | null;
  trayActions: string[];
  canSubmitActions: boolean;
  onClearSelection: () => void;
  onConfirmTile: (actionText: string) => void;
  onSubmitAction: (actionText: string) => void;
}

export function MahjongActionTray({
  selectedTile,
  trayActions,
  canSubmitActions,
  onClearSelection,
  onConfirmTile,
  onSubmitAction,
}: MahjongActionTrayProps) {
  if (!selectedTile && trayActions.length === 0) {
    return null;
  }

  return (
    <div className="mahjong-actions">
      {selectedTile ? (
        <>
          <button
            type="button"
            className="mahjong-action mahjong-action--confirm"
            aria-label={`Confirm ${selectedTile}`}
            disabled={!canSubmitActions}
            onClick={() => {
              onConfirmTile(selectedTile);
            }}
          >
            <span className="mahjong-action__label">{`Confirm ${selectedTile}`}</span>
          </button>
          <button
            type="button"
            className="mahjong-action mahjong-action--ghost"
            aria-label="Clear selected tile"
            onClick={onClearSelection}
          >
            <span className="mahjong-action__label">Clear</span>
          </button>
        </>
      ) : null}

      {trayActions.map((actionText) => (
        <button
          key={actionText}
          type="button"
          className="mahjong-action"
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
  );
}

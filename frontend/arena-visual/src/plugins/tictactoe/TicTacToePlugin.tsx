import type { ArenaPluginRenderProps } from "../sdk/contracts";
import { BoardGrid } from "../board/BoardGrid";
import {
  formatBoardActorLabel,
  readBoardScene,
  readLegalCoords,
  resolveBoardActorId,
} from "../board/boardScene";

export function TicTacToePlugin({
  session,
  scene,
  submitAction,
}: ArenaPluginRenderProps) {
  const boardScene = readBoardScene(scene);

  if (!boardScene) {
    return (
      <section className="plugin-stage-card">
        <p className="eyebrow">Tic-Tac-Toe</p>
        <h2>Board unavailable</h2>
        <p className="plugin-stage-card__copy">Waiting for structured board scene data.</p>
      </section>
    );
  }

  const legalCoords = readLegalCoords(scene);
  const resolvedActorId = resolveBoardActorId(session, scene, boardScene);
  const actorLabel = formatBoardActorLabel(session, boardScene.players, resolvedActorId);

  return (
    <BoardGrid
      gameLabel="Tic-Tac-Toe"
      actorLabel={actorLabel}
      boardSize={boardScene.board.size}
      cells={boardScene.board.cells}
      players={boardScene.players}
      legalCoords={legalCoords}
      canSubmitMoves={session.scheduling.acceptsHumanIntent}
      onSubmitMove={(coord) => {
        if (!resolvedActorId) {
          return;
        }
        void submitAction({
          playerId: resolvedActorId,
          action: { move: coord },
        });
      }}
    />
  );
}

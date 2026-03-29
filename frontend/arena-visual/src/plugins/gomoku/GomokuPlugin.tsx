import type { ArenaPluginRenderProps } from "../sdk/contracts";
import { BoardGrid } from "../board/BoardGrid";
import {
  formatBoardActorLabel,
  readBoardScene,
  readLegalCoords,
  resolveBoardActorId,
} from "../board/boardScene";

interface GomokuInputEvent {
  playerId: string;
  coord: string;
}

export function GomokuPlugin({
  session,
  scene,
  submitInput,
}: ArenaPluginRenderProps<GomokuInputEvent>) {
  const boardScene = readBoardScene(scene);

  if (!boardScene) {
    return (
      <section className="plugin-stage-card">
        <p className="eyebrow">Gomoku</p>
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
      variant="gomoku"
      gameLabel="Gomoku"
      actorLabel={actorLabel}
      boardSize={boardScene.board.size}
      coordScheme={boardScene.board.coordScheme}
      cells={boardScene.board.cells}
      players={boardScene.players}
      status={boardScene.status}
      legalCoords={legalCoords}
      canSubmitMoves={session.scheduling.acceptsHumanIntent}
      onSubmitMove={(coord) => {
        if (!resolvedActorId) {
          return;
        }
        void submitInput({
          playerId: resolvedActorId,
          coord,
        });
      }}
    />
  );
}

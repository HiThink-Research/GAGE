import type { ArenaPluginRenderProps } from "../sdk/contracts";
import {
  readBoardScene,
  readLegalCoords,
  resolveBoardActorId,
} from "../board/boardScene";
import { TicTacToeBoard } from "./TicTacToeBoard";

export function TicTacToePlugin({
  session,
  scene,
  submitInput,
}: ArenaPluginRenderProps<{ playerId: string; coord: string }>) {
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

  return (
    <TicTacToeBoard
      board={boardScene.board}
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

import type { ArenaPluginRenderProps } from "../sdk/contracts";
import {
  readBoardScene,
  readLegalCoords,
  resolveBoardActorId,
} from "../board/boardScene";
import { GomokuBoard } from "./GomokuBoard";

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

  return (
    <GomokuBoard
      board={boardScene.board}
      legalCoords={legalCoords}
      winningLine={boardScene.status.winningLine}
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

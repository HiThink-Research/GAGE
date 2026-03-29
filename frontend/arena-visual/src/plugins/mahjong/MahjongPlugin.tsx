import type { ArenaPluginRenderProps } from "../sdk/contracts";
import {
  readTableActionTexts,
  readTableScene,
  resolveTableActorId,
} from "../table/TableLayout";
import { MahjongTable } from "./MahjongTable";

export function MahjongPlugin({
  session,
  scene,
  submitInput,
}: ArenaPluginRenderProps<{ playerId: string; actionText: string }>) {
  const tableScene = readTableScene(scene);

  if (!tableScene) {
    return (
      <section className="plugin-stage-card">
        <p className="eyebrow">Mahjong</p>
        <h2>Table unavailable</h2>
        <p className="plugin-stage-card__copy">Waiting for structured table scene data.</p>
      </section>
    );
  }

  const actionTexts = readTableActionTexts(scene);
  const resolvedActorId = resolveTableActorId(session, scene, tableScene);

  return (
    <MahjongTable
      tableScene={tableScene}
      actionTexts={actionTexts}
      canSubmitActions={session.scheduling.acceptsHumanIntent}
      onSubmitAction={(actionText) => {
        if (!resolvedActorId) {
          return;
        }
        void submitInput({
          playerId: resolvedActorId,
          actionText,
        });
      }}
    />
  );
}

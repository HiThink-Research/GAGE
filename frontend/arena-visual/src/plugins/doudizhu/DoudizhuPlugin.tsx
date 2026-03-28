import type { ArenaPluginRenderProps } from "../sdk/contracts";
import {
  formatTableActorLabel,
  readTableActionTexts,
  readTableScene,
  resolveTableActorId,
  TableLayout,
} from "../table/TableLayout";

export function DoudizhuPlugin({ session, scene, submitAction }: ArenaPluginRenderProps) {
  const tableScene = readTableScene(scene);

  if (!tableScene) {
    return (
      <section className="plugin-stage-card">
        <p className="eyebrow">Doudizhu</p>
        <h2>Table unavailable</h2>
        <p className="plugin-stage-card__copy">Waiting for structured table scene data.</p>
      </section>
    );
  }

  const actionTexts = readTableActionTexts(scene);
  const resolvedActorId = resolveTableActorId(session, scene, tableScene);
  const actorLabel = formatTableActorLabel(session, resolvedActorId);

  return (
    <TableLayout
      gameLabel="Doudizhu"
      actorLabel={actorLabel}
      tableScene={tableScene}
      actionTexts={actionTexts}
      canSubmitActions={session.scheduling.acceptsHumanIntent}
      onSubmitAction={(actionText) => {
        if (!resolvedActorId) {
          return;
        }
        void submitAction({
          playerId: resolvedActorId,
          action: { move: actionText },
        });
      }}
    />
  );
}

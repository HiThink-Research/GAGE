import { useCallback } from "react";
import type { ArenaPluginRenderProps } from "../sdk/contracts";
import {
  readTableActionTexts,
  readTableScene,
  resolveTableActorId,
} from "../table/TableLayout";
import { DoudizhuTable } from "./DoudizhuTable";

export function DoudizhuPlugin({
  session,
  scene,
  submitInput,
}: ArenaPluginRenderProps<{ playerId: string; actionText: string }>) {
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
  const handleSubmitAction = useCallback(
    (actionText: string) => {
      if (!resolvedActorId) {
        return;
      }
      void submitInput({
        playerId: resolvedActorId,
        actionText,
      });
    },
    [resolvedActorId, submitInput],
  );

  return (
    <DoudizhuTable
      tableScene={tableScene}
      actionTexts={actionTexts}
      canSubmitActions={session.scheduling.acceptsHumanIntent}
      onSubmitAction={handleSubmitAction}
    />
  );
}

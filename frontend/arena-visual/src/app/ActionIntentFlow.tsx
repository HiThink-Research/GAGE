import type { ActionIntentReceipt } from "../gateway/types";

interface ActionIntentFlowProps {
  latestReceipt?: ActionIntentReceipt;
  error?: string;
  isSubmitting?: boolean;
}

function formatReceipt(latestReceipt?: ActionIntentReceipt): string {
  if (!latestReceipt) {
    return "No host requests yet";
  }
  return `${latestReceipt.state} · ${latestReceipt.intentId}`;
}

export function ActionIntentFlow({
  latestReceipt,
  error,
  isSubmitting,
}: ActionIntentFlowProps) {
  return (
    <section className="action-intent-flow" aria-label="Action intent flow">
      <article className="side-panel__card">
        <h2>Host receipt</h2>
        <p>{isSubmitting ? "Submitting host request..." : formatReceipt(latestReceipt)}</p>
        {latestReceipt?.relatedEventSeq !== undefined ? (
          <p>{`Event seq ${latestReceipt.relatedEventSeq}`}</p>
        ) : null}
        {latestReceipt?.reason ? <p>{latestReceipt.reason}</p> : null}
      </article>
      {error ? (
        <article className="side-panel__card side-panel__card--error">
          <h2>Host error</h2>
          <p>{error}</p>
        </article>
      ) : null}
    </section>
  );
}

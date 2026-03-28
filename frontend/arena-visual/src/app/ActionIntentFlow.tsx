import type { ActionIntentReceipt } from "../gateway/types";

interface ActionIntentFlowProps {
  latestReceipt?: ActionIntentReceipt;
  error?: string;
  isSubmitting?: boolean;
}

function formatReceipt(latestReceipt?: ActionIntentReceipt): string {
  if (!latestReceipt) {
    return "No action submitted";
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
        <h2>Action intent</h2>
        <p>{isSubmitting ? "Submitting action..." : formatReceipt(latestReceipt)}</p>
        {latestReceipt?.reason ? <p>{latestReceipt.reason}</p> : null}
      </article>
      {error ? (
        <article className="side-panel__card side-panel__card--error">
          <h2>Action error</h2>
          <p>{error}</p>
        </article>
      ) : null}
    </section>
  );
}

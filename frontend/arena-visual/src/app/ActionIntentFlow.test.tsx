import { render, screen } from "@testing-library/react";

import { ActionIntentFlow } from "./ActionIntentFlow";

describe("ActionIntentFlow", () => {
  it("renders receipt transitions and submission errors from the action submit flow", () => {
    const { rerender } = render(
      <ActionIntentFlow
        isSubmitting={true}
        latestReceipt={{
          intentId: "sample-human-1:intent-1",
          state: "pending",
        }}
      />,
    );

    expect(screen.getByRole("heading", { name: /action intent/i })).toBeInTheDocument();
    expect(screen.getByText(/submitting action/i)).toBeInTheDocument();

    rerender(
      <ActionIntentFlow
        error="queue_unavailable"
        isSubmitting={false}
        latestReceipt={{
          intentId: "sample-human-1:intent-1",
          state: "accepted",
          relatedEventSeq: 1,
          reason: "queued",
        }}
      />,
    );

    expect(
      screen.getByText(/accepted · sample-human-1:intent-1/i),
    ).toBeInTheDocument();
    expect(screen.getByText(/queued/i)).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /action error/i })).toBeInTheDocument();
    expect(screen.getByText(/queue_unavailable/i)).toBeInTheDocument();
  });
});

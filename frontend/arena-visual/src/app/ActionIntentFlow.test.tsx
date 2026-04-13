import { render, screen } from "@testing-library/react";

import { ActionIntentFlow } from "./ActionIntentFlow";

describe("ActionIntentFlow", () => {
  it("renders generic host receipt transitions and submission errors", () => {
    const { rerender } = render(
      <ActionIntentFlow
        isSubmitting={true}
        latestReceipt={{
          intentId: "sample-human-1:intent-1",
          state: "pending",
        }}
      />,
    );

    expect(screen.getByRole("heading", { name: /host receipt/i })).toBeInTheDocument();
    expect(screen.getByText(/submitting host request/i)).toBeInTheDocument();

    rerender(
      <ActionIntentFlow
        error="queue_unavailable"
        isSubmitting={false}
        latestReceipt={{
          intentId: "chat-1",
          state: "committed",
          relatedEventSeq: 14,
          reason: "queued",
        }}
      />,
    );

    expect(screen.getByText(/committed · chat-1/i)).toBeInTheDocument();
    expect(screen.getByText(/event seq 14/i)).toBeInTheDocument();
    expect(screen.getByText(/queued/i)).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /host error/i })).toBeInTheDocument();
    expect(screen.getByText(/queue_unavailable/i)).toBeInTheDocument();
  });
});

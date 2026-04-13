import { fireEvent, render, screen } from "@testing-library/react";

import { TimelineView } from "./TimelineView";

describe("TimelineView", () => {
  it("renders only events that match the active timeline filters", () => {
    render(
      <TimelineView
        events={[
          {
            seq: 1,
            tsMs: 1001,
            type: "action_intent",
            label: "Human move",
            severity: "warn",
            tags: ["human_intent"],
          },
          {
            seq: 2,
            tsMs: 1002,
            type: "result",
            label: "Result posted",
            severity: "info",
          },
        ]}
        filters={{
          eventTypes: ["action_intent"],
          severity: "warn",
          humanIntentOnly: true,
        }}
        currentSeq={1}
        status="ready"
        hasMore={false}
        onSelectEvent={vi.fn()}
        onLoadMore={vi.fn()}
        onFiltersChange={vi.fn()}
      />,
    );

    expect(screen.getByRole("button", { name: /human move/i })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /result posted/i })).not.toBeInTheDocument();
  });

  it("emits updated filter state when the host toggles filter controls", () => {
    const onFiltersChange = vi.fn();

    render(
      <TimelineView
        events={[
          {
            seq: 1,
            tsMs: 1001,
            type: "action_intent",
            label: "Human move",
            severity: "warn",
            tags: ["human_intent"],
          },
          {
            seq: 2,
            tsMs: 1002,
            type: "result",
            label: "Result posted",
            severity: "critical",
          },
        ]}
        filters={{
          eventTypes: [],
          severity: "all",
          humanIntentOnly: false,
        }}
        currentSeq={1}
        status="ready"
        hasMore={false}
        onSelectEvent={vi.fn()}
        onLoadMore={vi.fn()}
        onFiltersChange={onFiltersChange}
      />,
    );

    fireEvent.click(screen.getByLabelText(/action_intent/i));
    fireEvent.change(screen.getByLabelText(/severity/i), {
      target: { value: "critical" },
    });
    fireEvent.click(screen.getByLabelText(/human intent only/i));

    expect(onFiltersChange).toHaveBeenNthCalledWith(1, {
      eventTypes: ["action_intent"],
      severity: "all",
      humanIntentOnly: false,
    });
    expect(onFiltersChange).toHaveBeenNthCalledWith(2, {
      eventTypes: [],
      severity: "critical",
      humanIntentOnly: false,
    });
    expect(onFiltersChange).toHaveBeenNthCalledWith(3, {
      eventTypes: [],
      severity: "all",
      humanIntentOnly: true,
    });
  });
});

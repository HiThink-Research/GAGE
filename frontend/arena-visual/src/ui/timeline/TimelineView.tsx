import type { TimelineEvent } from "../../gateway/types";

interface TimelineViewProps {
  events: TimelineEvent[];
  currentSeq?: number;
  status: "idle" | "loading" | "ready" | "error";
  hasMore: boolean;
  onSelectEvent: (seq: number) => void;
  onLoadMore: () => void;
}

export function TimelineView({
  events,
  currentSeq,
  status,
  hasMore,
  onSelectEvent,
  onLoadMore,
}: TimelineViewProps) {
  return (
    <section className="timeline-view" aria-label="Timeline">
      <div className="timeline-view__header">
        <div>
          <p className="eyebrow">Timeline</p>
          <h2>Event stream</h2>
        </div>
        <span className="timeline-view__status">{status}</span>
      </div>
      <div className="timeline-view__list">
        {events.length === 0 ? (
          <p className="timeline-view__empty">No timeline events loaded yet.</p>
        ) : (
          events.map((event) => (
            <button
              type="button"
              key={event.seq}
              className={
                currentSeq === event.seq
                  ? "timeline-view__event is-active"
                  : "timeline-view__event"
              }
              onClick={() => onSelectEvent(event.seq)}
            >
              <span className="timeline-view__event-seq">#{event.seq}</span>
              <span className="timeline-view__event-body">
                <strong>{event.label}</strong>
                <small>{event.type}</small>
              </span>
            </button>
          ))
        )}
      </div>
      {hasMore ? (
        <button type="button" className="control-chip" onClick={onLoadMore}>
          Load more
        </button>
      ) : null}
    </section>
  );
}

import type { TimelineEvent } from "../../gateway/types";
import type { TimelineFilterState } from "../../app/store/arenaSessionStore";

interface TimelineViewProps {
  events: TimelineEvent[];
  filters: TimelineFilterState;
  currentSeq?: number;
  status: "idle" | "loading" | "ready" | "error";
  hasMore: boolean;
  onSelectEvent: (seq: number) => void;
  onLoadMore: () => void;
  onFiltersChange: (filters: TimelineFilterState) => void;
}

export function TimelineView({
  events,
  filters,
  currentSeq,
  status,
  hasMore,
  onSelectEvent,
  onLoadMore,
  onFiltersChange,
}: TimelineViewProps) {
  const visibleEvents = events.filter((event) => {
    if (filters.eventTypes.length > 0 && !filters.eventTypes.includes(event.type)) {
      return false;
    }
    if (filters.severity !== "all" && event.severity !== filters.severity) {
      return false;
    }
    if (filters.humanIntentOnly) {
      const tags = event.tags ?? [];
      if (!tags.includes("human_intent") && !tags.includes("human-intent")) {
        return false;
      }
    }
    return true;
  });
  const eventTypeOptions = Array.from(new Set(events.map((event) => event.type))).sort();

  return (
    <section className="timeline-view" aria-label="Timeline">
      <div className="timeline-view__header">
        <div>
          <p className="eyebrow">Timeline</p>
          <h2>Event stream</h2>
        </div>
        <span className="timeline-view__status">{status}</span>
      </div>
      <div className="control-bar__buttons">
        {eventTypeOptions.map((eventType) => (
          <label key={eventType} className="control-chip">
            <input
              type="checkbox"
              aria-label={eventType}
              checked={filters.eventTypes.includes(eventType)}
              onChange={(event) => {
                const eventTypes = event.target.checked
                  ? [...filters.eventTypes, eventType].sort()
                  : filters.eventTypes.filter((value) => value !== eventType);
                onFiltersChange({
                  ...filters,
                  eventTypes,
                });
              }}
            />
            {eventType}
          </label>
        ))}
        <label className="control-chip">
          <span>Severity</span>
          <select
            aria-label="Severity"
            value={filters.severity}
            onChange={(event) => {
              onFiltersChange({
                ...filters,
                severity: event.target.value as TimelineFilterState["severity"],
              });
            }}
          >
            <option value="all">All</option>
            <option value="info">Info</option>
            <option value="warn">Warn</option>
            <option value="critical">Critical</option>
          </select>
        </label>
        <label className="control-chip">
          <input
            type="checkbox"
            aria-label="Human intent only"
            checked={filters.humanIntentOnly}
            onChange={(event) => {
              onFiltersChange({
                ...filters,
                humanIntentOnly: event.target.checked,
              });
            }}
          />
          Human intent only
        </label>
      </div>
      <div className="timeline-view__list">
        {visibleEvents.length === 0 ? (
          <p className="timeline-view__empty">No timeline events loaded yet.</p>
        ) : (
          visibleEvents.map((event) => (
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

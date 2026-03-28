import type {
  ActionIntentReceipt,
  ObserverRef,
  VisualScene,
  VisualSession,
} from "../../gateway/types";
import { ActionIntentFlow } from "../../app/ActionIntentFlow";

interface SharedSidePanelProps {
  session?: VisualSession;
  scene?: VisualScene;
  latestActionReceipt?: ActionIntentReceipt;
  error?: string;
  isSubmitting?: boolean;
  onObserverChange?: (observer: ObserverRef) => void;
}

interface ScenePanelData {
  chatLog: Array<{
    playerId: string;
    text: string;
  }>;
}

interface ObserverOption {
  value: string;
  label: string;
  observer: ObserverRef;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function readString(value: unknown): string | null {
  return typeof value === "string" && value.trim() !== "" ? value : null;
}

function readScenePanels(scene?: VisualScene): ScenePanelData {
  if (!scene || !isRecord(scene.body) || !isRecord(scene.body.panels)) {
    return { chatLog: [] };
  }

  const panels = scene.body.panels;
  const chatLogRaw = Array.isArray(panels.chatLog) ? panels.chatLog : [];
  const chatLog = chatLogRaw
    .filter(isRecord)
    .map((entry) => ({
      playerId: readString(entry.playerId) ?? readString(entry.player_id) ?? "unknown",
      text: readString(entry.text) ?? "",
    }))
    .filter((entry) => entry.text !== "");

  return { chatLog };
}

function readObserverOptions(session?: VisualSession, scene?: VisualScene): ObserverOption[] {
  const players = new Map<string, string>();

  const addPlayer = (playerId: unknown, playerName?: unknown) => {
    const normalizedId = readString(playerId);
    if (normalizedId === null || players.has(normalizedId)) {
      return;
    }
    players.set(normalizedId, readString(playerName) ?? normalizedId);
  };

  const body = scene && isRecord(scene.body) ? scene.body : undefined;
  if (body) {
    const playersRaw = Array.isArray(body.players) ? body.players : [];
    for (const player of playersRaw) {
      if (!isRecord(player)) {
        continue;
      }
      addPlayer(player.playerId, player.playerName);
    }

    const table = isRecord(body.table) ? body.table : undefined;
    const status = isRecord(body.status) ? body.status : undefined;
    const seatsRaw = table && Array.isArray(table.seats) ? table.seats : [];
    const seatNames = new Map<string, string>();
    for (const seat of seatsRaw) {
      if (!isRecord(seat)) {
        continue;
      }
      const playerId = readString(seat.playerId);
      if (playerId !== null && !seatNames.has(playerId)) {
        seatNames.set(playerId, readString(seat.playerName) ?? playerId);
      }
    }
    if (status) {
      if (table) {
        const privateViewPlayerId = readString(status.privateViewPlayerId);
        if (privateViewPlayerId !== null) {
          addPlayer(privateViewPlayerId, seatNames.get(privateViewPlayerId));
        }
      } else {
        addPlayer(status.activePlayerId);
        addPlayer(status.observerPlayerId);
      }
    }
  }

  addPlayer(session?.scheduling.activeActorId);
  if (session?.observer.observerKind === "player") {
    addPlayer(session.observer.observerId);
  }

  return [
    {
      value: "neutral",
      label: "Neutral observer",
      observer: {
        observerId: "",
        observerKind: "spectator",
      },
    },
    ...Array.from(players.entries()).map(([playerId, playerName]) => ({
      value: `player:${playerId}`,
      label: playerName === playerId ? playerId : `${playerName} (${playerId})`,
      observer: {
        observerId: playerId,
        observerKind: "player" as const,
      },
    })),
  ];
}

function readSelectedObserverValue(session?: VisualSession): string {
  if (
    session?.observer.observerKind === "player" &&
    session.observer.observerId.trim() !== ""
  ) {
    return `player:${session.observer.observerId}`;
  }
  return "neutral";
}

export function SharedSidePanel({
  session,
  scene,
  latestActionReceipt,
  error,
  isSubmitting,
  onObserverChange,
}: SharedSidePanelProps) {
  const panels = readScenePanels(scene);
  const observerOptions = readObserverOptions(session, scene);
  const selectedObserverValue = readSelectedObserverValue(session);

  return (
    <section className="side-panel" aria-label="Shared session context">
      <article className="side-panel__card">
        <h2>Session</h2>
        <p>{session ? `${session.gameId} · ${session.pluginId}` : "Waiting for session..."}</p>
      </article>
      <article className="side-panel__card">
        <h2>Observer</h2>
        {session ? (
          <>
            <label className="side-panel__field">
              <span className="side-panel__field-label">Observer view</span>
              <select
                aria-label="Observer view"
                className="side-panel__select"
                value={selectedObserverValue}
                onChange={(event) => {
                  const nextObserver = observerOptions.find(
                    (option) => option.value === event.target.value,
                  )?.observer;
                  if (nextObserver) {
                    onObserverChange?.(nextObserver);
                  }
                }}
              >
                {observerOptions.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
            <p>{`${session.observer.observerKind} · ${session.observer.observerId || "default"}`}</p>
          </>
        ) : (
          <p>No observer selected</p>
        )}
      </article>
      <article className="side-panel__card">
        <h2>Scene</h2>
        <p>{scene ? `${scene.kind} · seq ${scene.seq}` : "No scene loaded yet"}</p>
      </article>
      <ActionIntentFlow
        error={error}
        isSubmitting={isSubmitting}
        latestReceipt={latestActionReceipt}
      />
      {panels.chatLog.length > 0 ? (
        <article className="side-panel__card">
          <h2>Chat log</h2>
          <ul className="side-panel__list">
            {panels.chatLog.map((entry, index) => (
              <li key={`${entry.playerId}-${index}`}>
                <strong>{entry.playerId}</strong>: {entry.text}
              </li>
            ))}
          </ul>
        </article>
      ) : null}
    </section>
  );
}

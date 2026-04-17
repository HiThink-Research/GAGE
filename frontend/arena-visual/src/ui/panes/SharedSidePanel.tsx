import type {
  ActionIntentReceipt,
  ObserverKind,
  ObserverRef,
  VisualScene,
  VisualSession,
} from "../../gateway/types";
import { useState } from "react";
import { ActionIntentFlow } from "../../app/ActionIntentFlow";
import { isRecord, readString } from "../../lib/sceneReaders";

export const SHARED_SIDE_PANEL_TABS = ["Control", "Players", "Events", "Chat", "Trace"] as const;
export type SharedSidePanelTab = (typeof SHARED_SIDE_PANEL_TABS)[number];

export interface SharedSidePanelControlPanel {
  title: string;
  meta: Array<{
    label: string;
    value: string;
  }>;
  signals: string[];
  operatorHint?: string;
}

interface SharedSidePanelProps {
  session?: VisualSession;
  scene?: VisualScene;
  latestActionReceipt?: ActionIntentReceipt;
  error?: string;
  isSubmitting?: boolean;
  activeTab?: SharedSidePanelTab;
  controlPanel?: SharedSidePanelControlPanel;
  onObserverChange?: (observer: ObserverRef) => void;
  onActiveTabChange?: (tab: SharedSidePanelTab) => void;
  onChatSubmit?: (payload: Record<string, unknown>) => Promise<unknown>;
}

interface ScenePanelData {
  chatLog: Array<{
    playerId: string;
    text: string;
  }>;
  events: Array<{
    label: string;
    detail?: string;
  }>;
  trace: string[];
}

interface ObserverOption {
  value: string;
  label: string;
  observer: ObserverRef;
}

function readScenePanels(scene?: VisualScene): ScenePanelData {
  if (!scene || !isRecord(scene.body) || !isRecord(scene.body.panels)) {
    return { chatLog: [], events: [], trace: [] };
  }

  const panels = scene.body.panels;
  const chatLogRaw = Array.isArray(panels.chatLog) ? panels.chatLog : [];
  const eventsRaw = Array.isArray(panels.events) ? panels.events : [];
  const traceRaw = Array.isArray(panels.trace) ? panels.trace : [];
  const chatLog = chatLogRaw
    .filter(isRecord)
    .map((entry) => ({
      playerId: readString(entry.playerId) ?? readString(entry.player_id) ?? "unknown",
      text: readString(entry.text) ?? "",
    }))
    .filter((entry) => entry.text !== "");
  const events = eventsRaw
    .filter(isRecord)
    .map((entry) => ({
      label: readString(entry.label) ?? "Event",
      detail: readString(entry.detail) ?? undefined,
    }));
  const trace = traceRaw.map((entry) => readString(entry)).filter((entry): entry is string => entry !== null);

  return { chatLog, events, trace };
}

const OBSERVER_KIND_ORDER: ObserverKind[] = ["global", "camera", "spectator", "player"];

function readObserverModes(session?: VisualSession): ObserverKind[] {
  const rawModes = isRecord(session?.capabilities)
    ? session?.capabilities.observerModes
    : undefined;
  const supportedModes = Array.isArray(rawModes)
    ? rawModes.filter(
        (value): value is ObserverKind =>
          value === "global" ||
          value === "camera" ||
          value === "spectator" ||
          value === "player",
      )
    : [];
  const fallbackModes =
    supportedModes.length > 0
      ? supportedModes
      : session?.observer?.observerKind
        ? [session.observer.observerKind]
        : ["spectator"];

  return OBSERVER_KIND_ORDER.filter((mode, index) => fallbackModes.indexOf(mode) !== -1);
}

function readObserverLabel(kind: ObserverKind): string {
  switch (kind) {
    case "global":
      return "Host overview";
    case "camera":
      return "Camera view";
    case "spectator":
      return "Spectator view";
    case "player":
      return "Player view";
  }
}

function readObserverOptions(session?: VisualSession, scene?: VisualScene): ObserverOption[] {
  const players = new Map<string, string>();
  const observerModes = readObserverModes(session);

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
        addPlayer(playerId, readString(seat.playerName) ?? playerId);
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

  const options: ObserverOption[] = observerModes
    .filter((mode) => mode !== "player")
    .map((mode) => ({
      value: mode,
      label: readObserverLabel(mode),
      observer: {
        observerId:
          session?.observer.observerKind === mode ? session.observer.observerId : "",
        observerKind: mode,
      },
    }));

  if (observerModes.includes("player")) {
    options.push(
      ...Array.from(players.entries()).map(([playerId, playerName]) => ({
        value: `player:${playerId}`,
        label: playerName,
        observer: {
          observerId: playerId,
          observerKind: "player" as const,
        },
      })),
    );
  }

  return options;
}

function readSelectedObserverValue(session?: VisualSession): string {
  if (session?.observer.observerKind === "player" && session.observer.observerId.trim() !== "") {
    return `player:${session.observer.observerId}`;
  }
  return session?.observer.observerKind ?? "spectator";
}

function readPlayers(scene?: VisualScene): Array<{ playerId: string; playerName: string }> {
  if (!scene || !isRecord(scene.body)) {
    return [];
  }

  const players = new Map<string, { playerId: string; playerName: string }>();
  const addPlayer = (playerId: unknown, playerName?: unknown) => {
    const normalizedId = readString(playerId);
    if (normalizedId === null || players.has(normalizedId)) {
      return;
    }
    players.set(normalizedId, {
      playerId: normalizedId,
      playerName: readString(playerName) ?? normalizedId,
    });
  };

  const playersRaw = Array.isArray(scene.body.players) ? scene.body.players : [];
  for (const player of playersRaw) {
    if (!isRecord(player)) {
      continue;
    }
    addPlayer(player.playerId, player.playerName);
  }

  const table = isRecord(scene.body.table) ? scene.body.table : undefined;
  const seatsRaw = table && Array.isArray(table.seats) ? table.seats : [];
  for (const seat of seatsRaw) {
    if (!isRecord(seat)) {
      continue;
    }
    addPlayer(seat.playerId, seat.playerName);
  }

  return Array.from(players.values());
}

function readChatPlayerId(session?: VisualSession): string | undefined {
  if (session?.observer.observerKind === "player" && session.observer.observerId.trim() !== "") {
    return session.observer.observerId;
  }
  const activeActorId = readString(session?.scheduling.activeActorId);
  if (activeActorId) {
    return activeActorId;
  }
  return readString(session?.observer.observerId) ?? undefined;
}

export function SharedSidePanel({
  session,
  scene,
  latestActionReceipt,
  error,
  isSubmitting,
  activeTab: controlledActiveTab,
  controlPanel,
  onObserverChange,
  onActiveTabChange,
  onChatSubmit,
}: SharedSidePanelProps) {
  const [uncontrolledActiveTab, setUncontrolledActiveTab] = useState<SharedSidePanelTab>("Players");
  const [chatMessage, setChatMessage] = useState("");
  const [isSubmittingChat, setIsSubmittingChat] = useState(false);
  const panels = readScenePanels(scene);
  const players = readPlayers(scene);
  const observerOptions = readObserverOptions(session, scene);
  const selectedObserverValue = readSelectedObserverValue(session);
  const activeTab = controlledActiveTab ?? uncontrolledActiveTab;
  const showsInternalTabSwitches = controlledActiveTab === undefined;

  const setActiveTab = (tab: SharedSidePanelTab) => {
    if (controlledActiveTab === undefined) {
      setUncontrolledActiveTab(tab);
    }
    onActiveTabChange?.(tab);
  };

  async function handleChatSubmit(): Promise<void> {
    const nextMessage = chatMessage.trim();
    const playerId = readChatPlayerId(session);
    if (nextMessage === "" || !playerId || !onChatSubmit) {
      return;
    }

    setIsSubmittingChat(true);
    try {
      await onChatSubmit({
        playerId,
        text: nextMessage,
      });
      setChatMessage("");
    } catch {
      // Host receipts surface submission outcomes; keep local UI quiet.
    } finally {
      setIsSubmittingChat(false);
    }
  }

  return (
    <section className="side-panel" aria-label="Shared session context">
      {showsInternalTabSwitches ? (
        <div role="tablist" aria-label="Shared side panel sections">
          {SHARED_SIDE_PANEL_TABS.map((tab) => (
            <button
              key={tab}
              type="button"
              role="tab"
              aria-selected={activeTab === tab}
              onClick={() => {
                setActiveTab(tab);
              }}
            >
              {tab}
            </button>
          ))}
        </div>
      ) : null}

      {activeTab === "Control" ? (
        <article className="side-panel__card side-panel__card--control">
          <h2>Control</h2>
          <p>{controlPanel?.title ?? "Waiting for session control state..."}</p>
          {controlPanel?.meta.length ? (
            <div className="side-panel__control-meta">
              {controlPanel.meta.map((entry) => (
                <span key={entry.label}>{entry.label} {entry.value}</span>
              ))}
            </div>
          ) : null}
          {controlPanel?.signals.length ? (
            <div className="side-panel__control-signals">
              {controlPanel.signals.map((signal) => (
                <span className="side-panel__control-chip" key={signal}>
                  {signal}
                </span>
              ))}
            </div>
          ) : null}
          {controlPanel?.operatorHint ? (
            <div
              className="side-panel__control-hint"
              title={controlPanel.operatorHint}
            >
              {controlPanel.operatorHint}
            </div>
          ) : null}
        </article>
      ) : null}

      {activeTab === "Players" ? (
        <article className="side-panel__card">
          <h2>Players</h2>
          <p>{session ? `${session.gameId} · ${session.pluginId}` : "Waiting for session..."}</p>
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
          {players.length > 0 ? (
            <ul className="side-panel__list">
              {players.map((player) => (
                <li key={player.playerId}>
                  <strong>{player.playerName}</strong>
                </li>
              ))}
            </ul>
          ) : null}
        </article>
      ) : null}

      {activeTab === "Events" ? (
        <article className="side-panel__card">
          <h2>Events</h2>
          <p>{scene ? `${scene.kind} · seq ${scene.seq}` : "No scene loaded yet"}</p>
          {panels.events.length > 0 ? (
            <ul className="side-panel__list">
              {panels.events.map((entry, index) => (
                <li key={`${entry.label}-${index}`}>
                  {entry.detail ? `${entry.label}: ${entry.detail}` : entry.label}
                </li>
              ))}
            </ul>
          ) : (
            <p>No host events for this scene.</p>
          )}
        </article>
      ) : null}

      {activeTab === "Chat" ? (
        <article className="side-panel__card">
          <h2>Chat</h2>
          <label className="side-panel__field">
            <span className="side-panel__field-label">Chat message</span>
            <input
              aria-label="Chat message"
              className="side-panel__select"
              value={chatMessage}
              onChange={(event) => {
                setChatMessage(event.target.value);
              }}
            />
          </label>
          <button
            type="button"
            disabled={isSubmittingChat || chatMessage.trim() === ""}
            onClick={() => {
              void handleChatSubmit();
            }}
          >
            Send chat
          </button>
          {panels.chatLog.length > 0 ? (
            <ul className="side-panel__list">
              {panels.chatLog.map((entry, index) => (
                <li key={`${entry.playerId}-${index}`}>
                  <strong>{entry.playerId}</strong>: {entry.text}
                </li>
              ))}
            </ul>
          ) : (
            <p>No chat yet.</p>
          )}
        </article>
      ) : null}

      {activeTab === "Trace" ? (
        <>
          <ActionIntentFlow
            error={error}
            isSubmitting={isSubmitting}
            latestReceipt={latestActionReceipt}
          />
          {panels.trace.length > 0 ? (
            <article className="side-panel__card">
              <h2>Trace</h2>
              <ul className="side-panel__list">
                {panels.trace.map((entry, index) => (
                  <li key={`${entry}-${index}`}>{entry}</li>
                ))}
              </ul>
            </article>
          ) : null}
        </>
      ) : null}
    </section>
  );
}

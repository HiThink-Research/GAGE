import { useEffect, useState, useSyncExternalStore } from "react";
import { Link, useParams } from "react-router-dom";

import { usePlaybackControls } from "../hooks/usePlaybackControls";
import { createArenaSessionStore } from "../store/arenaSessionStore";
import { createArenaGatewayClient } from "../../gateway/client";
import { createArenaMediaResolver } from "../../gateway/media";
import { resolveArenaPlugin } from "../../plugins/registry";
import { useInputBridge } from "../../plugins/sdk/useInputBridge";
import { GlobalControlBar } from "../../ui/controls/GlobalControlBar";
import { ArenaLayout } from "../../ui/layout/ArenaLayout";
import { SharedSidePanel } from "../../ui/panes/SharedSidePanel";
import { TimelineView } from "../../ui/timeline/TimelineView";

function readGatewayBaseUrl(): string {
  const envBaseUrl = import.meta.env.VITE_ARENA_GATEWAY_BASE_URL;
  if (typeof envBaseUrl === "string" && envBaseUrl.trim()) {
    return envBaseUrl.trim();
  }
  return window.location.origin;
}

export function SessionPage() {
  const { sessionId } = useParams();
  const [client] = useState(() =>
    createArenaGatewayClient({ baseUrl: readGatewayBaseUrl() }),
  );
  const [store] = useState(() => createArenaSessionStore(client));
  const [mediaResolver] = useState(() => createArenaMediaResolver(client));
  const snapshot = useSyncExternalStore(
    store.subscribe,
    store.getSnapshot,
    store.getSnapshot,
  );
  const playbackControls = usePlaybackControls(store);
  const plugin =
    snapshot.session !== undefined
      ? resolveArenaPlugin(snapshot.session.pluginId)
      : undefined;
  const inputBridge = useInputBridge({
    latestReceipt: snapshot.latestActionReceipt,
    submitAction: store.submitAction,
    interpreter: plugin?.inputInterpreter,
  });

  useEffect(() => {
    if (!sessionId) {
      return;
    }

    void store.loadSession({ sessionId });
  }, [sessionId, store]);

  useEffect(() => {
    if (
      snapshot.status !== "ready" ||
      snapshot.sceneStatus === "loading" ||
      snapshot.currentSceneSeq === undefined ||
      snapshot.scene?.seq === snapshot.currentSceneSeq
    ) {
      return;
    }

    void store.loadScene({ seq: snapshot.currentSceneSeq }).catch(() => {});
  }, [snapshot.currentSceneSeq, snapshot.scene?.seq, snapshot.sceneStatus, snapshot.status, store]);

  const PluginView = plugin?.render;

  return (
    <main className="app-shell__body">
      <section className="hero-panel">
        <p className="eyebrow">Session Workspace</p>
        <h1>{sessionId ?? "Unknown session"}</h1>
        <p className="hero-copy">
          The host store, gateway client, and plugin registry are now wired.
          The next slices will swap these placeholder surfaces for real game
          renderers.
        </p>
      </section>

      <ArenaLayout
        controls={
          <GlobalControlBar
            playbackMode={snapshot.session?.playback.mode ?? "live_tail"}
            playbackSpeed={snapshot.session?.playback.speed ?? 1}
            disabled={snapshot.status !== "ready" || snapshot.session === undefined}
            scheduling={snapshot.session?.scheduling}
            onPause={() => {
              void playbackControls.pause();
            }}
            onPlayLive={() => {
              void playbackControls.playLive();
            }}
            onReplay={() => {
              void playbackControls.playReplay();
            }}
            onSetSpeed={(speed) => {
              void playbackControls.setSpeed(speed);
            }}
            onStep={(delta) => {
              if (delta > 0) {
                void playbackControls.stepForward();
                return;
              }
              void playbackControls.stepBackward();
            }}
            onSeekEnd={() => {
              void playbackControls.seekEnd();
            }}
            onBackToTail={() => {
              void playbackControls.backToTail();
            }}
          />
        }
        stage={
          snapshot.session && plugin && PluginView ? (
            <PluginView
              session={snapshot.session}
              scene={snapshot.scene}
              latestActionReceipt={inputBridge.latestReceipt}
              submitAction={inputBridge.submitAction}
              submitInput={inputBridge.submitInput}
              mediaSubscribe={mediaResolver.subscribe}
              isFallback={plugin.isFallback}
              requestedPluginId={plugin.requestedPluginId}
            />
          ) : (
            <section className="plugin-stage-card">
              <p className="eyebrow">Host State</p>
              <h2>{snapshot.status === "loading" ? "Loading session..." : "No session loaded"}</h2>
              <p className="plugin-stage-card__copy">
                {snapshot.status === "error"
                  ? snapshot.error ?? "The session failed to load."
                  : "Open a valid session id to populate the visual host."}
              </p>
            </section>
          )
        }
        timeline={
          <TimelineView
            events={snapshot.timeline.events}
            filters={snapshot.timeline.filters}
            currentSeq={snapshot.currentSceneSeq}
            status={snapshot.timeline.status}
            hasMore={snapshot.timeline.hasMore}
            onSelectEvent={(seq) => {
              void playbackControls.selectEvent(seq);
            }}
            onLoadMore={() => {
              void playbackControls.loadMoreTimeline();
            }}
            onFiltersChange={(filters) => {
              store.setTimelineFilters(filters);
            }}
          />
        }
        sidePanel={
          <SharedSidePanel
            session={snapshot.session}
            scene={snapshot.scene}
            latestActionReceipt={snapshot.latestActionReceipt}
            error={inputBridge.error ?? snapshot.error}
            isSubmitting={inputBridge.isSubmitting}
            onObserverChange={(observer) => {
              void store.setObserver(observer).catch(() => {});
            }}
            onChatSubmit={(payload) => store.submitChat(payload)}
          />
        }
      />

      <div className="app-shell__footer">
        <Link className="app-shell__nav-link" to="/">
          Back to host
        </Link>
      </div>
    </main>
  );
}

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
  const inputBridge = useInputBridge({
    latestReceipt: snapshot.latestActionReceipt,
    submitAction: store.submitAction,
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

  const plugin =
    snapshot.session !== undefined
      ? resolveArenaPlugin(snapshot.session.pluginId)
      : undefined;
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
            scheduling={snapshot.session?.scheduling}
            onPause={playbackControls.pause}
            onPlayLive={playbackControls.playLive}
            onReplay={playbackControls.playReplay}
          />
        }
        stage={
          snapshot.session && plugin && PluginView ? (
            <PluginView
              session={snapshot.session}
              scene={snapshot.scene}
              latestActionReceipt={inputBridge.latestReceipt}
              submitAction={inputBridge.submitAction}
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
            currentSeq={snapshot.currentSceneSeq}
            status={snapshot.timeline.status}
            hasMore={snapshot.timeline.hasMore}
            onSelectEvent={(seq) => {
              void playbackControls.selectEvent(seq);
            }}
            onLoadMore={() => {
              void playbackControls.loadMoreTimeline();
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

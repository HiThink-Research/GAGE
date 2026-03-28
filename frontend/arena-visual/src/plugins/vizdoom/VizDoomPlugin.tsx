import type { ArenaPluginRenderProps } from "../sdk/contracts";
import { FrameSurface } from "../frame/FrameSurface";

export function VizDoomPlugin({
  session,
  scene,
  submitAction,
  mediaSubscribe,
}: ArenaPluginRenderProps) {
  return (
    <FrameSurface
      gameLabel="ViZDoom"
      session={session}
      scene={scene}
      submitAction={submitAction}
      mediaSubscribe={mediaSubscribe}
    />
  );
}

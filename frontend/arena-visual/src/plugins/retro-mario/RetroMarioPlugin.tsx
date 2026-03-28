import type { ArenaPluginRenderProps } from "../sdk/contracts";
import { FrameSurface } from "../frame/FrameSurface";

export function RetroMarioPlugin({
  session,
  scene,
  submitAction,
  mediaSubscribe,
}: ArenaPluginRenderProps) {
  return (
    <FrameSurface
      gameLabel="Retro Mario"
      session={session}
      scene={scene}
      submitAction={submitAction}
      mediaSubscribe={mediaSubscribe}
    />
  );
}

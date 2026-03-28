import type { ArenaPluginRenderProps } from "../sdk/contracts";
import { FrameSurface } from "../frame/FrameSurface";

export function PettingZooPlugin({
  session,
  scene,
  submitAction,
  mediaSubscribe,
}: ArenaPluginRenderProps) {
  return (
    <FrameSurface
      gameLabel="PettingZoo"
      session={session}
      scene={scene}
      submitAction={submitAction}
      mediaSubscribe={mediaSubscribe}
    />
  );
}

import type { ArenaPluginRenderProps } from "../sdk/contracts";
import { FrameSurface } from "../frame/FrameSurface";

export function RetroMarioPlugin({
  session,
  scene,
  submitInput,
  mediaSubscribe,
}: ArenaPluginRenderProps<{
  playerId: string;
  actionPayload: Record<string, unknown>;
}>) {
  return (
    <FrameSurface
      gameLabel="Retro Mario"
      session={session}
      scene={scene}
      submitInput={submitInput}
      mediaSubscribe={mediaSubscribe}
    />
  );
}

import type { ArenaPluginRenderProps } from "../sdk/contracts";
import { FrameSurface } from "../frame/FrameSurface";
import { buildRetroMarioKeyboardControls } from "../frame/keyboardControls";

const RETRO_MARIO_KEYBOARD_CONTROLS = buildRetroMarioKeyboardControls();

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
      keyboardControls={RETRO_MARIO_KEYBOARD_CONTROLS}
    />
  );
}

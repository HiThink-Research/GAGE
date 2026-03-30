import type { ArenaPluginRenderProps } from "../sdk/contracts";
import { FrameSurface } from "../frame/FrameSurface";
import { buildVizDoomKeyboardControls } from "../frame/keyboardControls";

const VIZDOOM_KEYBOARD_CONTROLS = buildVizDoomKeyboardControls();

export function VizDoomPlugin({
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
      gameLabel="ViZDoom"
      session={session}
      scene={scene}
      submitInput={submitInput}
      mediaSubscribe={mediaSubscribe}
      keyboardControls={VIZDOOM_KEYBOARD_CONTROLS}
    />
  );
}

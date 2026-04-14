import type { ArenaPluginRenderProps } from "../sdk/contracts";
import { FrameSurface } from "../frame/FrameSurface";
import { buildPettingZooKeyboardControls } from "../frame/keyboardControls";

const PETTINGZOO_KEYBOARD_CONTROLS = buildPettingZooKeyboardControls();

export function PettingZooPlugin({
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
      gameLabel="PettingZoo"
      session={session}
      scene={scene}
      submitInput={submitInput}
      mediaSubscribe={mediaSubscribe}
      keyboardControls={PETTINGZOO_KEYBOARD_CONTROLS}
    />
  );
}

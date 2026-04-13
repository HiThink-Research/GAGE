export interface FrameActionDescriptor {
  id: string;
  label: string;
  payload: Record<string, unknown>;
}

export interface FrameKeyboardControls {
  hint: string;
  watchedKeys: readonly string[];
  resolveAction: (
    actionDescriptors: FrameActionDescriptor[],
    pressedKeys: ReadonlySet<string>,
  ) => FrameActionDescriptor | null;
}

export interface FrameActionDescriptor {
  id: string;
  label: string;
  payload: Record<string, unknown>;
}

export interface FrameOptimisticOffset {
  x: number;
  y: number;
}

export interface FrameKeyboardControls {
  hint: string;
  watchedKeys: readonly string[];
  holdTickMs?: number;
  holdTicksMin?: number;
  holdTicksMax?: number;
  initialHoldTicks?: number;
  heartbeatMs?: number;
  heartbeatHoldTicks?: number;
  resolveAction: (
    actionDescriptors: FrameActionDescriptor[],
    pressedKeys: ReadonlySet<string>,
  ) => FrameActionDescriptor | null;
  resolveActionThrottleMs?: (actionDescriptor: FrameActionDescriptor) => number;
  resolveOptimisticOffset?: (pressedKeys: ReadonlySet<string>) => FrameOptimisticOffset;
}

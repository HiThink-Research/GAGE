import type { FrameActionDescriptor, FrameKeyboardControls } from "./contracts";

function normalizeKeyboardKey(key: string): string {
  const trimmed = key.trim();
  if (trimmed === "" || key === " ") {
    return "space";
  }
  if (trimmed.toLowerCase() === "spacebar") {
    return "space";
  }
  return trimmed.toLowerCase();
}

function buildWatchedKeys(keys: readonly string[]): readonly string[] {
  return keys.map((key) => normalizeKeyboardKey(key));
}

function actionSearchText(actionDescriptor: FrameActionDescriptor): string {
  return `${actionDescriptor.id} ${actionDescriptor.label}`.toLowerCase();
}

function findActionByToken(
  actionDescriptors: FrameActionDescriptor[],
  ...tokens: string[]
): FrameActionDescriptor | null {
  const normalizedTokens = tokens.map((token) => token.toLowerCase());
  for (const actionDescriptor of actionDescriptors) {
    const haystack = actionSearchText(actionDescriptor);
    if (normalizedTokens.every((token) => haystack.includes(token))) {
      return actionDescriptor;
    }
  }
  return null;
}

function findActionById(
  actionDescriptors: FrameActionDescriptor[],
  actionId: string,
): FrameActionDescriptor | null {
  return actionDescriptors.find((actionDescriptor) => actionDescriptor.id === actionId) ?? null;
}

function hasAnyPressed(pressedKeys: ReadonlySet<string>, keys: readonly string[]): boolean {
  return keys.some((key) => pressedKeys.has(normalizeKeyboardKey(key)));
}

export function buildRetroMarioKeyboardControls(): FrameKeyboardControls {
  return {
    hint: "Keyboard: arrows/WASD move, Space/J/Z jump, X/K run, Enter start, Shift/L select.",
    holdTickMs: 16,
    holdTicksMin: 1,
    holdTicksMax: 30,
    initialHoldTicks: 3,
    heartbeatMs: 80,
    heartbeatHoldTicks: 5,
    watchedKeys: buildWatchedKeys([
      "ArrowLeft",
      "ArrowRight",
      "ArrowUp",
      "ArrowDown",
      "a",
      "d",
      "w",
      "s",
      "Space",
      "j",
      "z",
      "c",
      "x",
      "k",
      "Enter",
      "Shift",
      "l",
    ]),
    resolveAction(actionDescriptors, pressedKeys) {
      const left = hasAnyPressed(pressedKeys, ["ArrowLeft", "a"]);
      const right = hasAnyPressed(pressedKeys, ["ArrowRight", "d"]);
      const up = hasAnyPressed(pressedKeys, ["ArrowUp", "w"]);
      const down = hasAnyPressed(pressedKeys, ["ArrowDown", "s"]);
      const jump = hasAnyPressed(pressedKeys, ["Space", "j", "z", "c"]);
      const run = hasAnyPressed(pressedKeys, ["x", "k"]);
      const start = hasAnyPressed(pressedKeys, ["Enter"]);
      const select = hasAnyPressed(pressedKeys, ["Shift", "l"]);

      if (start) {
        return findActionByToken(actionDescriptors, "start");
      }
      if (select) {
        return findActionByToken(actionDescriptors, "select");
      }

      const resolvedLeft = left && !right;
      const resolvedRight = right && !left;
      if (resolvedLeft) {
        return (
          findActionById(actionDescriptors, run && jump ? "left_run_jump" : "") ??
          findActionById(actionDescriptors, jump ? "left_jump" : "") ??
          findActionById(actionDescriptors, run ? "left_run" : "") ??
          findActionById(actionDescriptors, "left") ??
          findActionById(actionDescriptors, "noop")
        );
      }
      if (resolvedRight) {
        return (
          findActionById(actionDescriptors, run && jump ? "right_run_jump" : "") ??
          findActionById(actionDescriptors, jump ? "right_jump" : "") ??
          findActionById(actionDescriptors, run ? "right_run" : "") ??
          findActionById(actionDescriptors, "right") ??
          findActionById(actionDescriptors, "noop")
        );
      }
      if (up) {
        return findActionById(actionDescriptors, "up") ?? findActionById(actionDescriptors, "noop");
      }
      if (down) {
        return findActionById(actionDescriptors, "down") ?? findActionById(actionDescriptors, "noop");
      }
      if (jump) {
        return findActionById(actionDescriptors, "jump") ?? findActionById(actionDescriptors, "noop");
      }
      if (run) {
        return findActionById(actionDescriptors, "run") ?? findActionById(actionDescriptors, "noop");
      }
      return findActionById(actionDescriptors, "noop");
    },
    resolveOptimisticOffset(pressedKeys) {
      const left = hasAnyPressed(pressedKeys, ["ArrowLeft", "a"]);
      const right = hasAnyPressed(pressedKeys, ["ArrowRight", "d"]);
      if (left && !right) {
        return { x: -3, y: 0 };
      }
      if (right && !left) {
        return { x: 3, y: 0 };
      }
      return { x: 0, y: 0 };
    },
  };
}

export function buildPettingZooKeyboardControls(): FrameKeyboardControls {
  return {
    hint: "Keyboard: arrows move, Space/J/Z fires.",
    holdTickMs: 16,
    holdTicksMin: 1,
    holdTicksMax: 4,
    initialHoldTicks: 1,
    heartbeatMs: 33,
    heartbeatHoldTicks: 1,
    watchedKeys: buildWatchedKeys([
      "ArrowLeft",
      "ArrowRight",
      "ArrowUp",
      "ArrowDown",
      "a",
      "d",
      "w",
      "s",
      "Space",
      "j",
      "z",
    ]),
    resolveAction(actionDescriptors, pressedKeys) {
      const left = hasAnyPressed(pressedKeys, ["ArrowLeft", "a"]);
      const right = hasAnyPressed(pressedKeys, ["ArrowRight", "d"]);
      const up = hasAnyPressed(pressedKeys, ["ArrowUp", "w"]);
      const down = hasAnyPressed(pressedKeys, ["ArrowDown", "s"]);
      const fire = hasAnyPressed(pressedKeys, ["Space", "j", "z"]);

      if (fire && left && !right) {
        return (
          findActionByToken(actionDescriptors, "left", "fire") ??
          findActionByToken(actionDescriptors, "fire")
        );
      }
      if (fire && right && !left) {
        return (
          findActionByToken(actionDescriptors, "right", "fire") ??
          findActionByToken(actionDescriptors, "fire")
        );
      }
      if (fire) {
        return findActionByToken(actionDescriptors, "fire");
      }
      if (left && !right) {
        return findActionByToken(actionDescriptors, "left");
      }
      if (right && !left) {
        return findActionByToken(actionDescriptors, "right");
      }
      if (up && !down) {
        return findActionByToken(actionDescriptors, "up");
      }
      if (down && !up) {
        return findActionByToken(actionDescriptors, "down");
      }
      return null;
    },
    resolveActionThrottleMs(actionDescriptor) {
      return actionSearchText(actionDescriptor).includes("fire") ? 500 : 0;
    },
  };
}

export function buildVizDoomKeyboardControls(): FrameKeyboardControls {
  return {
    hint: "Keyboard: W or Up moves, A/Left and D/Right turn, Space or J fires.",
    watchedKeys: buildWatchedKeys([
      "ArrowLeft",
      "ArrowRight",
      "ArrowUp",
      "a",
      "d",
      "w",
      "Space",
      "Enter",
      "j",
    ]),
    resolveAction(actionDescriptors, pressedKeys) {
      if (hasAnyPressed(pressedKeys, ["Space", "Enter", "j"])) {
        return (
          findActionByToken(actionDescriptors, "fire") ??
          findActionByToken(actionDescriptors, "attack") ??
          findActionById(actionDescriptors, "0")
        );
      }
      if (hasAnyPressed(pressedKeys, ["ArrowUp", "w"])) {
        return (
          findActionByToken(actionDescriptors, "forward") ??
          findActionById(actionDescriptors, "0")
        );
      }
      if (hasAnyPressed(pressedKeys, ["ArrowLeft", "a"])) {
        return (
          findActionByToken(actionDescriptors, "left") ??
          findActionById(actionDescriptors, "0")
        );
      }
      if (hasAnyPressed(pressedKeys, ["ArrowRight", "d"])) {
        return (
          findActionByToken(actionDescriptors, "right") ??
          findActionById(actionDescriptors, "0")
        );
      }
      return findActionById(actionDescriptors, "0");
    },
  };
}

export { normalizeKeyboardKey };

import { useState } from "react";

import type { ActionIntentReceipt } from "../../gateway/types";
import type { ActionIntent, InputInterpreter } from "./input";

interface UseInputBridgeOptions<
  TDeviceEvent = never,
  TIntent extends ActionIntent = ActionIntent,
> {
  latestReceipt?: ActionIntentReceipt;
  submitAction: (payload: TIntent) => Promise<ActionIntentReceipt>;
  interpreter?: InputInterpreter<TDeviceEvent, TIntent>;
}

export function useInputBridge<
  TDeviceEvent = never,
  TIntent extends ActionIntent = ActionIntent,
>({
  latestReceipt,
  submitAction,
  interpreter,
}: UseInputBridgeOptions<TDeviceEvent, TIntent>) {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string>();

  async function submit(payload: TIntent): Promise<void> {
    setIsSubmitting(true);
    setError(undefined);
    try {
      await submitAction(payload);
    } catch (caughtError) {
      const message =
        caughtError instanceof Error ? caughtError.message : "Action submission failed.";
      setError(message);
      await Promise.resolve();
      throw caughtError;
    } finally {
      setIsSubmitting(false);
    }
  }

  async function submitInput(event: TDeviceEvent): Promise<void> {
    if (!interpreter) {
      throw new Error("Plugin input interpreter is not available.");
    }
    await submit(interpreter.interpret(event));
  }

  return {
    latestReceipt,
    isSubmitting,
    error,
    submitAction: submit,
    submitInput,
  };
}

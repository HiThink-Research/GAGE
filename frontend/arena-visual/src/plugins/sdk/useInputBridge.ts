import { useState } from "react";

import type { ActionIntentReceipt } from "../../gateway/types";

interface UseInputBridgeOptions {
  latestReceipt?: ActionIntentReceipt;
  submitAction: (payload: Record<string, unknown>) => Promise<ActionIntentReceipt>;
}

export function useInputBridge({
  latestReceipt,
  submitAction,
}: UseInputBridgeOptions) {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string>();

  async function submit(payload: Record<string, unknown>): Promise<void> {
    setIsSubmitting(true);
    setError(undefined);
    try {
      await submitAction(payload);
    } catch (caughtError) {
      setError(
        caughtError instanceof Error ? caughtError.message : "Action submission failed.",
      );
      throw caughtError;
    } finally {
      setIsSubmitting(false);
    }
  }

  return {
    latestReceipt,
    isSubmitting,
    error,
    submitAction: submit,
  };
}

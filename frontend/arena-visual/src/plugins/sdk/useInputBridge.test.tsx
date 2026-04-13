import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { ActionIntentReceipt } from "../../gateway/types";
import { createInputInterpreter } from "./input";
import { useInputBridge } from "./useInputBridge";

describe("useInputBridge", () => {
  it("routes a device event through the interpreter into submitAction", async () => {
    const receipt: ActionIntentReceipt = {
      intentId: "intent-1",
      state: "accepted",
      relatedEventSeq: 42,
    };
    const submitAction = vi.fn().mockResolvedValue(receipt);
    const interpreter = createInputInterpreter<{
      playerId: string;
      coord: string;
    }>(({ playerId, coord }) => ({
      playerId,
      action: { move: coord },
    }));

    const { result } = renderHook(() =>
      useInputBridge({
        latestReceipt: receipt,
        submitAction,
        interpreter,
      }),
    );

    await act(async () => {
      await result.current.submitInput({
        playerId: "player_0",
        coord: "B2",
      });
    });

    expect(submitAction).toHaveBeenCalledWith({
      playerId: "player_0",
      action: { move: "B2" },
    });
    expect(result.current.latestReceipt).toEqual(receipt);
    expect(result.current.error).toBeUndefined();
    expect(result.current.isSubmitting).toBe(false);
  });

  it("captures interpreter-backed submit failures without changing receipt state", async () => {
    const receipt: ActionIntentReceipt = {
      intentId: "intent-2",
      state: "pending",
    };
    const submitAction = vi.fn().mockRejectedValue(new Error("network unavailable"));
    const interpreter = createInputInterpreter<{ playerId: string; move: string }>(
      ({ playerId, move }) => ({
        playerId,
        action: { move },
      }),
    );

    const { result } = renderHook(() =>
      useInputBridge({
        latestReceipt: receipt,
        submitAction,
        interpreter,
      }),
    );

    let thrownError: Error | undefined;
    await act(async () => {
      try {
        await result.current.submitInput({ playerId: "player_1", move: "pass" });
      } catch (error) {
        thrownError = error as Error;
      }
    });

    expect(thrownError?.message).toBe("network unavailable");
    expect(result.current.latestReceipt).toEqual(receipt);
    await waitFor(() => expect(result.current.error).toBe("network unavailable"));
    expect(result.current.isSubmitting).toBe(false);
  });
});

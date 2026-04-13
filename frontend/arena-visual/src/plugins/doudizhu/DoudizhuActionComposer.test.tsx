import { describe, expect, it } from "vitest";

import {
  matchLegalActionForSelection,
  resolveHintAction,
  selectHandIndexesForAction,
} from "./doudizhuCards";

describe("doudizhu action helpers", () => {
  it("matches a legal action from selected duplicate cards", () => {
    expect(
      matchLegalActionForSelection(["pass", "6JJJ", "QQ", "R"], ["6", "J", "J", "J"]),
    ).toBe("6JJJ");
    expect(
      matchLegalActionForSelection(["pass", "BR", "QQ"], ["BlackJoker", "RedJoker"]),
    ).toBe("BR");
  });

  it("finds the card indexes needed for a hinted action", () => {
    expect(selectHandIndexesForAction(["6", "J", "J", "J", "Q"], "6JJJ")).toEqual([0, 1, 2, 3]);
    expect(selectHandIndexesForAction(["Q", "Q", "R"], "QQ")).toEqual([0, 1]);
  });

  it("prefers the first playable non-pass action as the hint", () => {
    expect(resolveHintAction(["pass", "6JJJ", "QQ"], ["6", "J", "J", "J", "Q", "Q"])).toBe("6JJJ");
    expect(resolveHintAction(["pass"], ["6", "J"])).toBeNull();
  });
});

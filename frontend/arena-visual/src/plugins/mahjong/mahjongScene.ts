import type {
  TableDiscardLane,
  TableMeldGroup,
  TableSceneData,
  TableSeat,
} from "../table/TableLayout";

export interface MahjongHandState {
  mainTiles: string[];
  drawTile: string | null;
}

export function isMahjongTileCode(value: string): boolean {
  const normalized = value.trim();
  return /^[BCD][1-9]$/i.test(normalized) || /^(East|South|West|North|Green|Red|White)$/i.test(normalized);
}

export function normalizeTileCode(value: string): string {
  const trimmed = value.trim();
  if (/^[BCD][1-9]$/i.test(trimmed)) {
    return `${trimmed.charAt(0).toUpperCase()}${trimmed.slice(1)}`;
  }
  if (/^(East|South|West|North|Green|Red|White)$/i.test(trimmed)) {
    return trimmed.charAt(0).toUpperCase() + trimmed.slice(1).toLowerCase();
  }
  return trimmed;
}

export function parseMahjongMeldNote(note: string): TableMeldGroup {
  const tiles = Array.from(
    note.matchAll(/\b([BCD][1-9]|East|South|West|North|Green|Red|White)\b/gi),
    (match) => normalizeTileCode(match[1] ?? match[0]),
  );
  const prefix = note.trim().split(/\s+/)[0]?.toLowerCase() ?? "";

  if (tiles.length === 1 && (prefix === "pong" || prefix === "peng")) {
    return {
      type: "pong",
      label: note,
      tiles: [tiles[0], tiles[0], tiles[0]],
    };
  }
  if (tiles.length === 1 && (prefix === "kong" || prefix === "gang")) {
    return {
      type: "kong",
      label: note,
      tiles: [tiles[0], tiles[0], tiles[0], tiles[0]],
    };
  }
  if (tiles.length >= 3 && (prefix === "chow" || prefix === "chi")) {
    return {
      type: "chow",
      label: note,
      tiles: tiles.slice(0, 3),
    };
  }
  return {
    type: null,
    label: note,
    tiles,
  };
}

export function resolveMahjongMeldGroups(seat: TableSeat): TableMeldGroup[] {
  if (seat.meldGroups.length > 0) {
    return seat.meldGroups.map((group) => ({
      type: group.type,
      label: group.label,
      tiles: group.tiles.map(normalizeTileCode),
    }));
  }
  return seat.publicNotes.map(parseMahjongMeldNote);
}

export function splitMahjongSeatHand(seat: TableSeat): MahjongHandState {
  const normalizedCards = seat.hand.cards.map(normalizeTileCode);
  const explicitDrawTile = seat.drawTile ? normalizeTileCode(seat.drawTile) : seat.hand.drawTile ? normalizeTileCode(seat.hand.drawTile) : null;
  if (explicitDrawTile) {
    const drawIndex = normalizedCards.lastIndexOf(explicitDrawTile);
    if (drawIndex >= 0) {
      return {
        mainTiles: normalizedCards.filter((_, index) => index !== drawIndex),
        drawTile: explicitDrawTile,
      };
    }
    return {
      mainTiles: normalizedCards,
      drawTile: explicitDrawTile,
    };
  }
  if (normalizedCards.length > 1 && normalizedCards.length % 3 === 2) {
    return {
      mainTiles: normalizedCards.slice(0, -1),
      drawTile: normalizedCards.at(-1) ?? null,
    };
  }
  return {
    mainTiles: normalizedCards,
    drawTile: null,
  };
}

export function uniqueMahjongActionTexts(actionTexts: string[]): string[] {
  const seen = new Set<string>();
  return actionTexts.filter((actionText) => {
    const normalized = normalizeTileCode(actionText);
    if (seen.has(normalized)) {
      return false;
    }
    seen.add(normalized);
    return true;
  });
}

export function resolveMahjongDiscardLanes(tableScene: TableSceneData): TableDiscardLane[] {
  if (tableScene.table.center.discardLanes.length > 0) {
    return tableScene.table.center.discardLanes.map((lane) => ({
      seatId: lane.seatId,
      playerId: lane.playerId,
      cards: lane.cards.map(normalizeTileCode),
    }));
  }

  const fallbackCards = tableScene.table.center.cards.map(normalizeTileCode);
  if (fallbackCards.length === 0) {
    return [];
  }

  const anchoredSeatIds = tableScene.table.seats.map((seat) => seat.seatId);
  return anchoredSeatIds.map((seatId, index) => ({
    seatId,
    playerId: seatId,
    cards: fallbackCards.filter((_, cardIndex) => cardIndex % anchoredSeatIds.length === index),
  }));
}

import type { VisualScene, VisualSession } from "../../gateway/types";
import {
  isRecord,
  readBoolean,
  readNumber,
  readString,
} from "../../lib/sceneReaders";

export interface BoardCell {
  coord: string;
  row: number;
  col: number;
  occupant: string | null;
  playerId: string | null;
  playerName: string | null;
  isLastMove: boolean;
  isWinningCell: boolean;
  isLegalAction: boolean;
}

export interface BoardPlayer {
  playerId: string;
  playerName: string;
  token: string;
}

export interface BoardStatus {
  activePlayerId: string | null;
  observerPlayerId: string | null;
  moveCount: number;
  lastMove: string | null;
  winningLine: string[];
}

export interface BoardSceneData {
  board: {
    size: number;
    coordScheme: string;
    cells: BoardCell[];
  };
  players: BoardPlayer[];
  status: BoardStatus;
}

export function readBoardScene(scene?: VisualScene): BoardSceneData | null {
  if (!scene || scene.kind !== "board" || !isRecord(scene.body)) {
    return null;
  }

  const body = scene.body;
  const board = body.board;
  if (!isRecord(board) || !Array.isArray(board.cells)) {
    return null;
  }

  const cells: BoardCell[] = board.cells
    .filter(isRecord)
    .map((cell, index) => ({
      coord: readString(cell.coord) ?? `cell-${index}`,
      row: readNumber(cell.row),
      col: readNumber(cell.col),
      occupant: readString(cell.occupant),
      playerId: readString(cell.playerId),
      playerName: readString(cell.playerName),
      isLastMove: readBoolean(cell.isLastMove),
      isWinningCell: readBoolean(cell.isWinningCell),
      isLegalAction: readBoolean(cell.isLegalAction),
    }));

  const playersRaw = Array.isArray(body.players) ? body.players : [];
  const players: BoardPlayer[] = playersRaw
    .filter(isRecord)
    .map((player) => ({
      playerId: readString(player.playerId) ?? "",
      playerName: readString(player.playerName) ?? "",
      token: readString(player.token) ?? "",
    }))
    .filter((player) => player.playerId !== "");

  const statusRaw = isRecord(body.status) ? body.status : {};
  const status: BoardStatus = {
    activePlayerId: readString(statusRaw.activePlayerId),
    observerPlayerId: readString(statusRaw.observerPlayerId),
    moveCount: readNumber(statusRaw.moveCount),
    lastMove: readString(statusRaw.lastMove),
    winningLine: Array.isArray(statusRaw.winningLine)
      ? statusRaw.winningLine.map(readString).filter((coord): coord is string => coord !== null)
      : [],
  };

  const size = Math.max(1, Math.floor(readNumber(board.size)));
  const coordScheme = readString(board.coordScheme) ?? "A1";

  return {
    board: {
      size,
      coordScheme,
      cells,
    },
    players,
    status,
  };
}

export function readLegalCoords(scene?: VisualScene): Set<string> {
  const coords = new Set<string>();
  if (!scene) {
    return coords;
  }

  for (const action of scene.legalActions) {
    if (!isRecord(action)) {
      continue;
    }
    const coord = readString(action.coord);
    if (coord) {
      coords.add(coord);
    }
  }

  return coords;
}

export function resolveBoardActorId(
  session: VisualSession,
  scene: VisualScene | undefined,
  boardScene: BoardSceneData,
): string | null {
  if (
    session.observer.observerKind === "player" &&
    typeof session.observer.observerId === "string" &&
    session.observer.observerId.trim() !== ""
  ) {
    return session.observer.observerId;
  }

  const fallbackActorId =
    boardScene.status.activePlayerId ??
    (typeof scene?.activePlayerId === "string" ? scene.activePlayerId : null) ??
    session.scheduling.activeActorId ??
    null;
  return readString(fallbackActorId);
}

export function formatBoardActorLabel(
  session: VisualSession,
  _players: BoardPlayer[],
  resolvedActorId: string | null,
): string {
  if (
    session.observer.observerKind === "player" &&
    typeof session.observer.observerId === "string" &&
    session.observer.observerId.trim() !== ""
  ) {
    return `Observer: ${session.observer.observerId}`;
  }

  if (resolvedActorId) {
    return `Active player: ${resolvedActorId}`;
  }

  return "Active player: waiting";
}

import type { VisualScene } from "../../gateway/types";

const richScene: VisualScene = {
  sceneId: "doudizhu:seq:16",
  gameId: "doudizhu",
  pluginId: "arena.visualization.doudizhu.table_v1",
  kind: "table",
  tsMs: 1016,
  seq: 16,
  phase: "live",
  activePlayerId: "player_0",
  legalActions: [
    { id: "pass", label: "pass", text: "pass" },
    { id: "6JJJ", label: "6JJJ", text: "6JJJ" },
    { id: "QQ", label: "QQ", text: "QQ" },
    { id: "R", label: "R", text: "R" },
  ],
  summary: {
    eventType: "decision_window_open",
    eventLabel: "decision_window_open",
    seatCount: 3,
  },
  body: {
    table: {
      layout: "three-seat",
      seats: [
        {
          seatId: "bottom",
          playerId: "player_0",
          playerName: "Player 0",
          role: "landlord",
          isActive: true,
          isObserver: true,
          playedCards: ["H5"],
          publicNotes: [],
          hand: {
            isVisible: true,
            cards: ["S6", "HJ", "CJ", "DJ", "SQ", "HQ", "RedJoker"],
            maskedCount: 0,
          },
        },
        {
          seatId: "left",
          playerId: "player_1",
          playerName: "Player 1",
          role: "peasant",
          isActive: false,
          isObserver: false,
          playedCards: ["C9"],
          publicNotes: [],
          hand: { isVisible: false, cards: [], maskedCount: 8 },
        },
        {
          seatId: "right",
          playerId: "player_2",
          playerName: "Player 2",
          role: "peasant",
          isActive: false,
          isObserver: false,
          playedCards: ["D10"],
          publicNotes: [],
          hand: { isVisible: false, cards: [], maskedCount: 8 },
        },
      ],
      center: {
        label: "Seen cards",
        cards: ["3", "4", "5"],
        history: ["player_0: H5", "player_1: C9", "player_2: D10"],
      },
    },
    status: {
      activePlayerId: "player_0",
      observerPlayerId: "player_0",
      moveCount: 9,
      lastMove: "10",
      landlordId: "player_0",
    },
    panels: {
      chatLog: [
        { playerId: "player_1", text: "watch this" },
        { playerId: "player_0", text: "I have rocket" },
        { playerId: "player_1", text: "last warning" },
      ],
    },
  },
};

export default richScene;

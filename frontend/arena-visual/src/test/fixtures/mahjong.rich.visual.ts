import type { VisualScene } from "../../gateway/types";

const richScene: VisualScene = {
  sceneId: "mahjong:seq:21",
  gameId: "mahjong",
  pluginId: "arena.visualization.mahjong.table_v1",
  kind: "table",
  tsMs: 2021,
  seq: 21,
  phase: "live",
  activePlayerId: "east",
  legalActions: [
    { id: "B1", label: "B1", text: "B1" },
    { id: "Red", label: "Red", text: "Red" },
    { id: "Pong", label: "Pong", text: "Pong" },
    { id: "Hu", label: "Hu", text: "Hu" },
  ],
  summary: {
    eventType: "decision_window_open",
    eventLabel: "decision_window_open",
    seatCount: 4,
  },
  body: {
    table: {
      layout: "four-seat",
      seats: [
        {
          seatId: "east",
          playerId: "east",
          playerName: "East",
          role: null,
          isActive: true,
          isObserver: true,
          playedCards: [],
          publicNotes: ["Chow B2-B3-B4"],
          meldGroups: [
            {
              type: "chow",
              label: "Chow B2-B3-B4",
              tiles: ["B2", "B3", "B4"],
            },
          ],
          drawTile: "Red",
          hand: {
            isVisible: true,
            cards: [
              "B1",
              "B2",
              "B3",
              "B4",
              "B5",
              "B9",
              "C3",
              "C4",
              "C5",
              "D2",
              "D3",
              "D4",
              "Green",
              "Red",
            ],
            maskedCount: 0,
          },
        },
        {
          seatId: "south",
          playerId: "south",
          playerName: "South",
          role: null,
          isActive: false,
          isObserver: false,
          playedCards: [],
          publicNotes: ["Pong C3"],
          meldGroups: [
            {
              type: "pong",
              label: "Pong C3",
              tiles: ["C3", "C3", "C3"],
            },
          ],
          hand: { isVisible: false, cards: [], maskedCount: 13 },
        },
        {
          seatId: "west",
          playerId: "west",
          playerName: "West",
          role: null,
          isActive: false,
          isObserver: false,
          playedCards: [],
          publicNotes: [],
          hand: { isVisible: false, cards: [], maskedCount: 13 },
        },
        {
          seatId: "north",
          playerId: "north",
          playerName: "North",
          role: null,
          isActive: false,
          isObserver: false,
          playedCards: [],
          publicNotes: ["Kong D5"],
          meldGroups: [
            {
              type: "kong",
              label: "Kong D5",
              tiles: ["D5", "D5", "D5", "D5"],
            },
          ],
          hand: { isVisible: false, cards: [], maskedCount: 13 },
        },
      ],
      center: {
        label: "Discards",
        cards: ["B1", "C1", "D1", "East", "B9", "C9", "D9", "White"],
        history: [],
        discardLanes: [
          { seatId: "east", playerId: "east", cards: ["B1", "B9"] },
          { seatId: "south", playerId: "south", cards: ["C1", "C9"] },
          { seatId: "west", playerId: "west", cards: ["D1", "D9"] },
          { seatId: "north", playerId: "north", cards: ["East", "White"] },
        ],
      },
    },
    status: {
      activePlayerId: "east",
      observerPlayerId: "east",
      moveCount: 11,
      lastMove: "East",
      lastDiscard: {
        playerId: "north",
        tile: "East",
        isTsumogiri: false,
      },
      landlordId: null,
    },
    panels: {
      chatLog: [{ playerId: "south", text: "pon" }],
    },
  },
};

export default richScene;

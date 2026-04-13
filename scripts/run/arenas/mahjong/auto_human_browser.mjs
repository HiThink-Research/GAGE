#!/usr/bin/env node
import { chromium } from "playwright";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

const TILE_PATTERN = /^(?:[BCD][1-9]|East|South|West|North|Green|Red|White)$/i;

function parseArgs(argv) {
  const args = {
    baseUrl: "http://127.0.0.1:5806",
    sessionId: "mahjong_match_0001",
    runId: "",
    observerId: "east",
    timeoutMs: 90 * 60 * 1000,
    pollMs: 1000,
    headed: false,
    strategy: "active",
    outDir: "",
  };
  for (let index = 0; index < argv.length; index += 1) {
    const item = argv[index];
    const readValue = () => {
      const value = argv[index + 1];
      if (value === undefined || value.startsWith("--")) {
        throw new Error(`Missing value for ${item}`);
      }
      index += 1;
      return value;
    };
    if (item === "--base-url") {
      args.baseUrl = readValue();
    } else if (item === "--session-id") {
      args.sessionId = readValue();
    } else if (item === "--run-id") {
      args.runId = readValue();
    } else if (item === "--observer-id") {
      args.observerId = readValue();
    } else if (item === "--out-dir") {
      args.outDir = readValue();
    } else if (item === "--timeout-ms") {
      args.timeoutMs = Number(readValue());
    } else if (item === "--timeout-minutes") {
      args.timeoutMs = Number(readValue()) * 60 * 1000;
    } else if (item === "--poll-ms") {
      args.pollMs = Number(readValue());
    } else if (item === "--strategy") {
      args.strategy = readValue();
    } else if (item === "--headed") {
      args.headed = true;
    } else if (item === "--headless") {
      args.headed = false;
    } else if (item === "--help" || item === "-h") {
      printHelp();
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${item}`);
    }
  }
  if (!args.runId.trim()) {
    throw new Error("--run-id is required");
  }
  if (!Number.isFinite(args.timeoutMs) || args.timeoutMs <= 0) {
    throw new Error("--timeout-ms/--timeout-minutes must be positive");
  }
  if (!Number.isFinite(args.pollMs) || args.pollMs <= 0) {
    throw new Error("--poll-ms must be positive");
  }
  if (!["active", "first", "pass-first"].includes(args.strategy)) {
    throw new Error("--strategy must be active, first, or pass-first");
  }
  if (!args.outDir.trim()) {
    args.outDir = path.resolve("runs", args.runId, "browser_auto_full");
  }
  return args;
}

function printHelp() {
  console.log(`Usage: node auto_human_browser.mjs --run-id RUN_ID [options]

Options:
  --base-url URL          Arena visual server base URL. Default: http://127.0.0.1:5806
  --session-id ID         Session/sample id. Default: mahjong_match_0001
  --observer-id ID        Human player id. Default: east
  --out-dir DIR           Output directory. Default: runs/RUN_ID/browser_auto_full
  --timeout-minutes N     Full-game timeout. Default: 90
  --poll-ms N             Poll interval. Default: 1000
  --strategy NAME         active, first, or pass-first. Default: active
  --headed                Open a headed Chromium window
  --headless              Use headless Chromium
`);
}

function normalizeBaseUrl(value) {
  return value.endsWith("/") ? value.slice(0, -1) : value;
}

function buildViewerUrl({ baseUrl, sessionId, runId }) {
  const url = new URL(`/sessions/${encodeURIComponent(sessionId)}`, normalizeBaseUrl(baseUrl));
  url.searchParams.set("run_id", runId);
  return url.toString();
}

function buildSessionPath({ sessionId, runId, observerId }) {
  const query = new URLSearchParams({
    run_id: runId,
    observer_kind: "player",
    observer_id: observerId,
  });
  return `/arena_visual/sessions/${encodeURIComponent(sessionId)}?${query.toString()}`;
}

function buildScenePath({ sessionId, runId, observerId, seq }) {
  const query = new URLSearchParams({
    seq: String(seq),
    run_id: runId,
    observer_kind: "player",
    observer_id: observerId,
  });
  return `/arena_visual/sessions/${encodeURIComponent(sessionId)}/scene?${query.toString()}`;
}

function buildActionPath({ sessionId, runId }) {
  const query = new URLSearchParams({ run_id: runId });
  return `/arena_visual/sessions/${encodeURIComponent(sessionId)}/actions?${query.toString()}`;
}

async function fetchJsonFromPage(page, pathName) {
  return page.evaluate(async (targetPath) => {
    const response = await fetch(targetPath, { cache: "no-store" });
    const text = await response.text();
    let data = null;
    try {
      data = text ? JSON.parse(text) : null;
    } catch {
      data = { raw: text };
    }
    if (!response.ok) {
      throw new Error(`HTTP ${response.status} for ${targetPath}: ${text.slice(0, 400)}`);
    }
    return data;
  }, pathName);
}

async function readTurn(page, args) {
  const session = await fetchJsonFromPage(page, buildSessionPath(args));
  const seq = session?.playback?.cursorEventSeq ?? session?.timeline?.tailSeq ?? null;
  const scene = Number.isFinite(seq)
    ? await fetchJsonFromPage(page, buildScenePath({ ...args, seq }))
    : null;
  return { session, scene, seq };
}

function readLegalActions(scene) {
  return (scene?.legalActions ?? [])
    .map((item) => String(item?.text ?? item?.label ?? item?.id ?? "").trim())
    .filter(Boolean);
}

function isTileAction(actionText) {
  return TILE_PATTERN.test(String(actionText).trim());
}

function normalizeTileCode(value) {
  const trimmed = String(value).trim();
  if (/^[BCD][1-9]$/i.test(trimmed)) {
    return `${trimmed.charAt(0).toUpperCase()}${trimmed.slice(1)}`;
  }
  if (/^(East|South|West|North|Green|Red|White)$/i.test(trimmed)) {
    return trimmed.charAt(0).toUpperCase() + trimmed.slice(1).toLowerCase();
  }
  return trimmed;
}

function findBottomSeat(scene) {
  const seats = scene?.body?.table?.seats;
  if (!Array.isArray(seats)) {
    return null;
  }
  return seats.find((seat) => seat?.isObserver) ?? seats.find((seat) => seat?.playerId === scene?.body?.status?.observerPlayerId) ?? null;
}

function resolvePreferredTile(scene, legalActions) {
  const legalTiles = legalActions.filter(isTileAction).map(normalizeTileCode);
  if (legalTiles.length === 0) {
    return undefined;
  }
  const seat = findBottomSeat(scene);
  const drawTile = normalizeTileCode(seat?.drawTile ?? seat?.hand?.drawTile ?? "");
  if (drawTile && legalTiles.includes(drawTile)) {
    return drawTile;
  }
  const visibleCards = Array.isArray(seat?.hand?.cards) ? seat.hand.cards.map(normalizeTileCode) : [];
  const visibleLegal = visibleCards.filter((tile) => legalTiles.includes(tile));
  if (visibleLegal.length > 0) {
    return visibleLegal.at(-1);
  }
  return legalTiles[0];
}

function chooseAction(scene, legalActions, strategy) {
  if (legalActions.length === 0) {
    return undefined;
  }
  if (strategy === "first") {
    return normalizeTileCode(legalActions[0]);
  }
  const byLower = new Map(legalActions.map((action) => [action.toLowerCase(), action]));
  if (byLower.has("hu")) {
    return byLower.get("hu");
  }
  if (strategy === "pass-first") {
    for (const candidate of ["pass", "skip", "stand"]) {
      if (byLower.has(candidate)) {
        return byLower.get(candidate);
      }
    }
  }
  const tile = resolvePreferredTile(scene, legalActions);
  if (tile) {
    return tile;
  }
  for (const candidate of ["stand", "pass", "skip"]) {
    if (byLower.has(candidate)) {
      return byLower.get(candidate);
    }
  }
  for (const candidate of ["gong", "kong", "pong", "chow"]) {
    if (byLower.has(candidate)) {
      return byLower.get(candidate);
    }
  }
  return normalizeTileCode(legalActions[0]);
}

function escapeRegex(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

async function waitForUsable(locator, timeoutMs = 5000) {
  const deadlineAt = Date.now() + timeoutMs;
  let lastError = null;
  while (Date.now() < deadlineAt) {
    try {
      if ((await locator.count()) > 0 && (await locator.first().isVisible()) && (await locator.first().isEnabled())) {
        return locator.first();
      }
    } catch (error) {
      lastError = error;
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  if (lastError) {
    throw lastError;
  }
  return null;
}

async function waitForUiSeq(page, seq, timeoutMs = 5000) {
  if (!Number.isFinite(seq)) {
    return;
  }
  try {
    await page.waitForFunction(
      ({ expectedSeq }) => (document.body.innerText ?? "").includes(`seq ${expectedSeq}`),
      { expectedSeq: seq },
      { timeout: timeoutMs },
    );
  } catch {
    // The polling API is authoritative; submitViaBrowserFetch remains the final fallback.
  }
}

async function isUsable(locator) {
  try {
    return (await locator.count()) > 0 && (await locator.first().isVisible()) && (await locator.first().isEnabled());
  } catch {
    return false;
  }
}

async function submitTileViaUi(page, action) {
  const label = new RegExp(`^Select ${escapeRegex(action)}$`);
  const candidates = [
    page.getByTestId("mahjong-seat-bottom-hand").getByRole("button", { name: label }),
    page.getByTestId("mahjong-draw-slot").getByRole("button", { name: label }),
    page.getByRole("button", { name: label }),
  ];
  for (const candidate of candidates) {
    const usable = await waitForUsable(candidate);
    if (usable) {
      await usable.scrollIntoViewIfNeeded();
      await usable.dblclick();
      return { method: "ui-tile-doubleclick", action };
    }
  }
  throw new Error(`Tile button was not usable: ${action}`);
}

async function submitActionButtonViaUi(page, action) {
  const actionButton = page.getByRole("button", { name: new RegExp(`^Play ${escapeRegex(action)}$`, "i") });
  if (await isUsable(actionButton)) {
    await actionButton.first().click();
    return { method: "ui-action", action };
  }
  throw new Error(`Action button was not usable: ${action}`);
}

async function submitViaUi(page, action) {
  if (isTileAction(action)) {
    return submitTileViaUi(page, normalizeTileCode(action));
  }
  return submitActionButtonViaUi(page, action);
}

async function submitViaBrowserFetch(page, args, action) {
  const receipt = await page.evaluate(
    async ({ actionPath, observerId, actionText }) => {
      const response = await fetch(actionPath, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ playerId: observerId, action: { move: actionText } }),
      });
      const text = await response.text();
      let data = null;
      try {
        data = text ? JSON.parse(text) : null;
      } catch {
        data = { raw: text };
      }
      if (!response.ok) {
        throw new Error(`Action POST failed ${response.status}: ${text.slice(0, 400)}`);
      }
      return data;
    },
    {
      actionPath: buildActionPath(args),
      observerId: args.observerId,
      actionText: action,
    },
  );
  return { method: "browser-fetch", action, receipt };
}

async function submitAction(page, args, action) {
  try {
    return await submitViaUi(page, action);
  } catch (error) {
    const receipt = await submitViaBrowserFetch(page, args, action);
    return {
      ...receipt,
      uiError: error instanceof Error ? error.message : String(error),
    };
  }
}

function summarizeScene(scene) {
  if (!scene) {
    return null;
  }
  return {
    seq: scene.seq,
    phase: scene.phase,
    activePlayerId: scene.activePlayerId,
    legalCount: Array.isArray(scene.legalActions) ? scene.legalActions.length : null,
    status: scene.body?.status,
    hands: scene.body?.table?.seats?.map((seat) => ({
      seatId: seat.seatId,
      playerId: seat.playerId,
      maskedCount: seat.hand?.maskedCount,
      visibleCards: Array.isArray(seat.hand?.cards) ? seat.hand.cards.length : null,
      drawTile: seat.drawTile ?? seat.hand?.drawTile ?? null,
      isObserver: seat.isObserver === true,
    })),
  };
}

async function inspectDom(page) {
  return page.evaluate(() => ({
    hiddenTileCount: document.querySelectorAll(".mahjong-tile.is-hidden").length,
    visibleTileCount: document.querySelectorAll(".mahjong-tile:not(.is-hidden)").length,
    enabledActionButtons: [...document.querySelectorAll(".mahjong-action:not(:disabled)")].map(
      (button) => button.textContent?.trim() ?? "",
    ),
    bottomHandText: document.querySelector('[data-testid="mahjong-seat-bottom-hand"]')?.textContent?.slice(0, 800) ?? "",
    stageText: document.querySelector('[data-testid="mahjong-stage"]')?.textContent?.slice(0, 1600) ?? "",
    resultText: document.querySelector('[data-testid="mahjong-result-banner"]')?.textContent?.slice(0, 800) ?? "",
  }));
}

async function waitForRenderedTerminal(page, terminal) {
  if (terminal?.phase !== "completed") {
    return;
  }
  const seq = terminal?.seq;
  try {
    await page.waitForFunction(
      ({ expectedSeq }) => {
        const bodyText = document.body.innerText ?? "";
        const stageText = document.querySelector('[data-testid="mahjong-stage"]')?.textContent ?? "";
        const seqReady = expectedSeq == null || bodyText.includes(`seq ${expectedSeq}`);
        return seqReady && (stageText.includes("Win") || stageText.includes("Draw") || stageText.includes("Hand result"));
      },
      { expectedSeq: seq },
      { timeout: 15000 },
    );
  } catch {
    // The API result is authoritative; keep the automation artifact even if the UI lags.
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  await mkdir(args.outDir, { recursive: true });

  const startedAt = Date.now();
  const logs = [];
  const browser = await chromium.launch({ headless: !args.headed });
  const page = await browser.newPage({
    viewport: { width: 1440, height: 1100 },
    deviceScaleFactor: 1,
  });
  page.on("console", (message) =>
    logs.push({ type: "console", level: message.type(), text: message.text(), atMs: Date.now() - startedAt }),
  );
  page.on("pageerror", (error) =>
    logs.push({ type: "pageerror", text: String(error?.stack ?? error), atMs: Date.now() - startedAt }),
  );
  page.on("requestfailed", (request) =>
    logs.push({
      type: "requestfailed",
      url: request.url(),
      failure: request.failure()?.errorText,
      atMs: Date.now() - startedAt,
    }),
  );

  let result;
  try {
    const viewerUrl = buildViewerUrl(args);
    await page.goto(viewerUrl, { waitUntil: "domcontentloaded" });
    await page.waitForSelector('[data-testid="mahjong-stage"]', { timeout: 30000 });
    await page.screenshot({ path: path.join(args.outDir, "initial.png"), fullPage: true });

    const actions = [];
    let lastSubmittedSeq = null;
    let terminal = null;
    let lastSession = null;
    let lastScene = null;
    const deadlineAt = startedAt + args.timeoutMs;
    while (Date.now() < deadlineAt) {
      const turn = await readTurn(page, args);
      const { session, scene, seq } = turn;
      lastSession = session;
      lastScene = scene;

      const lifecycle = String(session?.lifecycle ?? "");
      const phase = String(session?.scheduling?.phase ?? "");
      const activeActorId = String(session?.scheduling?.activeActorId ?? scene?.activePlayerId ?? "");
      const acceptsHumanIntent = session?.scheduling?.acceptsHumanIntent === true;
      if (lifecycle && lifecycle !== "live_running") {
        terminal = { reason: "lifecycle", lifecycle, phase, seq };
        break;
      }

      if (acceptsHumanIntent && activeActorId === args.observerId && seq !== lastSubmittedSeq) {
        const legalActions = readLegalActions(scene);
        const action = chooseAction(scene, legalActions, args.strategy);
        if (!action) {
          throw new Error(`No legal action available at seq ${seq}`);
        }
        await waitForUiSeq(page, seq);
        const submitted = await submitAction(page, args, action);
        lastSubmittedSeq = seq;
        actions.push({
          atMs: Date.now() - startedAt,
          before: {
            seq,
            lifecycle,
            phase,
            activeActorId,
            legalCount: legalActions.length,
            legalPreview: legalActions.slice(0, 16),
          },
          submitted,
        });
        await page.waitForTimeout(args.pollMs);
        continue;
      }

      await page.waitForTimeout(args.pollMs);
    }

    if (!terminal) {
      const turn = await readTurn(page, args);
      terminal = {
        reason: "timeout",
        lifecycle: turn.session?.lifecycle,
        phase: turn.session?.scheduling?.phase,
        seq: turn.seq,
      };
      lastSession = turn.session;
      lastScene = turn.scene;
    }

    await waitForRenderedTerminal(page, terminal);
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.waitForTimeout(150);
    await page.screenshot({ path: path.join(args.outDir, "final.png"), fullPage: false });
    result = {
      runId: args.runId,
      sessionId: args.sessionId,
      viewerUrl,
      strategy: args.strategy,
      timeoutMs: args.timeoutMs,
      terminal,
      actionCount: actions.length,
      actions,
      lastSession,
      lastSceneSummary: summarizeScene(lastScene),
      dom: await inspectDom(page),
      logs,
      screenshots: {
        initial: path.join(args.outDir, "initial.png"),
        final: path.join(args.outDir, "final.png"),
      },
    };
  } finally {
    await browser.close();
  }

  const resultPath = path.join(args.outDir, "automation-result.json");
  await writeFile(resultPath, JSON.stringify(result, null, 2), "utf8");
  console.log(
    JSON.stringify(
      {
        terminal: result.terminal,
        actionCount: result.actionCount,
        strategy: result.strategy,
        firstAction: result.actions[0]?.submitted,
        lastAction: result.actions.at(-1)?.submitted,
        result: result.lastSession?.summary?.result,
        screenshots: result.screenshots,
        resultPath,
      },
      null,
      2,
    ),
  );

  if (result.terminal?.reason === "timeout" && result.terminal?.lifecycle === "live_running") {
    process.exitCode = 2;
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack : error);
  process.exitCode = 1;
});

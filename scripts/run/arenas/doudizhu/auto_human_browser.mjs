#!/usr/bin/env node
import { chromium } from "playwright";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

function parseArgs(argv) {
  const args = {
    baseUrl: "http://127.0.0.1:5804",
    sessionId: "doudizhu_match_0001",
    runId: "",
    observerId: "player_0",
    timeoutMs: 45 * 60 * 1000,
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
  --base-url URL          Arena visual server base URL. Default: http://127.0.0.1:5804
  --session-id ID         Session/sample id. Default: doudizhu_match_0001
  --observer-id ID        Human player id. Default: player_0
  --out-dir DIR           Output directory. Default: runs/RUN_ID/browser_auto_full
  --timeout-minutes N     Full-game timeout. Default: 45
  --poll-ms N             Poll interval. Default: 1000
  --strategy NAME         active, first, or pass-first. Default: active
  --headed                Open a headed Chromium window
  --headless              Use headless Chromium
`);
}

function buildViewerUrl({ baseUrl, sessionId, runId }) {
  const url = new URL(`/sessions/${encodeURIComponent(sessionId)}`, normalizeBaseUrl(baseUrl));
  url.searchParams.set("run_id", runId);
  return url.toString();
}

function normalizeBaseUrl(value) {
  return value.endsWith("/") ? value.slice(0, -1) : value;
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

function estimateActionCardCount(actionText) {
  const text = String(actionText).trim();
  if (text.toLowerCase() === "pass") {
    return 0;
  }
  return text.replace(/\s+/g, "").length;
}

function chooseAction(legalActions, strategy) {
  if (legalActions.length === 0) {
    return undefined;
  }
  if (strategy === "first") {
    return legalActions[0];
  }
  if (strategy === "pass-first" && legalActions.some((action) => action.toLowerCase() === "pass")) {
    return "pass";
  }

  const nonPassActions = legalActions.filter((action) => action.toLowerCase() !== "pass");
  if (nonPassActions.length === 0) {
    return legalActions[0];
  }
  return nonPassActions
    .slice()
    .sort(
      (left, right) =>
        estimateActionCardCount(right) - estimateActionCardCount(left) ||
        left.localeCompare(right),
    )[0];
}

function escapeRegex(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

async function isUsable(locator) {
  try {
    return (await locator.count()) > 0 && await locator.first().isVisible() && await locator.first().isEnabled();
  } catch {
    return false;
  }
}

async function submitViaUi(page, action) {
  if (action.toLowerCase() === "pass") {
    const passButton = page.getByRole("button", { name: /^Pass$/ });
    if (await isUsable(passButton)) {
      await passButton.first().click();
      return { method: "ui-pass", action };
    }
  }

  const exactLegalActionButton = page.getByRole("button", {
    name: new RegExp(`^Play legal ${escapeRegex(action)}$`),
  });
  if (!(await isUsable(exactLegalActionButton))) {
    const showLegalActionsButton = page.getByRole("button", { name: /^Show legal actions/ });
    if (await isUsable(showLegalActionsButton)) {
      await showLegalActionsButton.first().click();
    }
  }
  await exactLegalActionButton.first().waitFor({ state: "visible", timeout: 3000 });
  if (await exactLegalActionButton.first().isEnabled()) {
    await exactLegalActionButton.first().click();
    return { method: "ui-legal-exact", action };
  }
  throw new Error(`Legal action button was not enabled: ${action}`);
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
      playerId: seat.playerId,
      maskedCount: seat.hand?.maskedCount,
      visibleCards: Array.isArray(seat.hand?.cards) ? seat.hand.cards.length : null,
    })),
  };
}

async function inspectDom(page) {
  return page.evaluate(() => ({
    leftBacks: document.querySelectorAll('[data-testid="doudizhu-seat-left-hand"] .doudizhu-card--back').length,
    rightBacks: document.querySelectorAll('[data-testid="doudizhu-seat-right-hand"] .doudizhu-card--back').length,
    bottomCards: document.querySelectorAll('[data-testid="doudizhu-seat-bottom-hand"] .doudizhu-card').length,
    enabledActionButtons: [...document.querySelectorAll(".doudizhu-action:not(:disabled), .doudizhu-fallback__toggle:not(:disabled)")].map(
      (button) => button.textContent?.trim() ?? "",
    ),
    stageText: document.querySelector('[data-testid="doudizhu-stage"]')?.textContent?.slice(0, 1200) ?? "",
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
        const stageText = document.querySelector('[data-testid="doudizhu-stage"]')?.textContent ?? "";
        const seqReady = expectedSeq == null || bodyText.includes(`seq ${expectedSeq}`);
        return seqReady && stageText.includes("MATCH COMPLETE");
      },
      { expectedSeq: seq },
      { timeout: 10000 },
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
    viewport: { width: 1440, height: 960 },
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
    await page.waitForSelector('[data-testid="doudizhu-stage"]', { timeout: 30000 });
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
        const action = chooseAction(legalActions, args.strategy);
        if (!action) {
          throw new Error(`No legal action available at seq ${seq}`);
        }
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
            legalPreview: legalActions.slice(0, 12),
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

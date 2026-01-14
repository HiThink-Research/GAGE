import Button from '@material-ui/core/Button';
import Dialog from '@material-ui/core/Dialog';
import DialogActions from '@material-ui/core/DialogActions';
import DialogContent from '@material-ui/core/DialogContent';
import DialogContentText from '@material-ui/core/DialogContentText';
import DialogTitle from '@material-ui/core/DialogTitle';
import Divider from '@material-ui/core/Divider';
import LinearProgress from '@material-ui/core/LinearProgress';
import Paper from '@material-ui/core/Paper';
import Slider from '@material-ui/core/Slider';
import NotInterestedIcon from '@material-ui/icons/NotInterested';
import PauseCircleOutlineRoundedIcon from '@material-ui/icons/PauseCircleOutlineRounded';
import PlayArrowRoundedIcon from '@material-ui/icons/PlayArrowRounded';
import ReplayRoundedIcon from '@material-ui/icons/ReplayRounded';
import SkipNextIcon from '@material-ui/icons/SkipNext';
import SkipPreviousIcon from '@material-ui/icons/SkipPrevious';
import axios from 'axios';
import { Layout, Loading, Message } from 'element-react';
import qs from 'query-string';
import React from 'react';
import '../../assets/gameview.scss';
import { DoudizhuGameBoard } from '../../components/GameBoard';
import {
    card2SuiteAndRank,
    computeHandCardsWidth,
    deepCopy,
    doubleRaf,
    removeCards,
    translateCardData,
} from '../../utils';
import { actionUrl, apiUrl } from '../../utils/config';

class DoudizhuReplayView extends React.Component {
    constructor(props) {
        super(props);

        const mainViewerId = 0; // Id of the player at the bottom of screen
        this.defaultConsiderationTime = 10000;
        this.initConsiderationTime = this.defaultConsiderationTime;
        this.considerationTimeDeduction = 200;
        this.gameStateTimeout = null;
        this.moveHistory = [];
        this.gameStateHistory = [];
        this.chatLog = [];
        this.pendingChatLog = [];
        this.chatPanelRef = React.createRef();
        this.historyPanelRef = React.createRef();
        this.turnStartMs = null;
        this.replayStartMs = null;
        this.moveIntervals = [];
        this.defaultReplayStepMs = 900;
        this.defaultLiveStepMs = 220;
        this.liveReplay = false;
        this.livePending = false;
        this.liveInterval = null;
        this.livePollMs = 1500;
        this.requestUrl = '';
        this.humanPlayEnabled = false;
        this.fastFinishEnabled = false;
        this.fastFinishAction = 'finish';
        this.legalMoves = [];
        this.initGameState = {
            gameStatus: 'ready', // "ready", "playing", "paused", "over"
            playerInfo: [],
            hands: [],
            latestAction: [[], [], []],
            mainViewerId: mainViewerId,
            turn: 0,
            toggleFade: '',

            currentPlayer: null,
            considerationTime: this.initConsiderationTime,
            completedPercent: 0,
        };

        this.chatEntryFirstSeen = new Map();
        this.chatUpdateInterval = null;

        this.state = {
            gameInfo: this.initGameState,
            gameStateLoop: null,
            gameSpeed: 0,
            gameEndDialog: false,
            gameEndDialogText: '',
            fullScreenLoading: false,
            chatLog: [],
            selectedCards: [],
            isPassDisabled: false,
            isHintDisabled: true,
            pendingAction: false,
            pendingChat: false,
            legalMoves: [],
            chatDraft: '',
            hintCursor: null,
        };
    }

    componentDidMount() {
        const query = qs.parse(window.location.search);
        this.humanPlayEnabled = this.parseLiveFlag(query.play || query.human || query.interactive);
        const autoStart = this.parseLiveFlag(query.autoplay || query.auto || query.start || query.live);
        if (autoStart) {
            this.startReplay();
        }
        this.chatUpdateInterval = setInterval(() => {
            if (this.chatEntryFirstSeen.size > 0) {
                this.forceUpdate();
            }
        }, 1000);
    }

    componentDidUpdate(prevProps, prevState) {
        const turnChanged = prevState.gameInfo.turn !== this.state.gameInfo.turn;
        const chatChanged = prevState.chatLog.length !== this.state.chatLog.length;
        if (turnChanged || chatChanged) {
            doubleRaf(() => {
                this.scrollPanels();
            });
        }
    }

    componentWillUnmount() {
        this.clearTimers();
        this.clearTimers();
        this.stopLivePolling();
        if (this.chatUpdateInterval) {
            clearInterval(this.chatUpdateInterval);
            this.chatUpdateInterval = null;
        }
    }

    pickQueryValue(value) {
        if (Array.isArray(value)) {
            return value.length > 0 ? String(value[0]) : '';
        }
        if (value === undefined || value === null) {
            return '';
        }
        return String(value);
    }

    parseLiveFlag(value) {
        const normalized = this.pickQueryValue(value).toLowerCase();
        return ['1', 'true', 'yes', 'on'].includes(normalized);
    }

    resolveActionEndpoint() {
        const query = qs.parse(window.location.search);
        const override = this.pickQueryValue(query.action_url || query.actionUrl || query.action);
        if (override) {
            const trimmed = override.endsWith('/') ? override.slice(0, -1) : override;
            return trimmed.endsWith('/tournament/action') ? trimmed : `${trimmed}/tournament/action`;
        }
        const base = actionUrl.endsWith('/') ? actionUrl.slice(0, -1) : actionUrl;
        return `${base}/tournament/action`;
    }

    resolveChatEndpoint() {
        const actionEndpoint = this.resolveActionEndpoint();
        if (actionEndpoint.endsWith('/tournament/action')) {
            return actionEndpoint.replace('/tournament/action', '/tournament/chat');
        }
        const trimmed = actionEndpoint.endsWith('/') ? actionEndpoint.slice(0, -1) : actionEndpoint;
        return `${trimmed}/tournament/chat`;
    }

    isHumanTurnForState(gameInfo) {
        if (!this.humanPlayEnabled || !this.liveReplay || !gameInfo) {
            return false;
        }
        return (
            gameInfo.gameStatus === 'playing' &&
            gameInfo.currentPlayer !== null &&
            gameInfo.currentPlayer === gameInfo.mainViewerId
        );
    }

    isHumanTurn() {
        return this.isHumanTurnForState(this.state.gameInfo);
    }

    computeHumanControls() {
        const legalMoves = Array.isArray(this.state.legalMoves) ? this.state.legalMoves.map(String) : [];
        const legalSet = new Set(legalMoves);
        const hasLegal = legalSet.size > 0;
        const actionText = this.buildActionText(this.state.selectedCards);
        const canPass = this.isHumanTurn() && hasLegal && legalSet.has('pass');
        const canPlay = this.isHumanTurn() && hasLegal && actionText && legalSet.has(actionText);
        const canHint = this.isHumanTurn() && hasLegal;
        return { canPass, canPlay, canHint };
    }

    buildActionText(cards) {
        if (!Array.isArray(cards) || cards.length === 0) {
            return '';
        }
        const ranks = cards
            .map((card) => {
                let { rank } = card2SuiteAndRank(card);
                if (rank === 'X') rank = 'B';
                else if (rank === 'D') rank = 'R';
                return rank;
            });
        const legalMoves = Array.isArray(this.state.legalMoves) ? this.state.legalMoves : [];
        const targetCounts = this.buildRankCounts(ranks);
        for (const move of legalMoves) {
            if (!move || move === 'pass') {
                continue;
            }
            const moveRanks = this.extractActionRanks(move);
            if (this.isSameRankCounts(targetCounts, this.buildRankCounts(moveRanks))) {
                return String(move);
            }
        }
        return ranks.join('');
    }

    buildRankCounts(ranks) {
        const counts = {};
        (ranks || []).forEach((rank) => {
            if (!rank) {
                return;
            }
            counts[rank] = (counts[rank] || 0) + 1;
        });
        return counts;
    }

    extractActionRanks(actionText) {
        const matches = String(actionText || '')
            .toUpperCase()
            .match(/[3456789TJQKA2BR]/g);
        return matches ? matches : [];
    }

    isSameRankCounts(leftCounts, rightCounts) {
        const leftKeys = Object.keys(leftCounts || {});
        const rightKeys = Object.keys(rightCounts || {});
        if (leftKeys.length !== rightKeys.length) {
            return false;
        }
        for (const key of leftKeys) {
            if (leftCounts[key] !== rightCounts[key]) {
                return false;
            }
        }
        return true;
    }

    resolveActionPayload(actionText, chatText) {
        const query = qs.parse(window.location.search);
        const payload = { action: actionText };
        const runId = this.pickQueryValue(query.run_id || query.runId);
        const sampleId = this.pickQueryValue(query.sample_id || query.sampleId);
        if (runId) {
            payload.run_id = runId;
        }
        if (sampleId) {
            payload.sample_id = sampleId;
        }
        if (chatText) {
            payload.chat = chatText;
        }
        return payload;
    }

    resolveChatPayload(chatText) {
        const query = qs.parse(window.location.search);
        const payload = { chat: chatText };
        const runId = this.pickQueryValue(query.run_id || query.runId);
        const sampleId = this.pickQueryValue(query.sample_id || query.sampleId);
        if (runId) {
            payload.run_id = runId;
        }
        if (sampleId) {
            payload.sample_id = sampleId;
        }
        if (this.state.gameInfo && this.state.gameInfo.mainViewerId !== undefined) {
            payload.player_idx = this.state.gameInfo.mainViewerId;
        }
        return payload;
    }

    buildChatKey(entry) {
        const { playerIdx, playerId } = this.resolveChatPlayer(entry || {});
        const rawText = entry.text || entry.message || entry.chat || '';
        const cleaned = this.normalizeChatText(rawText);
        if (!cleaned) {
            return null;
        }
        const key = `${playerIdx ?? playerId ?? 'unknown'}::${cleaned}`;
        return { key, text: cleaned };
    }

    submitHumanAction(actionText) {
        if (!actionText || this.state.pendingAction) {
            return;
        }
        const endpoint = this.resolveActionEndpoint();
        const chatText = (this.state.chatDraft || '').trim();
        const payload = this.resolveActionPayload(actionText, chatText);
        this.setState({ pendingAction: true });
        axios
            .post(endpoint, payload)
            .catch((err) => {
                const status = err && err.response ? err.response.status : null;
                const detail = err && err.response && err.response.data ? err.response.data.error : '';
                Message({
                    message: `Failed to submit action${status ? ` (${status})` : ''}${detail ? `: ${detail}` : ''}`,
                    type: 'error',
                    showClose: true,
                });
            })
            .finally(() => {
                this.setState({ pendingAction: false, chatDraft: '' });
            });
    }

    submitHumanChat() {
        const chatText = (this.state.chatDraft || '').trim();
        if (!chatText || this.state.pendingChat) {
            return;
        }
        const endpoint = this.resolveChatEndpoint();
        const payload = this.resolveChatPayload(chatText);
        this.setState({ pendingChat: true });
        axios
            .post(endpoint, payload)
            .then(() => {
                const playerIdx = this.state.gameInfo.mainViewerId ?? 0;
                const entry = {
                    player_idx: playerIdx,
                    text: chatText,
                    timestamp_ms: Date.now(),
                    local: true,
                };
                this.pendingChatLog.push(entry);
                const mergedChatLog = this.mergePendingChatLog(this.state.chatLog || []);
                this.chatLog = mergedChatLog;
                this.setState({ chatDraft: '', chatLog: mergedChatLog });
            })
            .catch((err) => {
                const status = err && err.response ? err.response.status : null;
                const detail = err && err.response && err.response.data ? err.response.data.error : '';
                Message({
                    message: `Failed to submit chat${status ? ` (${status})` : ''}${detail ? `: ${detail}` : ''}`,
                    type: 'error',
                    showClose: true,
                });
            })
            .finally(() => {
                this.setState({ pendingChat: false });
            });
    }

    handleSelectedCards(cards) {
        if (!this.isHumanTurn()) {
            return;
        }
        const selection = Array.isArray(this.state.selectedCards) ? this.state.selectedCards.slice() : [];
        (Array.isArray(cards) ? cards : []).forEach((card) => {
            const idx = selection.indexOf(card);
            if (idx >= 0) {
                selection.splice(idx, 1);
            } else {
                selection.push(card);
            }
        });
        this.setState({ selectedCards: selection });
    }

    handleMainPlayerAct(actionType) {
        if (!this.humanPlayEnabled) {
            return;
        }
        const controls = this.computeHumanControls();
        if (actionType === 'deselect') {
            this.setState({ selectedCards: [] });
            return;
        }
        if (actionType === 'hint') {
            if (!controls.canHint) {
                return;
            }
            const hint = this.pickNextHint();
            if (hint) {
                this.setState({ selectedCards: hint.cards, hintCursor: hint.cursor });
                return;
            }
            return;
        }
        if (actionType === 'finish') {
            if (!this.fastFinishEnabled || !this.isHumanTurn()) {
                return;
            }
            this.setState({ selectedCards: [], hintCursor: null });
            this.submitHumanAction(this.fastFinishAction);
            return;
        }
        if (actionType === 'pass') {
            if (!controls.canPass) {
                return;
            }
            this.setState({ selectedCards: [], hintCursor: null });
            this.submitHumanAction('pass');
            return;
        }
        if (actionType === 'play') {
            const actionText = this.buildActionText(this.state.selectedCards);
            if (!actionText || !controls.canPlay) {
                Message({
                    message: 'Selected cards are not legal',
                    type: 'warning',
                    showClose: true,
                });
                return;
            }
            this.setState({ selectedCards: [], hintCursor: null });
            this.submitHumanAction(actionText);
        }
    }

    handleChatDraftChange(value) {
        this.setState({ chatDraft: value });
    }

    pickNextHint() {
        const legalMoves = Array.isArray(this.state.legalMoves) ? this.state.legalMoves : [];
        const hand = this.state.gameInfo.hands[this.state.gameInfo.mainViewerId] || [];
        const candidates = legalMoves.filter((move) => move && move !== 'pass');
        if (candidates.length === 0) {
            return null;
        }
        const current = this.state.hintCursor;
        const currentIndex = current ? candidates.findIndex((move) => String(move) === String(current)) : -1;
        for (let offset = 1; offset <= candidates.length; offset += 1) {
            const nextIndex = (currentIndex + offset) % candidates.length;
            const move = String(candidates[nextIndex]);
            const cards = this.resolveCardsForAction(move, hand);
            if (cards.length > 0) {
                return { cards, cursor: move };
            }
        }
        return null;
    }

    resolveCardsForAction(actionText, hand) {
        if (!actionText || actionText === 'pass') {
            return [];
        }
        const ranks = this.extractActionRanks(actionText);
        if (ranks.length === 0) {
            return [];
        }
        const cardsLeft = Array.isArray(hand) ? hand.slice() : [];
        const selected = [];
        const trans = { B: 'BJ', R: 'RJ' };
        for (const char of ranks) {
            const target = trans[char];
            if (target) {
                const idx = cardsLeft.indexOf(target);
                if (idx >= 0) {
                    selected.push(target);
                    cardsLeft.splice(idx, 1);
                    continue;
                }
                return [];
            }
            const idx = cardsLeft.findIndex((card) => card && card[1] === char);
            if (idx >= 0) {
                const picked = cardsLeft[idx];
                selected.push(picked);
                cardsLeft.splice(idx, 1);
            } else {
                return [];
            }
        }
        return selected;
    }

    resolveReplayRequest(query) {
        const replayUrl = this.pickQueryValue(query.replay_url || query.replayUrl || query.replay);
        if (replayUrl) {
            return replayUrl;
        }

        const replayPath = this.pickQueryValue(query.replay_path || query.replayPath);
        if (replayPath) {
            const params = new URLSearchParams({ replay_path: replayPath });
            return `${apiUrl}/tournament/replay?${params.toString()}`;
        }

        const runId = this.pickQueryValue(query.run_id || query.runId);
        if (runId) {
            const sampleId = this.pickQueryValue(query.sample_id || query.sampleId);
            const params = new URLSearchParams({ run_id: runId });
            if (sampleId) {
                params.set('sample_id', sampleId);
            }
            return `${apiUrl}/tournament/replay?${params.toString()}`;
        }

        const name = this.pickQueryValue(query.name);
        const agent0 = this.pickQueryValue(query.agent0);
        const agent1 = this.pickQueryValue(query.agent1);
        const index = this.pickQueryValue(query.index);
        const params = new URLSearchParams({
            name,
            agent0,
            agent1,
            index,
        });
        return `${apiUrl}/tournament/replay?${params.toString()}`;
    }

    normalizeReplayPayload(rawPayload) {
        const payload = rawPayload || {};
        const playerInfo = payload.playerInfo || payload.player_info || [];
        const initHands = payload.initHands || payload.init_hands || [];
        const moveHistory = payload.moveHistory || payload.move_history || [];
        const chatLog = payload.chatLog || payload.chat_log || [];
        const legalMoves = payload.legalMoves || payload.legal_moves || [];
        const currentPlayer =
            payload.currentPlayer ??
            payload.current_player ??
            payload.active_player_idx ??
            payload.activePlayer ??
            null;
        const startTimeMs =
            payload.startTimeMs ||
            payload.start_time_ms ||
            payload.start_time ||
            payload.startTime ||
            null;
        return {
            playerInfo,
            initHands,
            moveHistory,
            chatLog,
            legalMoves,
            currentPlayer,
            startTimeMs,
        };
    }

    normalizeMoveHistory(entries) {
        return (entries || []).map((entry) => {
            const normalized = { ...entry };
            if (normalized.playerIdx === undefined) {
                const playerIdx =
                    normalized.player_idx !== undefined
                        ? normalized.player_idx
                        : normalized.playerId !== undefined
                          ? normalized.playerId
                          : normalized.player_id;
                if (playerIdx !== undefined) {
                    normalized.playerIdx = Number(playerIdx);
                }
            }
            if (normalized.move === undefined) {
                normalized.move =
                    normalized.action_cards !== undefined
                        ? normalized.action_cards
                        : normalized.action_text !== undefined
                          ? normalized.action_text
                          : normalized.action;
            }
            if (normalized.timestampMs === undefined) {
                const rawTimestamp =
                    normalized.timestamp_ms ??
                    normalized.timestamp ??
                    normalized.time_ms ??
                    normalized.time ??
                    normalized.ts;
                if (rawTimestamp !== undefined && rawTimestamp !== null) {
                    const parsed = Number(rawTimestamp);
                    if (!Number.isNaN(parsed)) {
                        normalized.timestampMs = parsed;
                    }
                }
            }
            if (!normalized.info) {
                normalized.info = {};
            }
            return normalized;
        });
    }

    buildMoveIntervals(startTimeMs) {
        const intervals = [];
        let lastTimestamp = null;
        let hasStart = Number.isFinite(startTimeMs);
        for (let idx = 0; idx < this.moveHistory.length; idx += 1) {
            const entry = this.moveHistory[idx];
            const timestamp = Number.isFinite(entry.timestampMs) ? entry.timestampMs : null;
            let interval = this.defaultReplayStepMs;
            if (timestamp !== null) {
                if (lastTimestamp !== null) {
                    interval = timestamp - lastTimestamp;
                } else if (hasStart) {
                    interval = timestamp - startTimeMs;
                } else {
                    interval = this.defaultReplayStepMs;
                }
                lastTimestamp = timestamp;
            }
            if (!Number.isFinite(interval) || interval < 0) {
                interval = this.defaultReplayStepMs;
            }
            intervals.push(interval);
        }
        this.moveIntervals = intervals;
    }

    resolveNextDelayMs() {
        if (!this.moveIntervals || this.moveIntervals.length === 0) {
            return this.defaultReplayStepMs;
        }
        const idx = this.state.gameInfo.turn || 0;
        const interval = this.moveIntervals[idx];
        if (!Number.isFinite(interval) || interval < 0) {
            return this.defaultReplayStepMs;
        }
        const speedFactor = Math.pow(2, this.state.gameSpeed || 0);
        return interval / speedFactor;
    }

    resolveLiveDelayMs() {
        const speedFactor = Math.pow(2, this.state.gameSpeed || 0);
        return this.defaultLiveStepMs / speedFactor;
    }

    clearTimers() {
        if (this.gameStateTimeout) {
            window.clearTimeout(this.gameStateTimeout);
            this.gameStateTimeout = null;
        }
    }

    cardStr2Arr(cardStr) {
        return cardStr === 'pass' || cardStr === '' ? 'pass' : cardStr.split(' ');
    }

    formatMoveText(move) {
        if (Array.isArray(move)) {
            return move.join(' ');
        }
        if (move === undefined || move === null) {
            return '';
        }
        return String(move);
    }

    resolvePlayerName(playerIdx, playerInfo) {
        const fallback = playerIdx !== undefined ? `Player ${playerIdx}` : 'Player';
        if (!Array.isArray(playerInfo) || playerIdx === undefined || playerIdx === null) {
            return fallback;
        }
        const entry = playerInfo.find((item) => item.index === playerIdx || item.id === playerIdx);
        if (!entry) {
            return fallback;
        }
        if (entry.agentInfo && entry.agentInfo.name) {
            return entry.agentInfo.name;
        }
        if (entry.name) {
            return entry.name;
        }
        return fallback;
    }

    scrollPanels() {
        const panels = [this.chatPanelRef.current, this.historyPanelRef.current];
        panels.forEach((panel) => {
            if (!panel) {
                return;
            }
            panel.scrollTop = panel.scrollHeight;
        });
    }

    normalizeChatText(text) {
        if (text === undefined || text === null) {
            return '';
        }
        return String(text).replace(/\s+/g, ' ').trim();
    }

    truncateChatText(text, maxLen = 42) {
        if (text.length <= maxLen) {
            return text;
        }
        return `${text.slice(0, Math.max(0, maxLen - 3))}...`;
    }

    resolveChatPlayer(entry) {
        const rawPlayerId = entry.player_id || entry.playerId || entry.player;
        const rawPlayerIdx = entry.player_idx ?? entry.playerIdx;
        let playerIdx = rawPlayerIdx !== undefined ? Number(rawPlayerIdx) : undefined;
        if (playerIdx !== undefined && Number.isNaN(playerIdx)) {
            playerIdx = undefined;
        }
        if (playerIdx === undefined && typeof rawPlayerId === 'string' && rawPlayerId.startsWith('player_')) {
            const parsedIdx = Number(rawPlayerId.split('_')[1]);
            if (!Number.isNaN(parsedIdx)) {
                playerIdx = parsedIdx;
            }
        }
        const playerId = rawPlayerId !== undefined ? String(rawPlayerId) : '';
        return { playerIdx, playerId };
    }

    buildChatBubbleMap(chatLog) {
        const bubbles = {};
        const now = Date.now();
        const expirationMs = 5000;
        
        (chatLog || []).forEach((entry) => {
            const { playerIdx, playerId } = this.resolveChatPlayer(entry);
            const rawText = entry.text || entry.message || entry.chat || '';
            const cleaned = this.normalizeChatText(rawText);
            if (!cleaned) {
                return;
            }
            
            // Generate a unique key for the message
            // Use timestamp if available from entry to differentiate same content messages if possible
            const timestamp = entry.timestamp_ms || entry.timestampMs;
            const uniqueKey = `${playerIdx ?? playerId ?? 'unknown'}::${cleaned}::${timestamp || 'no_ts'}`;
            
            if (!this.chatEntryFirstSeen.has(uniqueKey)) {
                this.chatEntryFirstSeen.set(uniqueKey, now);
            }
            
            const firstSeen = this.chatEntryFirstSeen.get(uniqueKey);
            
            // Only show if seen within last 5 seconds
            if (now - firstSeen > expirationMs) {
                return;
            }

            const displayText = this.truncateChatText(cleaned);
            if (playerIdx !== undefined) {
                bubbles[playerIdx] = displayText;
            } else if (playerId) {
                bubbles[playerId] = displayText;
            }
        });
        return bubbles;
    }

    mergeChatLog(chatLog, moveHistory) {
        const moves = Array.isArray(moveHistory) ? moveHistory : [];
        const merged = [];
        if (moves.length > 0) {
            for (const entry of moves) {
                const rawText = entry.chat || entry.chat_text || entry.chatText || '';
                const cleaned = this.normalizeChatText(rawText);
                if (!cleaned) {
                    continue;
                }
                merged.push({
                    player_id: entry.player_id || entry.playerId || entry.player,
                    player_idx: entry.player_idx ?? entry.playerIdx,
                    text: cleaned,
                });
            }
        }
        const normalized = Array.isArray(chatLog) ? [...chatLog] : [];
        merged.push(...normalized);
        return this.compactChatLog(merged);
    }

    mergePendingChatLog(chatLog) {
        const base = this.compactChatLog(Array.isArray(chatLog) ? [...chatLog] : []);
        const existingKeys = new Set();
        base.forEach((entry) => {
            const info = this.buildChatKey(entry);
            if (info) {
                existingKeys.add(info.key);
            }
        });
        const pending = [];
        const merged = [...base];
        (this.pendingChatLog || []).forEach((entry) => {
            const info = this.buildChatKey(entry);
            if (!info) {
                return;
            }
            if (existingKeys.has(info.key)) {
                return;
            }
            pending.push(entry);
            merged.push({ ...entry, text: info.text });
            existingKeys.add(info.key);
        });
        this.pendingChatLog = pending;
        return merged;
    }

    compactChatLog(entries) {
        const compacted = [];
        let lastKey = null;
        for (const entry of entries) {
            const { playerIdx, playerId } = this.resolveChatPlayer(entry || {});
            const rawText = entry.text || entry.message || entry.chat || '';
            const cleaned = this.normalizeChatText(rawText);
            if (!cleaned) {
                continue;
            }
            const key = `${playerIdx ?? playerId ?? 'unknown'}::${cleaned}`;
            if (key === lastKey) {
                continue;
            }
            compacted.push({ ...entry, text: cleaned });
            lastKey = key;
        }
        return compacted;
    }

    generateNewState() {
        let gameInfo = deepCopy(this.state.gameInfo);
        if (this.state.gameInfo.turn === this.moveHistory.length) return gameInfo;
        // check if the game state of next turn is already in game state history
        if (this.state.gameInfo.turn + 1 < this.gameStateHistory.length) {
            gameInfo = deepCopy(this.gameStateHistory[this.state.gameInfo.turn + 1]);
        } else {
            let newMove = this.moveHistory[this.state.gameInfo.turn];
            if (newMove.playerIdx === undefined || newMove.playerIdx === null) {
                return gameInfo;
            }
            if (newMove.playerIdx !== this.state.gameInfo.currentPlayer) {
                gameInfo.currentPlayer = newMove.playerIdx;
            }
            gameInfo.latestAction[newMove.playerIdx] = this.cardStr2Arr(
                Array.isArray(newMove.move) ? newMove.move.join(' ') : newMove.move,
            );
            gameInfo.turn++;
            gameInfo.currentPlayer = (gameInfo.currentPlayer + 1) % 3;
            // take away played cards from player's hands
            const remainedCards = removeCards(
                gameInfo.latestAction[newMove.playerIdx],
                gameInfo.hands[newMove.playerIdx],
            );
            if (remainedCards !== false) {
                gameInfo.hands[newMove.playerIdx] = remainedCards;
            } else {
                Message({
                    message: "Cannot find cards in move from player's hand",
                    type: 'error',
                    showClose: true,
                });
            }
            // check if game ends
            if (remainedCards.length === 0) {
                doubleRaf(() => {
                    const winner = this.state.gameInfo.playerInfo.find((element) => {
                        return element.index === newMove.playerIdx;
                    });
                    if (winner) {
                        gameInfo.gameStatus = 'over';
                        this.setState({ gameInfo: gameInfo });
                        if (winner.role === 'landlord')
                            setTimeout(() => {
                                const mes = 'Landlord Wins';
                                this.setState({ gameEndDialog: true, gameEndDialogText: mes });
                            }, 200);
                        else
                            setTimeout(() => {
                                const mes = 'Peasants Win';
                                this.setState({ gameEndDialog: true, gameEndDialogText: mes });
                            }, 200);
                    } else {
                        Message({
                            message: 'Error in finding winner',
                            type: 'error',
                            showClose: true,
                        });
                    }
                });
                return gameInfo;
            }
            gameInfo.considerationTime = this.initConsiderationTime;
            gameInfo.completedPercent += 100.0 / (this.moveHistory.length - 1);
            // if current state is new to game state history, push it to the game state history array
            if (gameInfo.turn === this.gameStateHistory.length) {
                this.gameStateHistory.push(gameInfo);
            } else {
                this.gameStateHistory = this.gameStateHistory.slice(0, gameInfo.turn);
                this.gameStateHistory.push(gameInfo);
            }
        }
        return gameInfo;
    }

    gameStateTimer() {
        const tickMs = this.considerationTimeDeduction;
        this.gameStateTimeout = setTimeout(() => {
            if (this.liveReplay) {
                if (this.state.gameInfo.gameStatus !== 'playing') {
                    return;
                }
                const now = Date.now();
                if (!this.turnStartMs) {
                    this.turnStartMs = now;
                }
                const elapsed = Math.max(0, now - this.turnStartMs);
                let gameInfo = deepCopy(this.state.gameInfo);
                gameInfo.considerationTime = elapsed;
                const hasNextMove = this.state.gameInfo.turn < this.moveHistory.length;
                const nextDelayMs = this.humanPlayEnabled ? this.resolveLiveDelayMs() : this.resolveNextDelayMs();
                if (hasNextMove && elapsed >= nextDelayMs) {
                    this.turnStartMs = Date.now();
                    const nextGameInfo = this.generateNewState();
                    nextGameInfo.considerationTime = 0;
                    this.setState({ gameInfo: nextGameInfo }, () => {
                        if (this.state.gameInfo.gameStatus !== 'over') {
                            this.gameStateTimer();
                        }
                    });
                } else {
                    this.setState({ gameInfo: gameInfo }, () => {
                        if (this.state.gameInfo.gameStatus !== 'over') {
                            this.gameStateTimer();
                        }
                    });
                }
                return;
            }
            let currentConsiderationTime = this.state.gameInfo.considerationTime;
            if (currentConsiderationTime > 0) {
                currentConsiderationTime -= this.considerationTimeDeduction * Math.pow(2, this.state.gameSpeed);
                currentConsiderationTime = currentConsiderationTime < 0 ? 0 : currentConsiderationTime;
                if (currentConsiderationTime === 0 && this.state.gameSpeed < 2) {
                    let gameInfo = deepCopy(this.state.gameInfo);
                    gameInfo.toggleFade = 'fade-out';
                    this.setState({ gameInfo: gameInfo });
                }
                let gameInfo = deepCopy(this.state.gameInfo);
                gameInfo.considerationTime = currentConsiderationTime;
                this.setState({ gameInfo: gameInfo });
                this.gameStateTimer();
            } else {
                let gameInfo = this.generateNewState();
                if (gameInfo.gameStatus === 'over') return;
                gameInfo.gameStatus = 'playing';
                if (this.state.gameInfo.toggleFade === 'fade-out') {
                    gameInfo.toggleFade = 'fade-in';
                }
                this.setState({ gameInfo: gameInfo }, () => {
                    // toggle fade in
                    if (this.state.gameInfo.toggleFade !== '') {
                        setTimeout(() => {
                            let gameInfo = deepCopy(this.state.gameInfo);
                            gameInfo.toggleFade = '';
                            this.setState({ gameInfo: gameInfo });
                        }, 200);
                    }
                });
            }
        }, tickMs);
    }

    fetchReplaySnapshot(requestUrl, allowMissing) {
        return axios
            .get(requestUrl)
            .then((res) => {
                let payload = res.data;

                // for test use
                if (typeof payload === 'string')
                    payload = JSON.parse(payload.replaceAll("'", '"').replaceAll('None', 'null'));

                if (payload && payload.error) {
                    if (allowMissing) {
                        return null;
                    }
                    throw new Error(payload.error);
                }
                return this.normalizeReplayPayload(payload);
            })
            .catch((err) => {
                if (allowMissing) {
                    return null;
                }
                throw err;
            });
    }

    initializeGameInfo(normalized) {
        let gameInfo = deepCopy(this.initGameState);
        gameInfo.gameStatus = 'playing';
        gameInfo.playerInfo = normalized.playerInfo;
        gameInfo.hands = normalized.initHands.map((element) => {
            if (Array.isArray(element)) {
                return element;
            }
            return this.cardStr2Arr(element);
        });
        if (normalized.currentPlayer !== null && normalized.currentPlayer !== undefined) {
            gameInfo.currentPlayer = Number(normalized.currentPlayer) || 0;
        } else {
            const landlordEntry = normalized.playerInfo.find((element) => element.role === 'landlord');
            if (landlordEntry) {
                gameInfo.currentPlayer = landlordEntry.index !== undefined ? landlordEntry.index : landlordEntry.id;
            } else {
                gameInfo.currentPlayer = 0;
            }
            gameInfo.currentPlayer = Number(gameInfo.currentPlayer) || 0;
        }
        return gameInfo;
    }

    applyReplaySnapshot(normalized, resetState) {
        const parsedStart = Number(normalized.startTimeMs);
        this.replayStartMs = Number.isFinite(parsedStart) ? parsedStart : null;
        this.moveHistory = this.normalizeMoveHistory(normalized.moveHistory);
        this.preprocessMoveHistory();
        this.chatLog = Array.isArray(normalized.chatLog) ? normalized.chatLog : [];
        this.chatLog = this.mergePendingChatLog(this.chatLog);
        this.legalMoves = Array.isArray(normalized.legalMoves) ? normalized.legalMoves : [];
        this.buildMoveIntervals(this.replayStartMs);

        if (resetState) {
            this.gameStateHistory = [];
            const gameInfo = this.initializeGameInfo(normalized);
            if (this.liveReplay) {
                this.turnStartMs = Date.now();
                gameInfo.considerationTime = 0;
            }
            this.gameStateHistory.push(gameInfo);
            this.setState(
                {
                    gameInfo: gameInfo,
                    fullScreenLoading: false,
                    chatLog: this.chatLog,
                    legalMoves: this.legalMoves,
                    selectedCards: [],
                    hintCursor: null,
                },
                () => {
                    this.clearTimers();
                    if (this.moveHistory.length > 0) {
                        this.gameStateTimer();
                    }
                    doubleRaf(() => {
                        this.scrollPanels();
                    });
                },
            );
            return;
        }

        let gameInfo = null;
        if (!this.state.gameInfo.playerInfo.length && normalized.playerInfo.length) {
            gameInfo = deepCopy(this.state.gameInfo);
            gameInfo.playerInfo = normalized.playerInfo;
            gameInfo.hands = normalized.initHands.map((element) => {
                if (Array.isArray(element)) {
                    return element;
                }
                return this.cardStr2Arr(element);
            });
            if (normalized.currentPlayer !== null && normalized.currentPlayer !== undefined) {
                gameInfo.currentPlayer = Number(normalized.currentPlayer) || 0;
            } else if (gameInfo.currentPlayer === null) {
                const landlordEntry = normalized.playerInfo.find((element) => element.role === 'landlord');
                if (landlordEntry) {
                    gameInfo.currentPlayer =
                        landlordEntry.index !== undefined ? landlordEntry.index : landlordEntry.id;
                } else {
                    gameInfo.currentPlayer = 0;
                }
                gameInfo.currentPlayer = Number(gameInfo.currentPlayer) || 0;
            }
        }

        const stateUpdate = { chatLog: this.chatLog, legalMoves: this.legalMoves };
        if (gameInfo) {
            stateUpdate.gameInfo = gameInfo;
        }
        const nextGameInfo = gameInfo || this.state.gameInfo;
        if (!this.isHumanTurnForState(nextGameInfo)) {
            stateUpdate.selectedCards = [];
            stateUpdate.hintCursor = null;
        }
        this.setState(stateUpdate, () => {
            if (
                (this.liveReplay || this.moveHistory.length > 0) &&
                !this.gameStateTimeout &&
                this.state.gameInfo.gameStatus !== 'over'
            ) {
                this.gameStateTimer();
            }
        });
    }

    preprocessMoveHistory() {
        for (const historyItem of this.moveHistory) {
            if (historyItem.info && !Array.isArray(historyItem.info)) {
                if ('probs' in historyItem.info) {
                    historyItem.info.probs = Object.entries(historyItem.info.probs).sort(
                        (a, b) => Number(b[1]) - Number(a[1]),
                    );
                } else if ('values' in historyItem.info) {
                    historyItem.info.values = Object.entries(historyItem.info.values).sort(
                        (a, b) => Number(b[1]) - Number(a[1]),
                    );
                }
            }
        }
    }

    startLivePolling() {
        if (this.liveInterval) {
            return;
        }
        this.liveInterval = setInterval(() => {
            this.refreshReplay();
        }, this.livePollMs);
    }

    stopLivePolling() {
        if (!this.liveInterval) {
            return;
        }
        window.clearInterval(this.liveInterval);
        this.liveInterval = null;
    }

    refreshReplay() {
        if (!this.requestUrl) {
            return;
        }
        this.fetchReplaySnapshot(this.requestUrl, true)
            .then((normalized) => {
                if (!normalized) {
                    return;
                }
                if (this.livePending || this.state.gameInfo.gameStatus === 'ready') {
                    this.livePending = false;
                    this.applyReplaySnapshot(normalized, true);
                } else {
                    this.applyReplaySnapshot(normalized, false);
                }
                if (this.state.gameInfo.gameStatus === 'over') {
                    this.stopLivePolling();
                }
            })
            .catch(() => {});
    }

    startReplay(options = {}) {
        const { forceNonLive = false } = options;
        const query = qs.parse(window.location.search);
        this.humanPlayEnabled = this.parseLiveFlag(query.play || query.human || query.interactive);
        this.fastFinishEnabled = this.parseLiveFlag(
            query.fast_finish || query.fastFinish || query.finish,
        );
        const fastAction = this.pickQueryValue(query.fast_finish_action || query.fastFinishAction);
        if (fastAction) {
            this.fastFinishAction = fastAction;
        }
        const requestUrl = this.resolveReplayRequest(query);
        if (!requestUrl) {
            Message({
                message: 'Missing replay request parameters',
                type: 'error',
                showClose: true,
            });
            return;
        }

        this.requestUrl = requestUrl;
        this.liveReplay = !forceNonLive && this.parseLiveFlag(query.live || query.streaming);
        this.initConsiderationTime = this.liveReplay ? 0 : this.defaultConsiderationTime;
        this.initGameState.considerationTime = this.initConsiderationTime;
        if (this.humanPlayEnabled) {
            this.livePollMs = 300;
        } else {
            this.livePollMs = 1500;
        }
        const pollMs = Number(this.pickQueryValue(query.live_poll_ms || query.poll_ms));
        if (!Number.isNaN(pollMs) && pollMs >= 200) {
            this.livePollMs = pollMs;
        }

        this.clearTimers();
        this.stopLivePolling();
        this.livePending = false;

        // start full screen loading
        this.setState({ fullScreenLoading: true });
        this.fetchReplaySnapshot(requestUrl, this.liveReplay)
            .then((normalized) => {
                if (!normalized) {
                    this.livePending = true;
                    if (this.liveReplay) {
                        this.startLivePolling();
                    }
                    return;
                }
                this.applyReplaySnapshot(normalized, true);
                if (this.liveReplay) {
                    this.startLivePolling();
                }
            })
            .catch(() => {
                this.setState({ fullScreenLoading: false });
                Message({
                    message: 'Error in getting replay data',
                    type: 'error',
                    showClose: true,
                });
            });
    }

    runNewTurn() {
        this.gameStateTimer();
    }

    pauseReplay() {
        if (this.gameStateTimeout) {
            window.clearTimeout(this.gameStateTimeout);
            this.gameStateTimeout = null;
        }
        let gameInfo = deepCopy(this.state.gameInfo);
        gameInfo.gameStatus = 'paused';
        this.setState({ gameInfo: gameInfo });
    }

    resumeReplay() {
        this.gameStateTimer();
        let gameInfo = deepCopy(this.state.gameInfo);
        gameInfo.gameStatus = 'playing';
        this.setState({ gameInfo: gameInfo });
    }

    changeGameSpeed(newVal) {
        this.setState({ gameSpeed: newVal });
    }

    gameStatusButton(status) {
        switch (status) {
            case 'ready':
                return (
                    <Button
                        className={'status-button'}
                        variant={'contained'}
                        startIcon={<PlayArrowRoundedIcon />}
                        color="primary"
                        onClick={() => {
                            this.startReplay();
                        }}
                    >
                        Start
                    </Button>
                );
            case 'playing':
                return (
                    <Button
                        className={'status-button'}
                        variant={'contained'}
                        startIcon={<PauseCircleOutlineRoundedIcon />}
                        color="secondary"
                        onClick={() => {
                            this.pauseReplay();
                        }}
                    >
                        Pause
                    </Button>
                );
            case 'paused':
                return (
                    <Button
                        className={'status-button'}
                        variant={'contained'}
                        startIcon={<PlayArrowRoundedIcon />}
                        color="primary"
                        onClick={() => {
                            this.resumeReplay();
                        }}
                    >
                        Resume
                    </Button>
                );
            case 'over':
                return (
                    <Button
                        className={'status-button'}
                        variant={'contained'}
                        startIcon={<ReplayRoundedIcon />}
                        color="primary"
                        onClick={() => {
                            this.startReplay({ forceNonLive: this.humanPlayEnabled });
                        }}
                    >
                        Restart
                    </Button>
                );
            default:
                alert(`undefined game status: ${status}`);
        }
    }

    computeSingleLineHand(cards) {
        if (cards === 'pass') {
            return (
                <div className={'non-card ' + this.state.gameInfo.toggleFade}>
                    <span>Pass</span>
                </div>
            );
        } else {
            return (
                <div className={'unselectable playingCards loose ' + this.state.gameInfo.toggleFade}>
                    <ul className="hand" style={{ width: computeHandCardsWidth(cards.length, 10) }}>
                        {cards.map((card) => {
                            const [rankClass, suitClass, rankText, suitText] = translateCardData(card);
                            return (
                                <li key={`handCard-${card}`}>
                                    <a className={`card ${rankClass} ${suitClass}`} href="/#">
                                        <span className="rank">{rankText}</span>
                                        <span className="suit">{suitText}</span>
                                    </a>
                                </li>
                            );
                        })}
                    </ul>
                </div>
            );
        }
    }

    computePredictionCards(cards, hands) {
        let computedCards = [];
        if (cards.length > 0) {
            hands.forEach((card) => {
                let { rank } = card2SuiteAndRank(card);

                // X is B, D is R
                if (rank === 'X') rank = 'B';
                else if (rank === 'D') rank = 'R';
                const idx = cards.indexOf(rank);
                if (idx >= 0) {
                    cards.splice(idx, 1);
                    computedCards.push(card);
                }
            });
        } else {
            computedCards = 'pass';
        }

        if (computedCards === 'pass') {
            return (
                <div className={'non-card ' + this.state.gameInfo.toggleFade}>
                    <span>{'PASS'}</span>
                </div>
            );
        } else {
            return (
                <div className={'unselectable playingCards loose ' + this.state.gameInfo.toggleFade}>
                    <ul className="hand" style={{ width: computeHandCardsWidth(computedCards.length, 10) }}>
                        {computedCards.map((card) => {
                            const [rankClass, suitClass, rankText, suitText] = translateCardData(card);
                            return (
                                <li key={`handCard-${card}`}>
                                    <label className={`card ${rankClass} ${suitClass}`} href="/#">
                                        <span className="rank">{rankText}</span>
                                        <span className="suit">{suitText}</span>
                                    </label>
                                </li>
                            );
                        })}
                    </ul>
                </div>
            );
        }
    }

    computeProbabilityItem(idx) {
        // return <span className={'waiting'}>Currently Unavailable...</span>;
        if (this.state.gameInfo.gameStatus !== 'ready' && this.state.gameInfo.turn < this.moveHistory.length) {
            let currentMove = null;
            if (this.state.gameInfo.turn !== this.moveHistory.length) {
                currentMove = this.moveHistory[this.state.gameInfo.turn];
            }

            let style = {};
            // style["backgroundColor"] = this.moveHistory[this.state.gameInfo.turn].probabilities.length > idx ? `rgba(63, 81, 181, ${this.moveHistory[this.state.gameInfo.turn].probabilities[idx].probability})` : "#bdbdbd";
            let probabilities = null;
            let probabilityItemType = null;

            if (currentMove) {
                if (Array.isArray(currentMove.info)) {
                    probabilityItemType = 'Rule';
                } else {
                    if ('probs' in currentMove.info) {
                        probabilityItemType = 'Probability';
                        probabilities = idx < currentMove.info.probs.length ? currentMove.info.probs[idx] : null;
                    } else if ('values' in currentMove.info) {
                        probabilityItemType = 'Expected payoff';
                        probabilities = idx < currentMove.info.values.length ? currentMove.info.values[idx] : null;
                    } else {
                        probabilityItemType = 'Rule';
                    }
                }
            }

            style['backgroundColor'] = currentMove !== null ? '#fff' : '#bdbdbd';

            return (
                <div className={'playing'} style={style}>
                    <div className="probability-move">
                        {probabilities ? (
                            this.computePredictionCards(
                                probabilities[0] === 'pass' ? [] : probabilities[0].split(''),
                                this.state.gameInfo.hands[currentMove.playerIdx],
                            )
                        ) : (
                            <NotInterestedIcon fontSize="large" />
                        )}
                    </div>
                    {probabilities ? (
                        <div className={'non-card ' + this.state.gameInfo.toggleFade}>
                            <span>
                                {probabilityItemType === 'Rule'
                                    ? 'Rule Based'
                                    : probabilityItemType === 'Probability'
                                    ? `Probability ${(probabilities[1] * 100).toFixed(2)}%`
                                    : `Expected payoff: ${probabilities[1].toFixed(4)}`}
                            </span>
                        </div>
                    ) : (
                        ''
                    )}
                </div>
            );
        } else {
            return <span className={'waiting'}>Waiting...</span>;
        }
    }

    go2PrevGameState() {
        let gameInfo = deepCopy(this.gameStateHistory[this.state.gameInfo.turn - 1]);
        gameInfo.gameStatus = 'paused';
        gameInfo.toggleFade = '';
        this.setState({ gameInfo: gameInfo });
    }

    go2NextGameState() {
        let gameInfo = this.generateNewState();
        if (gameInfo.gameStatus === 'over') return;
        gameInfo.gameStatus = 'paused';
        gameInfo.toggleFade = '';
        this.setState({ gameInfo: gameInfo });
    }

    handleCloseGameEndDialog() {
        this.setState({ gameEndDialog: false, gameEndDialogText: '' });
    }

    render() {
        let sliderValueText = (value) => {
            return value;
        };
        const gameSpeedMarks = [
            {
                value: -3,
                label: 'x0.125',
            },
            {
                value: -2,
                label: 'x0.25',
            },
            {
                value: -1,
                label: 'x0.5',
            },
            {
                value: 0,
                label: 'x1',
            },
            {
                value: 1,
                label: 'x2',
            },
            {
                value: 2,
                label: 'x4',
            },
            {
                value: 3,
                label: 'x8',
            },
        ];
        const playerInfo = this.state.gameInfo.playerInfo || [];
        const mergedChatLog = this.mergeChatLog(this.state.chatLog || [], this.moveHistory || []);
        const chatBubbles = this.buildChatBubbleMap(mergedChatLog);
        const controls = this.computeHumanControls();
        const isHumanTurn = this.isHumanTurn();
        const currentHistoryIndex = this.state.gameInfo.turn - 1;
        const chatEntries = mergedChatLog.map((entry, idx) => {
            const rawPlayerId = entry.player_id || entry.playerId || entry.player;
            const rawPlayerIdx = entry.player_idx ?? entry.playerIdx;
            let playerName = rawPlayerId ? String(rawPlayerId) : 'Player';
            if (rawPlayerIdx !== undefined) {
                const parsedIdx = Number(rawPlayerIdx);
                if (!Number.isNaN(parsedIdx)) {
                    playerName = this.resolvePlayerName(parsedIdx, playerInfo);
                }
            } else if (typeof rawPlayerId === 'string' && rawPlayerId.startsWith('player_')) {
                const parsedIdx = Number(rawPlayerId.split('_')[1]);
                if (!Number.isNaN(parsedIdx)) {
                    playerName = this.resolvePlayerName(parsedIdx, playerInfo);
                }
            }
            const text = entry.text || entry.message || entry.chat || '';
            return {
                key: `chat-${idx}`,
                playerName,
                text: String(text),
            };
        });
        const visibleHistory = (this.moveHistory || []).slice(0, Math.max(this.state.gameInfo.turn, 0));
        const historyEntries = visibleHistory.map((entry, idx) => {
            const rawPlayerIdx = entry.playerIdx ?? entry.player_idx ?? entry.playerId ?? entry.player_id;
            const parsedIdx = rawPlayerIdx !== undefined ? Number(rawPlayerIdx) : undefined;
            const playerName = this.resolvePlayerName(parsedIdx, playerInfo);
            const moveText = this.formatMoveText(
                entry.move !== undefined
                    ? entry.move
                    : entry.action_cards !== undefined
                      ? entry.action_cards
                      : entry.action_text !== undefined
                        ? entry.action_text
                        : entry.action,
            );
            const displayMove = moveText === 'pass' ? 'Pass' : moveText;
            const chatText = entry.chat || entry.chat_text || '';
            return {
                key: `history-${idx}`,
                index: idx + 1,
                playerName,
                moveText: displayMove || 'Unknown',
                chatText: String(chatText),
                isActive: idx === currentHistoryIndex,
            };
        });

        return (
            <div>
                <Dialog
                    open={this.state.gameEndDialog}
                    onClose={() => {
                        this.handleCloseGameEndDialog();
                    }}
                    aria-labelledby="alert-dialog-title"
                    aria-describedby="alert-dialog-description"
                >
                    <DialogTitle id="alert-dialog-title" style={{ width: '200px' }}>
                        {'Game Ends!'}
                    </DialogTitle>
                    <DialogContent>
                        <DialogContentText id="alert-dialog-description">
                            {this.state.gameEndDialogText}
                        </DialogContentText>
                    </DialogContent>
                    <DialogActions>
                        <Button
                            onClick={() => {
                                this.handleCloseGameEndDialog();
                            }}
                            color="primary"
                            autoFocus
                        >
                            OK
                        </Button>
                    </DialogActions>
                </Dialog>
                <div className={'doudizhu-view-container'}>
                    <Layout.Row style={{ height: '640px' }}>
                        <Layout.Col style={{ height: '100%' }} span="17">
                            <div style={{ height: '100%' }}>
                                <Paper className={'doudizhu-gameboard-paper'} elevation={3}>
                                    <DoudizhuGameBoard
                                        playerInfo={this.state.gameInfo.playerInfo}
                                        hands={this.state.gameInfo.hands}
                                        latestAction={this.state.gameInfo.latestAction}
                                        mainPlayerId={this.state.gameInfo.mainViewerId}
                                        currentPlayer={this.state.gameInfo.currentPlayer}
                                        considerationTime={this.state.gameInfo.considerationTime}
                                        considerationTimeMax={this.initConsiderationTime}
                                        timerMode="countup"
                                        turn={this.state.gameInfo.turn}
                                        runNewTurn={(prevTurn) => this.runNewTurn(prevTurn)}
                                        toggleFade={this.state.gameInfo.toggleFade}
                                        gameStatus={this.state.gameInfo.gameStatus}
                                        chatBubbles={chatBubbles}
                                        gamePlayable={isHumanTurn}
                                        showFinishButton={this.fastFinishEnabled}
                                        showCardBack={this.humanPlayEnabled && this.liveReplay}
                                        selectedCards={this.state.selectedCards}
                                        handleSelectedCards={(cards) => this.handleSelectedCards(cards)}
                                        handleMainPlayerAct={(action) => this.handleMainPlayerAct(action)}
                                        isPassDisabled={!controls.canPass || this.state.pendingAction}
                                        isHintDisabled={!controls.canHint || this.state.pendingAction}
                                        isPlayDisabled={!controls.canPlay || this.state.pendingAction}
                                        isFinishDisabled={!isHumanTurn || this.state.pendingAction}
                                    />
                                </Paper>
                            </div>
                        </Layout.Col>
                        <Layout.Col span="7" style={{ height: '100%' }}>
                            <Paper className={'doudizhu-probability-paper'} elevation={3}>
                                <div className={'probability-player'}>
                                    {this.state.gameInfo.playerInfo.length > 0 ? (
                                        <span>
                                            Current Player: {this.state.gameInfo.currentPlayer}
                                            <br />
                                            Role:{' '}
                                            {this.state.gameInfo.playerInfo[this.state.gameInfo.currentPlayer].role}
                                        </span>
                                    ) : (
                                        <span>Waiting...</span>
                                    )}
                                </div>
                                <Divider />
                                <div className={'probability-table'}>
                                    <div className={'probability-item'}>{this.computeProbabilityItem(0)}</div>
                                    <div className={'probability-item'}>{this.computeProbabilityItem(1)}</div>
                                    <div className={'probability-item'}>{this.computeProbabilityItem(2)}</div>
                                </div>
                            </Paper>
                        </Layout.Col>
                    </Layout.Row>
                    <Layout.Row style={{ height: '220px' }}>
                        <Layout.Col span="12" style={{ height: '100%' }}>
                            <Paper className={'doudizhu-chat-paper'} elevation={3}>
                                <div className="doudizhu-panel-header">Table Talk</div>
                                <div className="doudizhu-panel-body" ref={this.chatPanelRef}>
                                    {chatEntries.length > 0 ? (
                                        chatEntries.map((entry) => (
                                            <div className="doudizhu-chat-entry" key={entry.key}>
                                                <span className="chat-player">{entry.playerName}</span>
                                                <span className="chat-text">{entry.text}</span>
                                            </div>
                                        ))
                                    ) : (
                                        <div className="doudizhu-panel-empty">No chat yet</div>
                                    )}
                                </div>
                                {this.humanPlayEnabled && (
                                    <div className="doudizhu-chat-input">
                                        <input
                                            type="text"
                                            value={this.state.chatDraft}
                                            onChange={(event) => this.handleChatDraftChange(event.target.value)}
                                            placeholder="Table talk (send anytime)"
                                            onKeyDown={(event) => {
                                                if (event.key === 'Enter') {
                                                    event.preventDefault();
                                                    this.submitHumanChat();
                                                }
                                            }}
                                            disabled={this.state.pendingAction || this.state.pendingChat}
                                        />
                                        <Button
                                            variant="contained"
                                            color="primary"
                                            disabled={
                                                this.state.pendingAction ||
                                                this.state.pendingChat ||
                                                !(this.state.chatDraft || '').trim()
                                            }
                                            onClick={() => this.submitHumanChat()}
                                        >
                                            Send
                                        </Button>
                                    </div>
                                )}
                            </Paper>
                        </Layout.Col>
                        <Layout.Col span="12" style={{ height: '100%' }}>
                            <Paper className={'doudizhu-history-paper'} elevation={3}>
                                <div className="doudizhu-panel-header">Move History</div>
                                <div className="doudizhu-panel-body" ref={this.historyPanelRef}>
                                    {historyEntries.length > 0 ? (
                                        historyEntries.map((entry) => (
                                            <div
                                                className={`doudizhu-history-entry${
                                                    entry.isActive ? ' active' : ''
                                                }`}
                                                key={entry.key}
                                            >
                                                <span className="history-index">{entry.index}.</span>
                                                <span className="history-player">{entry.playerName}</span>
                                                <span className="history-move">{entry.moveText}</span>
                                                {entry.chatText ? (
                                                    <span className="history-chat">"{entry.chatText}"</span>
                                                ) : null}
                                            </div>
                                        ))
                                    ) : (
                                        <div className="doudizhu-panel-empty">No moves yet</div>
                                    )}
                                </div>
                            </Paper>
                        </Layout.Col>
                    </Layout.Row>
                    <div className="progress-bar">
                        <LinearProgress variant="determinate" value={this.state.gameInfo.completedPercent} />
                    </div>
                    <Loading loading={this.state.fullScreenLoading}>
                        <div className="game-controller">
                            <Paper className={'game-controller-paper'} elevation={3}>
                                <Layout.Row style={{ height: '51px' }}>
                                    <Layout.Col span="7" style={{ height: '51px', lineHeight: '48px' }}>
                                        <div>
                                            <Button
                                                variant="contained"
                                                color="primary"
                                                disabled={
                                                    this.state.gameInfo.gameStatus !== 'paused' ||
                                                    this.state.gameInfo.turn === 0
                                                }
                                                onClick={() => {
                                                    this.go2PrevGameState();
                                                }}
                                            >
                                                <SkipPreviousIcon />
                                            </Button>
                                            {this.gameStatusButton(this.state.gameInfo.gameStatus)}
                                            <Button
                                                variant="contained"
                                                color="primary"
                                                disabled={this.state.gameInfo.gameStatus !== 'paused'}
                                                onClick={() => {
                                                    this.go2NextGameState();
                                                }}
                                            >
                                                <SkipNextIcon />
                                            </Button>
                                        </div>
                                    </Layout.Col>
                                    <Layout.Col span="1" style={{ height: '100%', width: '1px' }}>
                                        <Divider orientation="vertical" />
                                    </Layout.Col>
                                    <Layout.Col
                                        span="3"
                                        style={{
                                            height: '51px',
                                            lineHeight: '51px',
                                            marginLeft: '-1px',
                                            marginRight: '-1px',
                                        }}
                                    >
                                        <div style={{ textAlign: 'center' }}>{`Turn ${this.state.gameInfo.turn}`}</div>
                                    </Layout.Col>
                                    <Layout.Col span="1" style={{ height: '100%', width: '1px' }}>
                                        <Divider orientation="vertical" />
                                    </Layout.Col>
                                    <Layout.Col span="14">
                                        <div>
                                            <label className={'form-label-left'}>Game Speed</label>
                                            <div style={{ marginLeft: '100px', marginRight: '10px' }}>
                                                <Slider
                                                    value={this.state.gameSpeed}
                                                    getAriaValueText={sliderValueText}
                                                    onChange={(e, newVal) => {
                                                        this.changeGameSpeed(newVal);
                                                    }}
                                                    aria-labelledby="discrete-slider-custom"
                                                    step={1}
                                                    min={-3}
                                                    max={3}
                                                    track={false}
                                                    valueLabelDisplay="off"
                                                    marks={gameSpeedMarks}
                                                />
                                            </div>
                                        </div>
                                    </Layout.Col>
                                </Layout.Row>
                            </Paper>
                        </div>
                    </Loading>
                </div>
            </div>
        );
    }
}

export default DoudizhuReplayView;

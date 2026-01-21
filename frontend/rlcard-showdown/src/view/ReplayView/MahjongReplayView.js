import React from 'react';
import { withRouter } from 'react-router-dom';
import qs from 'query-string';
import MahjongGameBoard from '../../components/GameBoard/MahjongGameBoard';
import { apiUrl, actionUrl } from '../../utils/config';
import axios from 'axios';
import { Message } from 'element-react';
import { Avatar, Chip, Button, Slider } from '@material-ui/core';

import PlayArrowRoundedIcon from '@material-ui/icons/PlayArrowRounded';
import PauseCircleOutlineRoundedIcon from '@material-ui/icons/PauseCircleOutlineRounded';
import SkipNextIcon from '@material-ui/icons/SkipNext';
import SkipPreviousIcon from '@material-ui/icons/SkipPrevious';

import '../../assets/mahjong.scss';

class MahjongReplayView extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            replayData: null,
            step: 0,
            autoPlay: false,
            error: null,
            showHands: true,
            gameSpeed: 1, // 1 = 1x speed
            legalMoves: [],
            activePlayerId: null,
            pendingAction: false,
            selectedTile: null,
            selectedTileKey: null,
            pendingChat: false,
            chatDraft: '',
            nowMs: Date.now(),
        };
        this.intervalId = null;
        this.autoPlayId = null;
        this.clockInterval = null;
        this.chatLogRef = React.createRef();
        this.historyPanelRef = React.createRef();
        this.humanPlayEnabled = false;
        this.liveFollowEnabled = false;
        this.lastMoveCount = null;
        this.lastTileClickAt = 0;
        this.lastTileClickKey = null;
        this.doubleClickWindowMs = 450;
        this.lastActionSubmitAt = 0;
        this.doubleClickGuardMs = 200;
        this.lastWinnerId = null;
        this.lastEndReason = null;
    }

    componentDidMount() {
        // Parse showHands from URL if needed, or default
        const { location } = this.props;
        const query = qs.parse(location.search);
        const mode = query.mode || 'ai';
        const playEnabled = this.parseLiveFlag(query.play || query.human || query.interactive);
        this.humanPlayEnabled = playEnabled;
        this.liveFollowEnabled = playEnabled || this.parseLiveFlag(query.live || query.follow);
        if (mode === 'human') {
            this.setState({ showHands: false });
        } else {
            this.setState({ showHands: true });
        }
        const pollMs = playEnabled ? 1000 : 5000;
        this.fetchReplay();
        this.intervalId = setInterval(this.fetchReplay, pollMs);
        this.clockInterval = setInterval(() => {
            this.setState({ nowMs: Date.now() });
        }, 500);
    }

    componentWillUnmount() {
        if (this.intervalId) clearInterval(this.intervalId);
        if (this.autoPlayId) clearInterval(this.autoPlayId);
        if (this.clockInterval) clearInterval(this.clockInterval);
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

    resolveActionPayload(actionText) {
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
        if (this.state.activePlayerId) {
            payload.player_id = this.state.activePlayerId;
        }
        return payload;
    }

    resolveChatEndpoint() {
        const actionEndpoint = this.resolveActionEndpoint();
        if (actionEndpoint.endsWith('/tournament/action')) {
            return actionEndpoint.replace('/tournament/action', '/tournament/chat');
        }
        const trimmed = actionEndpoint.endsWith('/') ? actionEndpoint.slice(0, -1) : actionEndpoint;
        return `${trimmed}/tournament/chat`;
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
        if (this.humanPlayEnabled) {
            payload.player_id = 'player_0';
        } else if (this.state.activePlayerId) {
            payload.player_id = this.state.activePlayerId;
        }
        return payload;
    }

    resolveActivePlayerId(payload) {
        const raw =
            payload.active_player_id ??
            payload.activePlayerId ??
            payload.active_player ??
            payload.activePlayer ??
            payload.active_player_idx ??
            payload.activePlayerIdx ??
            null;
        if (raw === null || raw === undefined) {
            return null;
        }
        if (typeof raw === 'number') {
            return `player_${raw}`;
        }
        const text = String(raw);
        if (/^\d+$/.test(text)) {
            return `player_${text}`;
        }
        return text;
    }

    resolveLegalMoves(payload) {
        const legalMoves = payload.legal_moves || payload.legalMoves || payload.legalActions || [];
        return Array.isArray(legalMoves) ? legalMoves : [];
    }

    resolvePlayerIndex(playerId) {
        if (playerId === null || playerId === undefined) {
            return null;
        }
        const text = String(playerId);
        if (text.startsWith('player_')) {
            const parsed = parseInt(text.replace('player_', ''), 10);
            return Number.isNaN(parsed) ? null : parsed;
        }
        const parsed = parseInt(text, 10);
        return Number.isNaN(parsed) ? null : parsed;
    }

    isTileAction(actionText) {
        const raw = String(actionText || '').trim();
        if (!raw) {
            return false;
        }
        const upper = raw.toUpperCase();
        if (/^[BCD][1-9]$/.test(upper)) {
            return true;
        }
        const titled = raw.charAt(0).toUpperCase() + raw.slice(1).toLowerCase();
        return ['East', 'South', 'West', 'North', 'Green', 'Red', 'White'].includes(titled);
    }

    isHumanTurn() {
        if (!this.humanPlayEnabled || !this.state.replayData) {
            return false;
        }
        if (
            this.state.replayData.winner ||
            this.state.replayData.end_reason ||
            this.state.replayData.endReason ||
            this.state.replayData.result
        ) {
            return false;
        }
        const moves = this.state.replayData.moves || [];
        const atTail = this.state.step >= moves.length;
        return atTail && this.state.activePlayerId === 'player_0';
    }

    clearSelection() {
        this.setState({ selectedTile: null, selectedTileKey: null });
    }

    handleTileClick(tileCode, tileKey) {
        if (!this.isHumanTurn()) {
            return;
        }
        const now = Date.now();
        if (this.lastTileClickKey === tileKey && (now - this.lastTileClickAt) <= this.doubleClickWindowMs) {
            this.lastTileClickAt = 0;
            this.lastTileClickKey = null;
            this.handleTileDoubleClick(tileCode);
            return;
        }
        this.lastTileClickAt = now;
        this.lastTileClickKey = tileKey;
        const legalMoves = Array.isArray(this.state.legalMoves) ? this.state.legalMoves : [];
        const legalTileSet = new Set(
            legalMoves.filter((move) => this.isTileAction(move)).map((move) => String(move).toLowerCase()),
        );
        const normalized = String(tileCode || '').toLowerCase();
        if (!legalTileSet.has(normalized)) {
            Message({
                message: 'Selected tile is not a legal action',
                type: 'warning',
                showClose: true,
            });
            return;
        }
        if (this.state.selectedTileKey === tileKey) {
            this.clearSelection();
            return;
        }
        this.setState({ selectedTile: tileCode, selectedTileKey: tileKey });
    }

    handleTileDoubleClick(tileCode) {
        if (!this.isHumanTurn()) {
            return;
        }
        const now = Date.now();
        if (now - this.lastActionSubmitAt < this.doubleClickGuardMs) {
            return;
        }
        const legalMoves = Array.isArray(this.state.legalMoves) ? this.state.legalMoves : [];
        const legalTileSet = new Set(
            legalMoves.filter((move) => this.isTileAction(move)).map((move) => String(move).toLowerCase()),
        );
        const normalized = String(tileCode || '').toLowerCase();
        if (!legalTileSet.has(normalized)) {
            Message({
                message: 'Selected tile is not a legal action',
                type: 'warning',
                showClose: true,
            });
            return;
        }
        this.lastActionSubmitAt = now;
        this.clearSelection();
        this.submitHumanAction(tileCode);
    }

    submitHumanAction(actionText) {
        if (!actionText || this.state.pendingAction) {
            return;
        }
        const endpoint = this.resolveActionEndpoint();
        const payload = this.resolveActionPayload(actionText);
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
                this.setState({ pendingAction: false });
                setTimeout(() => this.fetchReplay(), 250);
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
                this.setState({ chatDraft: '' });
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

    handleChatDraftChange(value) {
        this.setState({ chatDraft: value });
    }

    componentDidUpdate(prevProps, prevState) {
        // Handle Auto Play
        if (this.state.autoPlay && !prevState.autoPlay) {
            const { gameSpeed } = this.state;
            this.autoPlayId = setInterval(() => {
                const { step, replayData } = this.state;
                if (replayData && replayData.moves && step < replayData.moves.length) {
                    this.setState({ step: step + 1 });
                }
            }, 1000 / Math.pow(2, gameSpeed)); 
        } else if (!this.state.autoPlay && prevState.autoPlay) {
            if (this.autoPlayId) clearInterval(this.autoPlayId);
        }
        
        // Auto scroll
        if (this.chatLogRef.current) {
             this.chatLogRef.current.scrollTop = this.chatLogRef.current.scrollHeight;
        }
        if (this.historyPanelRef.current) {
             this.historyPanelRef.current.scrollTop = this.historyPanelRef.current.scrollHeight;
        }
    }

    fetchReplay = () => {
        const { location } = this.props;
        const query = qs.parse(location.search);
        const replayParam = query.replay_path || 'mahjong_replay.json'; 
        
        let fetchUrl = replayParam;
        if (!replayParam.startsWith('http')) {
             fetchUrl = `${apiUrl}/tournament/replay?replay_path=${encodeURIComponent(replayParam)}`;
        }

        axios.get(fetchUrl)
            .then(res => {
                const newData = res.data;
                const newMoves = newData.moves || [];
                const activePlayerId = this.resolveActivePlayerId(newData);
                const legalMoves = this.resolveLegalMoves(newData);
                const winnerId = newData.winner || null;
                const endReason = newData.end_reason || newData.endReason || null;
                if (endReason && endReason !== this.lastEndReason) {
                    const winnerIdx = winnerId ? this.resolvePlayerIndex(winnerId) : null;
                    const winnerLabel = winnerId ? (winnerIdx !== null ? `P${winnerIdx}` : String(winnerId)) : '';
                    const reasonLabel = endReason === 'hu' ? 'Hu' : endReason;
                    const messageText = winnerLabel
                        ? `Game Over: ${reasonLabel} (${winnerLabel})`
                        : `Game Over: ${reasonLabel}`;
                    Message({
                        message: messageText,
                        type: endReason === 'hu' ? 'success' : 'warning',
                        duration: 6000,
                        showClose: true,
                    });
                    this.lastEndReason = endReason;
                    this.lastWinnerId = winnerId;
                } else if (winnerId && winnerId !== this.lastWinnerId && !endReason) {
                    const winnerIdx = this.resolvePlayerIndex(winnerId);
                    const winnerLabel = winnerIdx !== null ? `P${winnerIdx}` : String(winnerId);
                    Message({
                        message: `Hu! Winner: ${winnerLabel}`,
                        type: 'success',
                        duration: 5000,
                        showClose: true,
                    });
                    this.lastWinnerId = winnerId;
                }
                const moveCount = newMoves.length;
                const resetSelection = this.lastMoveCount !== null && moveCount !== this.lastMoveCount;
                // Update step if new moves arrived and we were at the end? 
                // For now just update data.
                this.setState(prevState => {
                    let nextStep = prevState.step;
                    // If we want to auto-follow:
                    // if (prevState.step === (prevState.replayData?.moves?.length || 0)) {
                    //     nextStep = newMoves.length;
                    // }
                    // Actually, let's just keep step unless it's out of bounds?
                    // Or if we specifically want to jump to end?
                    // Let's just update data.
                    if (this.liveFollowEnabled) {
                        nextStep = newMoves.length;
                    } else if (newMoves.length < prevState.step) {
                        nextStep = newMoves.length;
                    }
                    const shouldClearSelection =
                        resetSelection || (activePlayerId && activePlayerId !== prevState.activePlayerId);
                    return {
                        replayData: newData,
                        step: nextStep,
                        error: null,
                        legalMoves,
                        activePlayerId,
                        selectedTile: shouldClearSelection ? null : prevState.selectedTile,
                        selectedTileKey: shouldClearSelection ? null : prevState.selectedTileKey,
                    };
                });
                this.lastMoveCount = moveCount;
            })
            .catch(err => {
                console.error("Error fetching replay:", err);
                let errorMsg = "Failed to load replay data.";
                if (err.message === "Network Error") {
                    errorMsg = "Network Error. Please check if the backend service is running.";
                } else if (err.response && err.response.status === 404) {
                    errorMsg = "Replay file not found.";
                }
                this.setState({ error: errorMsg });
            });
    }

    toggleShowHands = () => {
        this.setState(prev => ({ showHands: !prev.showHands }));
    }

    changeGameSpeed(newVal) {
        this.setState({ gameSpeed: newVal });
        // detailed handling of interval reset if playing is handled in DidUpdate roughly or we reset here
        if (this.state.autoPlay) {
            clearInterval(this.autoPlayId);
            this.autoPlayId = setInterval(() => {
                const { step, replayData } = this.state;
                if (replayData && replayData.moves && step < replayData.moves.length) {
                    this.setState({ step: step + 1 });
                }
            }, 1000 / Math.pow(2, newVal));
        }
    }

    render() {
        const {
            replayData,
            step,
            autoPlay,
            error,
            showHands,
            gameSpeed,
            legalMoves,
            activePlayerId,
            pendingAction,
            selectedTile,
            selectedTileKey,
            pendingChat,
            chatDraft,
            nowMs,
        } = this.state;
        const { location } = this.props;
        const query = qs.parse(location.search);
        
        if (error && !replayData) return <div style={{padding: 20}}>Error: {error}</div>;
        if (!replayData) return <div style={{padding: 20}}>Loading Mahjong Replay...</div>;

        const endReason = replayData.end_reason || replayData.endReason || null;
        const resultLabel = replayData.result || replayData.game_result || null;
        const remainingTiles = replayData.remaining_tiles ?? replayData.remainingTiles ?? null;
        const remainingTilesValue = Number(remainingTiles);
        const noTilesLeft = Number.isFinite(remainingTilesValue) && remainingTilesValue <= 0;
        const gameOver = Boolean(endReason || resultLabel || replayData.winner || noTilesLeft);
        const activePlayerIdx = this.resolvePlayerIndex(activePlayerId);
        const legalMovesList = Array.isArray(legalMoves) ? legalMoves.map(String) : [];
        const legalTileMoves = legalMovesList.filter((move) => this.isTileAction(move));
        const legalTileSet = new Set(legalTileMoves.map((move) => String(move).toLowerCase()));
        const specialMoves = [];
        const specialSeen = new Set();
        legalMovesList.forEach((move) => {
            if (!move || this.isTileAction(move)) {
                return;
            }
            const key = String(move).toLowerCase();
            if (specialSeen.has(key)) {
                return;
            }
            specialSeen.add(key);
            specialMoves.push(String(move));
        });
        const humanTurn = this.isHumanTurn();
        const canPlaySelected =
            humanTurn &&
            !!selectedTile &&
            legalTileSet.has(String(selectedTile).toLowerCase()) &&
            !pendingAction;
        const actionDisabled = !humanTurn || pendingAction;
        const actionLabelMap = {
            stand: 'Pass',
            pass: 'Pass',
            pong: 'Pong',
            chow: 'Chow',
            gong: 'Kong',
            hu: 'Hu',
        };
        const formatSpecialMove = (move) => {
            const key = String(move || '').toLowerCase();
            return actionLabelMap[key] || move;
        };
        const popupMoveSet = new Set(['pong', 'chow', 'gong', 'hu']);
        const popupMoves = specialMoves.filter((move) => popupMoveSet.has(String(move).toLowerCase()));
        const showActionOverlay = humanTurn && popupMoves.length > 0;

        const moves = replayData.moves || [];
        const resolveTimestamp = (entry) => {
            if (!entry) {
                return null;
            }
            const raw =
                entry.timestamp_ms ??
                entry.timestampMs ??
                entry.time_ms ??
                entry.time ??
                entry.ts ??
                null;
            const parsed = Number(raw);
            return Number.isNaN(parsed) ? null : parsed;
        };
        const startTimeRaw =
            replayData.start_time_ms ??
            replayData.startTimeMs ??
            replayData.start_time ??
            replayData.startTime ??
            null;
        const startTimeMs = Number(startTimeRaw);
        const hasStartTime = !Number.isNaN(startTimeMs);
        const lastMoveTimestamp =
            moves.length > 0 ? resolveTimestamp(moves[moves.length - 1]) : null;
        const turnStartMs = Number.isFinite(lastMoveTimestamp)
            ? lastMoveTimestamp
            : hasStartTime
              ? startTimeMs
              : Date.now();
        const currentNow = Number.isFinite(nowMs) ? nowMs : Date.now();
        const timerNow = gameOver ? (Number.isFinite(lastMoveTimestamp) ? lastMoveTimestamp : currentNow) : currentNow;
        const thinkingByPlayer = { 0: 0, 1: 0, 2: 0, 3: 0 };
        let prevTimestamp = hasStartTime ? startTimeMs : null;
        moves.forEach((move) => {
            const timestamp = resolveTimestamp(move);
            if (!Number.isFinite(timestamp)) {
                return;
            }
            const playerId = move.player_id ?? move.playerId ?? move.player ?? null;
            const pid = this.resolvePlayerIndex(playerId);
            if (pid !== null && prevTimestamp !== null) {
                const delta = Math.max(0, timestamp - prevTimestamp);
                thinkingByPlayer[pid] = delta;
            }
            prevTimestamp = timestamp;
        });
        if (activePlayerIdx !== null && !gameOver) {
            thinkingByPlayer[activePlayerIdx] = Math.max(0, timerNow - turnStartMs);
        }
        const formatDuration = (ms) => {
            const safeMs = Math.max(0, Math.floor(ms));
            const totalSec = Math.floor(safeMs / 1000);
            const minutes = Math.floor(totalSec / 60);
            const seconds = totalSec % 60;
            const mm = String(minutes).padStart(2, '0');
            const ss = String(seconds).padStart(2, '0');
            return `${mm}:${ss}`;
        };
        const currentMoves = moves.slice(0, step);
        const chatHistory = [];
        const moveHistory = [];
        const honorTiles = new Set(['East', 'South', 'West', 'North', 'Green', 'Red', 'White']);
        const normalizeTileCode = (rawCode) => {
            if (!rawCode) return null;
            const trimmed = String(rawCode).trim();
            if (!trimmed) return null;
            const upper = trimmed.toUpperCase();
            if (upper.length === 2 && ['B', 'C', 'D'].includes(upper[0]) && /\d/.test(upper[1])) {
                return upper;
            }
            const titled = trimmed.charAt(0).toUpperCase() + trimmed.slice(1).toLowerCase();
            if (honorTiles.has(titled)) {
                return titled;
            }
            return trimmed;
        };
        const isTileText = (text) => {
            const trimmed = String(text || '').trim();
            if (!trimmed) return false;
            const upper = trimmed.toUpperCase();
            if (/^[BCD][1-9]$/.test(upper)) return true;
            return /^(East|South|West|North|Green|Red|White)$/i.test(trimmed);
        };
        const extractActionCard = (text) => {
            const trimmed = String(text || '').trim();
            if (!trimmed) return null;
            const drawMatch = trimmed.match(/^(?:DREW|DRAW):?\s*([a-zA-Z0-9]+)$/i);
            if (drawMatch) {
                return normalizeTileCode(drawMatch[1]);
            }
            if (isTileText(trimmed)) {
                return normalizeTileCode(trimmed);
            }
            return null;
        };
        const resolveActionCard = (move) => {
            if (!move) return null;
            if (move.action_card) {
                return normalizeTileCode(move.action_card);
            }
            return extractActionCard(move.action_text);
        };
        // Reconstruct Board State
        // In a real app, we might need a heavy reducer. 
        // Here, we assume the backend might provide snapshots OR we rely on `MahjongGameBoard` 
        // to render based on the accumulated moves if it supported that.
        // BUT, `MahjongGameBoard` takes `hands`, `piles`, `discards`.
        // We need to parse frames.
        // Luckily, our backend `replayData` often includes `hand_history` snapshots or we just use `current_hands` if it's live?
        // Let's assume `replayData.hand_history[step]` exists if it's a full recorded replay.
        // If not, we might fallback to `replayData.current_hands` (but that's only for the FINAL state).
        
        // For simplicity in this demo, let's assume `hand_history` is available or we just render "current" if step==max.
        // If `hand_history` is missing, we might need a local reducer. 
        // Let's check the data structure from previous knowledge (DouZero). Use snapshots if available.
        
        // Discards: We can compute from moves easily.
        const discards = []; 
        const chatBubbles = {}; 
        let lastAction = "Game Start";
        const chatKeys = new Set();
        const pushChatEntry = (pid, msg, timestamp) => {
            const key = `${pid ?? 'unknown'}::${msg ?? ''}::${timestamp ?? 'na'}`;
            if (chatKeys.has(key)) {
                return;
            }
            chatKeys.add(key);
            chatHistory.push({ pid, msg, timestamp });
        };

        currentMoves.forEach((m, idx) => {
            const pidStr = m.player_id || "player_0";
            const pid = parseInt(pidStr.replace('player_', '')) || 0;
            const txt = m.action_text || "";
            const actionCard = resolveActionCard(m);
            const isDraw = String(txt || "").match(/^(?:DREW|DRAW):?\s*([a-zA-Z0-9]+)/i);
            lastAction = `${pidStr}: ${txt}`;
            
            moveHistory.push({ pid, action: txt, step: m.step || (idx + 1) });

            // Naive discard parsing: "B3", "East", etc.
            if (actionCard && !isDraw) {
                 discards.push(actionCard);
            }
            if (m.chat) {
                const chatTimestamp = resolveTimestamp(m);
                const showBubble = Number.isFinite(chatTimestamp)
                    ? (currentNow - chatTimestamp) <= 5000
                    : (step - idx) < 5;
                if (showBubble) {
                    chatBubbles[pid] = m.chat;
                }
                pushChatEntry(pid, m.chat, chatTimestamp ?? null);
            }
        });
        const extraChatLog = replayData.chat_log || replayData.chatLog || [];
        extraChatLog.forEach((entry) => {
            if (!entry) {
                return;
            }
            const playerId = entry.player_id ?? entry.playerId ?? entry.player ?? "player_0";
            const pid = this.resolvePlayerIndex(playerId) ?? 0;
            const msg = entry.text ?? entry.chat ?? entry.message ?? "";
            const timestamp =
                entry.timestamp_ms ??
                entry.timestampMs ??
                entry.time_ms ??
                entry.time ??
                null;
            if (!msg) {
                return;
            }
            const showBubble = Number.isFinite(Number(timestamp))
                ? (currentNow - Number(timestamp)) <= 5000
                : false;
            if (showBubble) {
                chatBubbles[pid] = msg;
            }
            pushChatEntry(pid, msg, Number(timestamp));
        });
        chatHistory.sort((a, b) => {
            if (a.timestamp === null && b.timestamp === null) {
                return 0;
            }
            if (a.timestamp === null) {
                return -1;
            }
            if (b.timestamp === null) {
                return 1;
            }
            return a.timestamp - b.timestamp;
        });

        // Hands & Piles
        // Ideally: replayData.snapshots[step]
        // Fallback: use replayData.current_hands and assume we are at end? 
        // Or if we don't have per-step snapshots, the board will look static (except discards).
        // Let's try to find `hand_history`.
        const handsSource = (replayData.hand_history && replayData.hand_history[step]) 
                          ? replayData.hand_history[step] 
                          : (replayData.current_hands || {});
        const prevHandsSource = (replayData.hand_history && step > 0 && replayData.hand_history[step - 1])
            ? replayData.hand_history[step - 1]
            : null;
        const resolveHandList = (source, pid) => {
            if (!source) return null;
            const data = source[pid] ?? source[String(pid)];
            if (!data) return null;
            if (Array.isArray(data)) return data;
            if (Array.isArray(data.hand)) return data.hand;
            return null;
        };
        const countTiles = (tiles) => {
            const counts = {};
            (tiles || []).forEach((tile) => {
                const normalized = normalizeTileCode(tile);
                if (!normalized || normalized.toLowerCase() === 'back') {
                    return;
                }
                counts[normalized] = (counts[normalized] || 0) + 1;
            });
            return counts;
        };
        const resolveDrawTile = (prevHand, currHand) => {
            if (!Array.isArray(prevHand) || !Array.isArray(currHand)) {
                return null;
            }
            if (currHand.length !== prevHand.length + 1) {
                return null;
            }
            const prevCounts = countTiles(prevHand);
            const currCounts = countTiles(currHand);
            let drawn = null;
            for (const [tile, count] of Object.entries(currCounts)) {
                const delta = count - (prevCounts[tile] || 0);
                if (delta === 1) {
                    if (drawn) {
                        return null;
                    }
                    drawn = tile;
                } else if (delta > 1) {
                    return null;
                }
            }
            return drawn;
        };
        const drawTileByPlayer = {};
        const prevHand0 = resolveHandList(prevHandsSource, 0);
        const currHand0 = resolveHandList(handsSource, 0);
        const drawTile0 = resolveDrawTile(prevHand0, currHand0);
        if (drawTile0) {
            drawTileByPlayer[0] = drawTile0;
        }

        const parseHand = (pid) => {
             const data = handsSource[pid] ?? handsSource[String(pid)];
             if (!data) return Array(13).fill('Back');
             const handList = data.hand || [];
             if (!showHands && pid !== 0) return Array(handList.length).fill('Back');
             return handList;
        };
        const parsePile = (pid) => (handsSource[pid] ?? handsSource[String(pid)])?.pile || [];

        const hands = { 0: parseHand(0), 1: parseHand(1), 2: parseHand(2), 3: parseHand(3) };
        const piles = { 0: parsePile(0), 1: parsePile(1), 2: parsePile(2), 3: parsePile(3) };
        
        const sliderValueText = (value) => `x${Math.pow(2, value)}`;
        const gameSpeedMarks = [
            { value: -2, label: 'x0.25' },
            { value: -1, label: 'x0.5' },
            { value: 0, label: 'x1' },
            { value: 1, label: 'x2' },
            { value: 2, label: 'x4' },
            { value: 3, label: 'x8' },
        ];

        // Tsumogiri (Draw-and-Play) Detection Logic
        let isTsumogiri = false;
        let lastPlayedTile = null;
        let drawingCard = null;
        let drawingPlayerPos = null; // We need to calculate this too if we want to be fully robust, but GameBoard handles parsing too. 
        // Ideally, we pass everything explicit.

        if (step > 0 && currentMoves.length > 0) {
            const lastMove = currentMoves[currentMoves.length - 1]; // Move at step N
            const prevMove = step > 1 ? currentMoves[currentMoves.length - 2] : null;

            // Parse Last Action
            const actionText = lastMove.action_text || "";
            const pidStr = lastMove.player_id || "player_0";
            const pid = parseInt(pidStr.replace('player_', '')) || 0;
            
            // Check if it's a Play action
            const resolvedCard = resolveActionCard(lastMove);
            const isDraw = actionText.match(/^(?:DREW|DRAW):?\s*([a-zA-Z0-9]+)/i);

            if (resolvedCard && !isDraw) {
                lastPlayedTile = resolvedCard;
                // Check Previous move for Tsumogiri
                if (prevMove && prevMove.player_id === lastMove.player_id) {
                     const prevActionText = prevMove.action_text || "";
                     const prevDrawMatch = prevActionText.match(/^(?:DREW|DRAW):?\s*([a-zA-Z0-9]+)/i);
                     if (prevDrawMatch) {
                         const drawnTile = prevDrawMatch[1];
                         // Compare tile codes loosely (case insensitive)
                         if (drawnTile.toLowerCase() === lastPlayedTile.toLowerCase()) {
                             isTsumogiri = true;
                         }
                     }
                }
            } else if (isDraw) {
                // If it is a DRAW action, we can explicitly pass it if GameBoard needs explanation
                // But GameBoard parses `lastAction` string for Draw animation.
                // We keep GameBoard's internal parsing for Draw, but override for Play/Tsumogiri.
            }
        }

        const hasMoreMoves = step < moves.length;
        const playStatus = gameOver
            ? 'Ended'
            : autoPlay
              ? (hasMoreMoves ? 'Playing' : 'Waiting')
              : (hasMoreMoves ? 'Paused' : 'Stopped');
        const winnerId = replayData.winner || null;
        const winnerIdx = winnerId ? this.resolvePlayerIndex(winnerId) : null;
        const winnerLabel = winnerId ? (winnerIdx !== null ? `P${winnerIdx}` : String(winnerId)) : '';
        const endLabel = endReason
            ? (endReason === 'hu' ? 'Hu' : endReason)
            : (resultLabel ? String(resultLabel) : '');
        const endMessage = (() => {
            if (!gameOver) {
                return '';
            }
            if (endReason === 'hu' || winnerId) {
                const name = winnerIdx !== null ? `玩家${winnerIdx}` : (winnerId ? String(winnerId) : '玩家');
                return `对局结束（${name}胡）`;
            }
            if (endReason === 'draw' || endReason === 'finished' || noTilesLeft) {
                return '对局结束（牌已摸完）';
            }
            if (endReason) {
                return `对局结束（${endReason}）`;
            }
            if (resultLabel) {
                return `对局结束（${resultLabel}）`;
            }
            return '对局结束';
        })();

        return (
            <div className="mahjong-replay-container">
                <div className="replay-header">
                    <h2>Mahjong Game Replay</h2>
                    <div className="header-actions">
                        <Chip label={`Turn: ${step}`} color="primary" variant="outlined" />
                        <Chip label={`Mode: ${query.mode || 'AI'}`} color="secondary" variant="outlined" />
                        <Chip label={`Play: ${playStatus}`} color="default" variant="outlined" />
                        {winnerLabel ? (
                            <Chip label={`Winner: ${winnerLabel}`} color="secondary" variant="outlined" />
                        ) : null}
                        {endLabel ? (
                            <Chip label={`Result: ${endLabel}`} color="secondary" variant="outlined" />
                        ) : null}
                        <Button
                            variant="outlined" 
                            color="primary"
                            onClick={this.toggleShowHands}
                            size="small"
                        >
                            {showHands ? "Hide Hands" : "Show Hands"}
                        </Button>
                    </div>
                </div>

                <div className="replay-main-content">
                    {/* Left: Game Board */}
                    <div className="board-section">
                        {gameOver && endMessage ? (
                            <div className="mahjong-end-banner">{endMessage}</div>
                        ) : null}
                        {showActionOverlay && (
                            <div className="mahjong-action-overlay">
                                <div className="overlay-title">Special Action</div>
                                <div className="overlay-buttons">
                                    {popupMoves.map((move) => (
                                        <Button
                                            key={move}
                                            variant="contained"
                                            color="secondary"
                                            size="small"
                                            disabled={actionDisabled}
                                            onClick={() => {
                                                this.clearSelection();
                                                this.submitHumanAction(move);
                                            }}
                                        >
                                            {formatSpecialMove(move)}
                                        </Button>
                                    ))}
                                </div>
                            </div>
                        )}
                        <MahjongGameBoard 
                            hands={hands}
                            piles={piles}
                            discards={discards}
                            chatBubbles={chatBubbles}
                            currentPlayer={activePlayerIdx ?? 0}
                            showAllHands={showHands}
                            lastAction={lastAction}
                            // New Props for Robustness
                            isTsumogiri={isTsumogiri}
                            lastPlayedTile={lastPlayedTile}
                            selectedTileKey={selectedTileKey}
                            selectableTiles={humanTurn ? legalTileSet : []}
                            onTileClick={(tile, key) => this.handleTileClick(tile, key)}
                            onTileDoubleClick={(tile) => this.handleTileDoubleClick(tile)}
                            interactivePlayerId={0}
                            drawTileByPlayer={drawTileByPlayer}
                        />
                    </div>

                    {/* Right: Player Status Sidebar */}
                    <div className="sidebar-section">
                        <div className="player-status-box">
                            <h3 style={{fontSize: '14px', color: '#909399'}}>Player Status</h3>
                            {[0, 1, 2, 3].map(pid => (
                                <div key={pid} className={"player-item " + (pid === activePlayerIdx ? "active" : "")}>
                                    <Avatar>{pid}</Avatar>
                                    <div style={{textAlign: 'left'}}>
                                        <div style={{fontWeight: 'bold'}}>Player {pid}</div>
                                        <div style={{fontSize: '12px', color: '#666'}}>
                                            {pid === 0 ? "Human Player" : "AI Agent"}
                                        </div>
                                        <div className="player-timer">
                                            {pid === activePlayerIdx ? 'Thinking' : 'Last'}: {formatDuration(thinkingByPlayer[pid] || 0)}
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                        {this.humanPlayEnabled && (
                            <div className="human-control-box">
                                <h3>Human Controls</h3>
                                <div className={`action-status ${humanTurn ? 'active' : ''}`}>
                                    {humanTurn ? 'Your turn' : 'Waiting for turn'}
                                </div>
                                <div className="action-row">
                                    <span>Selected</span>
                                    <span>{selectedTile || 'None'}</span>
                                </div>
                                <div className="action-buttons">
                                    {specialMoves.length > 0 ? (
                                        specialMoves.map((move) => (
                                        <Button
                                            key={move}
                                            variant="outlined"
                                            size="small"
                                            disabled={actionDisabled}
                                            onClick={() => {
                                                this.clearSelection();
                                                this.submitHumanAction(move);
                                            }}
                                        >
                                            {formatSpecialMove(move)}
                                        </Button>
                                        ))
                                    ) : (
                                        <span className="action-muted">No special actions</span>
                                    )}
                                </div>
                                <div className="tile-action-bar">
                                    <Button
                                        variant="contained"
                                        color="primary"
                                        size="small"
                                        disabled={!canPlaySelected}
                                        onClick={() => {
                                            this.clearSelection();
                                            this.submitHumanAction(selectedTile);
                                        }}
                                    >
                                        Play Selected
                                    </Button>
                                    <Button
                                        variant="outlined"
                                        size="small"
                                        disabled={!selectedTile || pendingAction}
                                        onClick={() => this.clearSelection()}
                                    >
                                        Clear
                                    </Button>
                                </div>
                                <div className="action-hint">
                                    Click a legal tile in your hand to select it.
                                </div>
                            </div>
                        )}
                        <div style={{padding: 15, flex: 1, color: '#999', fontSize: '13px'}}>
                            <div>Winner: {replayData.winner || 'Unknown'}</div>
                            <div style={{marginTop: 10}}>Status: {playStatus}</div>
                            <div style={{marginTop: 10}}>
                                Remaining: {remainingTiles !== null && remainingTiles !== undefined ? remainingTiles : 'N/A'}
                            </div>
                        </div>
                    </div>
                </div>

                <div className="replay-bottom-section">
                    <div className="record-panels">
                        {/* Table Talk */}
                        <div className="record-box">
                            <div className="box-title">TABLE TALK</div>
                            <div className="box-content" ref={this.chatLogRef}>
                                {chatHistory.length > 0 ? (
                                    chatHistory.map((c, i) => (
                                        <div key={i} style={{marginBottom: 5}}>
                                            <span style={{fontWeight:'bold', color: '#409EFF'}}>P{c.pid}:</span> {c.msg}
                                        </div>
                                    ))
                                ) : (
                                    <div style={{color:'#ccc'}}>No chat records</div>
                                )}
                            </div>
                            {this.humanPlayEnabled && (
                                <div className="tabletalk-input">
                                    <input
                                        type="text"
                                        placeholder="Say something..."
                                        value={chatDraft}
                                        onChange={(e) => this.handleChatDraftChange(e.target.value)}
                                        onKeyDown={(e) => {
                                            if (e.key === 'Enter') {
                                                this.submitHumanChat();
                                            }
                                        }}
                                        disabled={pendingChat}
                                    />
                                    <Button
                                        variant="contained"
                                        color="primary"
                                        size="small"
                                        disabled={pendingChat || !chatDraft.trim()}
                                        onClick={() => this.submitHumanChat()}
                                    >
                                        Send
                                    </Button>
                                </div>
                            )}
                        </div>
                        {/* Move History */}
                        <div className="record-box">
                            <div className="box-title">MOVE HISTORY</div>
                            <div className="box-content" ref={this.historyPanelRef}>
                                {moveHistory.length > 0 ? (
                                    moveHistory.map((h, i) => (
                                        <div key={i} style={{borderBottom: '1px dashed #eee', padding: '2px 0'}}>
                                            <span style={{display:'inline-block', width:30, color:'#999'}}>{h.step}.</span>
                                            <span style={{fontWeight:'bold'}}>P{h.pid}</span>: {h.action}
                                        </div>
                                    ))
                                ) : (
                                    <div style={{color:'#ccc'}}>No move records</div>
                                )}
                            </div>
                        </div>
                    </div>

                    <div className="control-bar">
                        <div className="playback-controls">
                            <Button 
                                variant="contained"
                                color="primary" 
                                startIcon={<SkipPreviousIcon />}
                                disabled={step <= 0}
                                onClick={() => this.setState({step: Math.max(0, step-1)})}
                            >
                                Prev
                            </Button>
                            
                            {!autoPlay ? (
                                <Button 
                                    variant="contained" 
                                    color="primary"
                                    startIcon={<PlayArrowRoundedIcon />}
                                    onClick={() => this.setState({autoPlay: true})}
                                >
                                    Resume
                                </Button>
                            ) : (
                                <Button 
                                    variant="contained" 
                                    color="secondary"
                                    startIcon={<PauseCircleOutlineRoundedIcon />}
                                    onClick={() => this.setState({autoPlay: false})}
                                >
                                    Pause
                                </Button>
                            )}

                            <Button 
                                variant="contained"
                                color="primary"
                                startIcon={<SkipNextIcon />}
                                disabled={step >= moves.length}
                                onClick={() => this.setState({step: Math.min(moves.length, step+1)})}
                            >
                                Next
                            </Button>
                        </div>
                        <div className="progress-slider">
                            <Slider
                                value={step} 
                                onChange={(e, val) => this.setState({step: val})}
                                min={0}
                                max={moves.length}
                            />
                        </div>
                        <div className="speed-control">
                            <span style={{fontSize: 12, marginRight: 15, color: '#666', whiteSpace:'nowrap'}}>Speed</span>
                            <Slider
                                value={gameSpeed}
                                onChange={(e, val) => this.changeGameSpeed(val)}
                                min={-2}
                                max={3}
                                step={1}
                                track={false}
                                marks={gameSpeedMarks}
                                valueLabelDisplay="off"
                                getAriaValueText={sliderValueText}
                            />
                        </div>
                    </div>
                </div>
            </div>
        );
    }
}

export default withRouter(MahjongReplayView);

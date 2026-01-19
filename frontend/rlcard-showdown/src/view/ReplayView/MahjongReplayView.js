import React from 'react';
import { withRouter } from 'react-router-dom';
import qs from 'query-string';
import MahjongGameBoard from '../../components/GameBoard/MahjongGameBoard';
import { apiUrl } from '../../utils/config';
import axios from 'axios';
import { Loading } from 'element-react';
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
        };
        this.intervalId = null;
        this.autoPlayId = null;
        this.chatLogRef = React.createRef();
        this.historyPanelRef = React.createRef();
    }

    componentDidMount() {
        this.fetchReplay();
        this.intervalId = setInterval(this.fetchReplay, 5000);
        
        // Parse showHands from URL if needed, or default
        const { location } = this.props;
        const query = qs.parse(location.search);
        const mode = query.mode || 'ai';
        if (mode === 'human') {
            this.setState({ showHands: false });
        } else {
            this.setState({ showHands: true });
        }
    }

    componentWillUnmount() {
        if (this.intervalId) clearInterval(this.intervalId);
        if (this.autoPlayId) clearInterval(this.autoPlayId);
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
                    if (newMoves.length < prevState.step) {
                        nextStep = newMoves.length;
                    }
                    return { replayData: newData, step: nextStep, error: null };
                });
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
        const { replayData, step, autoPlay, error, showHands, gameSpeed } = this.state;
        const { location } = this.props;
        const query = qs.parse(location.search);
        
        if (error && !replayData) return <div style={{padding: 20}}>Error: {error}</div>;
        if (!replayData) return <div style={{padding: 20}}>Loading Mahjong Replay...</div>;

        const moves = replayData.moves || [];
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
        const extractActionCard = (text) => {
            const match = String(text || "").match(/([BCD][1-9]|East|South|West|North|Green|Red|White)/i);
            if (!match) return null;
            return normalizeTileCode(match[0]);
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

        currentMoves.forEach((m, idx) => {
            const pidStr = m.player_id || "player_0";
            const pid = parseInt(pidStr.replace('player_', '')) || 0;
            const txt = m.action_text || "";
            const actionCard = resolveActionCard(m);
            lastAction = `${pidStr}: ${txt}`;
            
            moveHistory.push({ pid, action: txt, step: m.step || (idx + 1) });

            // Naive discard parsing: "B3", "East", etc.
            if (actionCard) {
                 discards.push(actionCard);
            }
            if (m.chat) {
                // Show bubble for recent chat
                if ((step - idx) < 5) chatBubbles[pid] = m.chat;
                chatHistory.push({ pid, msg: m.chat, step: idx });
            }
        });

        // Hands & Piles
        // Ideally: replayData.snapshots[step]
        // Fallback: use replayData.current_hands and assume we are at end? 
        // Or if we don't have per-step snapshots, the board will look static (except discards).
        // Let's try to find `hand_history`.
        const handsSource = (replayData.hand_history && replayData.hand_history[step]) 
                          ? replayData.hand_history[step] 
                          : (replayData.current_hands || {});

        const parseHand = (pid) => {
             const data = handsSource[pid];
             if (!data) return Array(13).fill('Back');
             const handList = data.hand || [];
             if (!showHands && pid !== 0) return Array(handList.length).fill('Back');
             return handList;
        };
        const parsePile = (pid) => handsSource[pid]?.pile || [];

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
        const playStatus = autoPlay
            ? (hasMoreMoves ? 'Playing' : 'Waiting')
            : (hasMoreMoves ? 'Paused' : 'Stopped');

        return (
            <div className="mahjong-replay-container">
                <div className="replay-header">
                    <h2>Mahjong Game Replay</h2>
                    <div className="header-actions">
                        <Chip label={`Turn: ${step}`} color="primary" variant="outlined" />
                        <Chip label={`Mode: ${query.mode || 'AI'}`} color="secondary" variant="outlined" />
                        <Chip label={`Play: ${playStatus}`} color="default" variant="outlined" />
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
                        <MahjongGameBoard 
                            hands={hands}
                            piles={piles}
                            discards={discards}
                            chatBubbles={chatBubbles}
                            currentPlayer={0}
                            showAllHands={showHands}
                            lastAction={lastAction}
                            // New Props for Robustness
                            isTsumogiri={isTsumogiri}
                            lastPlayedTile={lastPlayedTile}
                        />
                    </div>

                    {/* Right: Player Status Sidebar */}
                    <div className="sidebar-section">
                        <div className="player-status-box">
                            <h3 style={{fontSize: '14px', color: '#909399'}}>Player Status</h3>
                            {[0, 1, 2, 3].map(pid => (
                                <div key={pid} className={"player-item " + (pid === 0 ? "active" : "")}>
                                    <Avatar>{pid}</Avatar>
                                    <div style={{textAlign: 'left'}}>
                                        <div style={{fontWeight: 'bold'}}>Player {pid}</div>
                                        <div style={{fontSize: '12px', color: '#666'}}>
                                            {pid === 0 ? "Human Player" : "AI Agent"}
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                        <div style={{padding: 15, flex: 1, color: '#999', fontSize: '13px'}}>
                            <div>Winner: {replayData.winner || 'Unknown'}</div>
                            <div style={{marginTop: 10}}>Status: {playStatus}</div>
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

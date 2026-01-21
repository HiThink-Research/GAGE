import React, { useRef } from 'react';
import Avatar from '@material-ui/core/Avatar';
import Chip from '@material-ui/core/Chip';
import MahjongTile from './MahjongTile';
import '../../assets/mahjong.scss';
import PlaceHolderPlayer from '../../assets/images/Portrait/Player.png';

const MahjongGameBoard = ({
    hands,
    piles,
    discards,
    chatBubbles,
    currentPlayer,
    showAllHands,
    lastAction,
    isTsumogiri: propsIsTsumogiri,
    lastPlayedTile: propsLastPlayedTile,
    selectedTileKey,
    selectableTiles,
    onTileClick,
    onTileDoubleClick,
    interactivePlayerId = 0,
    drawTileByPlayer,
}) => {
    
    const lastDrawRef = useRef(null);
    const normalizeTileCode = (rawCode) => {
        if (!rawCode) return null;
        const trimmed = String(rawCode).trim();
        if (!trimmed) return null;
        const upper = trimmed.toUpperCase();
        if (upper.length === 2 && ['B', 'C', 'D'].includes(upper[0]) && /\d/.test(upper[1])) {
            return upper;
        }
        const titled = trimmed.charAt(0).toUpperCase() + trimmed.slice(1).toLowerCase();
        if (['East', 'South', 'West', 'North', 'Green', 'Red', 'White'].includes(titled)) {
            return titled;
        }
        return trimmed;
    };
    const normalizedPropTile = normalizeTileCode(propsLastPlayedTile);
    
    // Position classes: bottom (0), right (1), top (2), left (3)
    const getPositionClass = (pid) => {
        if (pid === 0) return 'bottom';
        if (pid === 1) return 'right';
        if (pid === 2) return 'top';
        if (pid === 3) return 'left';
        return 'bottom';
    }

    // --- PARSING LOGIC (Moved to top for scope access) ---
    let lastPlayedCard = null;
    let lastPlayerPos = null;
    let drawingCard = null;
    let drawingPlayerPos = null;
    let isTsumogiri = false;

    if (lastAction) {
        const parts = lastAction.split(':');
        if (parts.length > 1) {
           let pIdStr = parts[0].trim(); 
           // Re-join the rest to get full action text (handles "player_0: DREW:B3")
           let actionText = parts.slice(1).join(':').trim();
           
           let pid = 0;
           if (pIdStr.indexOf('player_') !== -1) {
               pid = parseInt(pIdStr.replace('player_', '')) || 0;
           } else if (pIdStr.indexOf('玩家') !== -1) {
               pid = parseInt(pIdStr.replace('玩家', '').trim()) || 0;
           } else {
               pid = parseInt(pIdStr) || 0;
           }
           const posClass = getPositionClass(pid);

           
           // 1. CHECK FOR DRAW ACTION
           // Use robust Regex to capture DREW/DRAW and the card code
           const drawMatch = actionText.match(/^(?:DREW|DRAW):?\s*([a-zA-Z0-9]+)/i);
           
           if (drawMatch) {
               const drawnCode = normalizeTileCode(drawMatch[1]);
               drawingPlayerPos = posClass;
               if (showAllHands || pid === 0) {
                   drawingCard = drawnCode;
               } else {
                   drawingCard = 'Back';
               }
               // Track for Tsumogiri detection (Fallback)
               lastDrawRef.current = drawnCode;
               console.log(`[Mahjong] Logged Draw: ${drawnCode}`);
           } 
           // 2. CHECK FOR DISCARD ACTION
           else {
                const match = actionText.match(/([BCD][1-9]|East|South|West|North|Green|Red|White)/i);
                if (match) {
                    const cardCode = match[0];
                    lastPlayerPos = posClass;
                    
                    const standardizedCode = normalizeTileCode(cardCode);
                    lastPlayedCard = standardizedCode;

                    // Detect Tsumogiri
                    // Priority: Props > Internal Heuristic
                    if (propsIsTsumogiri !== undefined) {
                         isTsumogiri = propsIsTsumogiri;
                    } else if (lastDrawRef.current && 
                        lastDrawRef.current.toUpperCase() === lastPlayedCard.toUpperCase()) {
                        isTsumogiri = true;
                    }
                    console.log(`[Mahjong] Play: ${lastPlayedCard} | IsTsumo: ${isTsumogiri} (Prop: ${propsIsTsumogiri})`);
                }
           }
        }
    }
    if (normalizedPropTile) {
        lastPlayedCard = normalizedPropTile;
        if (!lastPlayerPos) {
            lastPlayerPos = getPositionClass(currentPlayer);
        }
    }
    if (propsIsTsumogiri !== undefined && lastPlayedCard) {
        isTsumogiri = propsIsTsumogiri;
    }

    // --- RENDER HELPERS ---
    const selectableSet = selectableTiles instanceof Set ? selectableTiles : new Set(selectableTiles || []);

    const renderHand = (playerId) => {
        const handData = hands[playerId] || [];
        const pileData = piles ? (piles[playerId] || []) : [];
        const position = getPositionClass(playerId);
        const bubbleText = chatBubbles ? chatBubbles[playerId] : "";
        const isInteractive = Boolean(onTileClick) && playerId === interactivePlayerId;

        // Debug Log
        if (playerId === 0 && (lastPlayedCard || drawingCard)) {
             console.log(`[MahjongRender] Player 0 | Action: ${lastAction} | Draw: ${drawingCard} | Play: ${lastPlayedCard}`);
             console.log(`[MahjongRender] Hand Size: ${handData.length} | Hand: ${handData.join(',')}`);
        }

        // SEPARATION LOGIC: Extract the newly drawn card to render it in a separate slot
        let mainHand = [...handData]; // Copy to avoid mutation
        let newDrawCard = null;
        let showGap = false;

        const drawOverride = drawTileByPlayer
            ? (drawTileByPlayer[playerId] ?? drawTileByPlayer[String(playerId)])
            : null;
        if (drawOverride) {
            const target = normalizeTileCode(drawOverride);
            let removeIdx = -1;
            for (let i = mainHand.length - 1; i >= 0; i--) {
                const c = mainHand[i];
                if (normalizeTileCode(c) === target) {
                    removeIdx = i;
                    break;
                }
            }
            if (removeIdx !== -1) {
                newDrawCard = mainHand[removeIdx];
                mainHand.splice(removeIdx, 1);
            }
        } else if (drawingCard && drawingPlayerPos === position) {
            let removeIdx = -1;
            // Priority 1: Exact Match (Search backwards)
            for (let i = mainHand.length - 1; i >= 0; i--) {
                const c = mainHand[i];
                if (c && c.toUpperCase() === drawingCard.toUpperCase()) {
                    removeIdx = i;
                    break;
                }
            }
            // Priority 2: Back
            if (removeIdx === -1) {
                for (let i = mainHand.length - 1; i >= 0; i--) {
                    if (mainHand[i] && mainHand[i].toUpperCase() === 'BACK') {
                        removeIdx = i; break;
                    }
                }
            }
            if (removeIdx !== -1) {
                newDrawCard = mainHand[removeIdx];
                mainHand.splice(removeIdx, 1);
            }
        } 
        // Fallback: If no explicit draw event but hand size implies a draw (14 cards)
        else if (mainHand.length % 3 === 2) {
             // Isolate the last card
             newDrawCard = mainHand.pop();
        }
        
        else if (lastPlayedCard && lastPlayerPos === position) {
            // DISCARD STATE: Show a gap only if it's a Tsumogiri (Draw-and-Play)
            // If Tedashi (Play from hand), the hand naturally closes, so no gap needed.
            if (isTsumogiri) {
                showGap = true;
            }
        }

        return (
            <div key={playerId} className={`player-area player-${position}`}>
                {/* Info with Avatar and Chat */}
                <div className={`player-info pos-${position}`}>
                    {bubbleText && <div className={`player-chat-bubble bubble-${position}`}>{bubbleText}</div>}
                    <div>
                        <img src={PlaceHolderPlayer} alt={'Player'} height="50px" width="50px" />
                        <Chip 
                            style={{marginTop: 5, backgroundColor: 'rgba(255,255,255,0.8)'}}
                            avatar={<Avatar>{playerId}</Avatar>} 
                            label={`玩家 ${playerId}`} 
                            size="small"
                        />
                    </div>
                </div>
                
                {/* Hand Container */}
                <div className="hand-container">
                    {/* Melds (Piles) */}
                    <div className="melds-area">
                        {pileData.map((meld, mIdx) => (
                            <div key={mIdx} className="meld-group">
                                {meld.map((code, cIdx) => (
                                    <MahjongTile key={cIdx} tile={code} isHidden={false} />
                                ))}
                            </div>
                        ))}
                    </div>

                    {/* Active Hand (Remaining Cards) */}
                    <div className="active-hand">
                        {mainHand.map((code, idx) => {
                            const tileKey = `hand-${playerId}-${idx}`;
                            const tileCode = String(code || "");
                            const isHidden = tileCode === 'Back' || tileCode === 'back';
                            const normalized = tileCode.toLowerCase();
                            const isSelectable = isInteractive && !isHidden && selectableSet.has(normalized);
                            const isSelected = selectedTileKey === tileKey;
                            const handleClick = isSelectable
                                ? () => onTileClick(tileCode, tileKey)
                                : undefined;
                            const handleDoubleClick = isSelectable && onTileDoubleClick
                                ? () => onTileDoubleClick(tileCode, tileKey)
                                : undefined;
                            return (
                                <MahjongTile 
                                    key={idx} 
                                    tile={code} 
                                    isHidden={isHidden} 
                                    isSelected={isSelected}
                                    isSelectable={isSelectable}
                                    onClick={handleClick}
                                    onDoubleClick={handleDoubleClick}
                                />
                            );
                        })}
                    </div>

                    {/* New Draw Slot (Real or Ghost Gap) */}
                    {(newDrawCard || showGap) && (
                        <div className={`new-card-slot ${newDrawCard ? 'new-card-highlight' : ''}`} style={{marginLeft: '12px', position: 'relative'}}>
                            {newDrawCard ? (() => {
                                const tileKey = `draw-${playerId}`;
                                const tileCode = String(newDrawCard || "");
                                const isHidden = tileCode === 'Back' || tileCode === 'back';
                                const normalized = tileCode.toLowerCase();
                                const isSelectable = isInteractive && !isHidden && selectableSet.has(normalized);
                                const isSelected = selectedTileKey === tileKey;
                                const handleClick = isSelectable
                                    ? () => onTileClick(tileCode, tileKey)
                                    : undefined;
                                const handleDoubleClick = isSelectable && onTileDoubleClick
                                    ? () => onTileDoubleClick(tileCode, tileKey)
                                    : undefined;
                                return (
                                    <MahjongTile 
                                        tile={newDrawCard} 
                                        isHidden={isHidden} 
                                        isSelected={isSelected}
                                        isSelectable={isSelectable}
                                        onClick={handleClick}
                                        onDoubleClick={handleDoubleClick}
                                    />
                                );
                            })() : (
                                // Ghost Tile for visual gap
                                <div style={{width: 30, height: 42, opacity: 0}} />
                            )}
                        </div>
                    )}
                </div>
            </div>
        );
    };

    return (
        <div className="mahjong-board">
            
            {/* Last Played Card Animation Layer */}
            {lastPlayedCard && (
                <div className={`last-played-card last-card-pos-${lastPlayerPos}${isTsumogiri ? '-tsumo' : ''}`}>
                    <MahjongTile tile={lastPlayedCard} style={{width: '100%', height: '100%'}} />
                </div>
            )}
            
            {/* Center Area: Global Discards */}
            <div className="center-area">
                <div className="discard-pool">
                    {discards.map((code, idx) => {
                        // 弃牌堆最后一张牌延迟显示并高亮
                        const isLastDiscard = idx === discards.length - 1;
                        const shouldHighlight = isLastDiscard && lastPlayedCard !== null;
                        
                        return (
                            <div key={idx} className={shouldHighlight ? "last-discard-highlight" : ""} style={{position: 'relative'}}>
                                 <MahjongTile tile={code} isHidden={false} style={{width: '100%', height: '100%'}} />
                            </div>
                        );
                    })}
                </div>
                {/* Center Info Text (optional, can be removed if redundant) */}
                {/* <div className="center-info">{lastAction || "游戏开始"}</div> */}
            </div>

            {/* Render all 4 players */}
            {[0, 1, 2, 3].map(pid => renderHand(pid))}
        </div>
    );
};

export default MahjongGameBoard;

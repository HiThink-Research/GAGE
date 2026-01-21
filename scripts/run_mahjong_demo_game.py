
import random
import sys
import os
import json
import time

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), "src"))

from gage_eval.role.arena.games.mahjong.env import MahjongArena

class MahjongAgent:
    """
    A simple agent that plays Mahjong.
    In a real scenario, this would load a model or use a complex heuristic.
    Currently implements a random policy for demonstration of integration.
    """
    def __init__(self, player_id):
        self.player_id = player_id
    
    def step(self, state):
        """
        Given a state dictionary, return an action.
        state['legal_moves'] is a list of legal action strings.
        """
        legal_moves = state['legal_moves']
        if not legal_moves:
            return None
            
        # --- AI LOGIC GOES HERE ---
        # Example: prioritize 'Hu', then 'Pong', then discard isolated cards.
        # For this demo, we use random choice to simulate an untrained agent.
        action = random.choice(legal_moves)
        return action

def run_loop():
    print("Starting 4-AI Mahjong Agent Demo...")
    output_path = os.path.join(os.getcwd(), "runs", "mahjong_replay.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("Initializing Environment...")
    arena = MahjongArena()
    
    # Initialize 4 Agents
    print("Initializing 4 Agents...")
    agents = [MahjongAgent(i) for i in range(4)]
    
    # Track history
    hand_history = []
    all_moves = [] 
    
    # Save initial state
    hand_history.append(arena.get_all_hands())
    save_state(arena, output_path, winner=None, hand_history=hand_history, moves=all_moves)
    
    step = 0
    print("Game Loop Started.")
    while not arena.is_terminal():
        active_player_id = arena.get_active_player()
        
        # Helper to get integer ID
        if isinstance(active_player_id, str):
            if active_player_id.startswith('player_'):
                p_idx = int(active_player_id.split('_')[1])
            else:
                try:
                    p_idx = int(active_player_id)
                except:
                    p_idx = 0
        else:
            p_idx = int(active_player_id)
            
        active_agent = agents[p_idx]
        
        # 1. Observation
        obs = arena.observe(active_player_id)
        
        # 2. Agent Decision
        action_text = active_agent.step(obs)
        if not action_text:
            break
            
        # Artificial delay for visual demo
        time.sleep(0.1)
        
        # Add some flavor text occasionally
        chat_msg = ""
        if random.random() < 0.05:
            chat_msg = random.choice(["Thinking...", "Nice move", "Hurry up"])
        
        print(f"[Step {step:03d}] Agent {active_player_id} performs: {action_text}")
        
        # Optional: Simulate 'Draw' animation logic for visual consistency in Replay
        is_discard_action = action_text in ["East", "South", "West", "North", "Green", "Red", "White"] or \
                           (len(action_text) == 2 and action_text[0] in "BCD" and action_text[1].isdigit())
        
        if is_discard_action:
            all_moves.append({
                "player_id": active_player_id,
                "action_id": -1,
                "action_text": f"DREW:{action_text}", 
                "chat": ""
            })
            hand_history.append(arena.get_all_hands())
            save_state(arena, output_path, winner=None, hand_history=hand_history, moves=all_moves)
            
        # 3. Environment Step
        arena.apply({"player_id": active_player_id, "action": action_text, "chat": chat_msg})
        
        # Log move
        if arena._move_log:
            all_moves.append(arena._move_log[-1])
            
        step += 1
        hand_history.append(arena.get_all_hands())
        save_state(arena, output_path, winner=None, hand_history=hand_history, moves=all_moves)
        
        if step > 300: # Safety break
            break
            
    result = arena.build_result(reason="finished")
    save_state(arena, output_path, winner=result['winner'], hand_history=hand_history, moves=all_moves)
    print(f"Game Finished. Winner: {result['winner']}")

def save_state(arena, path, winner, hand_history=None, moves=None):
    payload = {
        "winner": winner,
        "moves": moves if moves is not None else [],
        "current_hands": arena.get_all_hands(),
        "hand_history": hand_history
    }
    with open(path, 'w') as f:
        json.dump(payload, f)

if __name__ == "__main__":
    run_loop()

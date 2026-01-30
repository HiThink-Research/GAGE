
import random
import sys
import os

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), "src"))

from gage_eval.role.arena.games.mahjong.env import MahjongArena

def run_test():
    print("Initializing Mahjong Arena...")
    arena = MahjongArena()
    print("Initialized.")
    
    step = 0
    while not arena.is_terminal():
        active_player = arena.get_active_player()
        obs = arena.observe(active_player)
        
        legal_moves = obs["legal_moves"]
        if not legal_moves:
            print(f"No legal moves for {active_player} but not terminal?")
            break
            
        # Random policy
        action_text = random.choice(legal_moves)
        
        print(f"Step {step}: Player {active_player} plays {action_text} (Legal: {legal_moves})")
        
        result = arena.apply({"player_id": active_player, "action": action_text})
        step += 1
        
        if step > 200:
            print("Force stop > 200 steps")
            break
            
    result = arena.build_result(result="draw", reason="finished")
    print(f"Game Over. Winner: {result.winner}")
    print(f"Result: {result.result}")

if __name__ == "__main__":
    run_test()

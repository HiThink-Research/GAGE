import rlcard
from rlcard.agents import RandomAgent
print("RLCard version:", rlcard.__version__)
try:
    from rlcard.agents import RuleBasedAgent
    print("RuleBasedAgent available")
except ImportError:
    print("RuleBasedAgent NOT available directly")

# Check if mahjong environment works
try:
    env = rlcard.make('mahjong')
    print("Mahjong Env created successfully")
    print("Action space:", env.num_actions)
except Exception as e:
    print("Error creating mahjong env:", e)


import os
import requests
import time

# Target directory
TARGET_DIR = "frontend/rlcard-showdown/src/assets/mahjong"
if not os.path.exists(TARGET_DIR):
    os.makedirs(TARGET_DIR)

# Base URL for open source SVG tiles (using FluffyStuff's set as standard)
BASE_URL = "https://raw.githubusercontent.com/FluffyStuff/riichi-mahjong-tiles/master/Regular"

# Mapping
# B = Sou, C = Man, D = Pin
# Winds: East=Ton, South=Nan, West=Sha, North=Pei
# Dragons: White=Haku, Green=Hatsu, Red=Chun
tiles = {}
for i in range(1, 10):
    tiles[f"B{i}"] = f"s{i}.svg"
    tiles[f"C{i}"] = f"m{i}.svg"
    tiles[f"D{i}"] = f"p{i}.svg"

tiles["East"] = "z1.svg"
tiles["South"] = "z2.svg"
tiles["West"] = "z3.svg"
tiles["North"] = "z4.svg"
tiles["White"] = "z5.svg"
tiles["Green"] = "z6.svg"
tiles["Red"] = "z7.svg"
tiles["Back"] = "Back.svg" # Usually Back.svg exists or similar

def download_file(name, filename):
    url = f"{BASE_URL}/{filename}"
    try:
        r = requests.get(url)
        if r.status_code == 200:
            with open(f"{TARGET_DIR}/{name}.svg", 'wb') as f:
                f.write(r.content)
            print(f"Downloaded {name}")
        else:
            print(f"Failed {name}: {r.status_code}")
    except Exception as e:
        print(f"Error {name}: {e}")

if __name__ == "__main__":
    print("Downloading Mahjong Tiles...")
    for k, v in tiles.items():
        download_file(k, v)
        time.sleep(0.1)
    
    # Back tile is special usually
    # Try downloading a generic back
    # FluffyStuff has 'Black.svg' or similar?
    # Let's check visually or failover.

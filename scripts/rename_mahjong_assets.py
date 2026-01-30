
import os
import shutil

TARGET_DIR = "frontend/rlcard-showdown/src/assets/mahjong"

# Map stored files (Man1.svg) to our codes (C1.svg)
mapping = {}
# Man (Characters) -> C
for i in range(1, 10): mapping[f"Man{i}.svg"] = f"C{i}.svg"
# Pin (Dots) -> D
for i in range(1, 10): mapping[f"Pin{i}.svg"] = f"D{i}.svg"
# Sou (Bamboo) -> B
for i in range(1, 10): mapping[f"Sou{i}.svg"] = f"B{i}.svg"

# Winds
mapping["Ton.svg"] = "East.svg"
mapping["Nan.svg"] = "South.svg"
mapping["Shaa.svg"] = "West.svg"
mapping["Pei.svg"] = "North.svg"

# Dragons
mapping["Haku.svg"] = "White.svg"
mapping["Hatsu.svg"] = "Green.svg"
mapping["Chun.svg"] = "Red.svg"

# Back
if os.path.exists(f"{TARGET_DIR}/Back.svg"):
    mapping["Back.svg"] = "Back.svg"

for src, dst in mapping.items():
    src_path = os.path.join(TARGET_DIR, src)
    dst_path = os.path.join(TARGET_DIR, dst)
    if os.path.exists(src_path):
        os.rename(src_path, dst_path)
        print(f"Renamed {src} -> {dst}")
    else:
        print(f"Missing {src}")

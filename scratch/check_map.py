import json
import os
from dotenv import load_dotenv

load_dotenv()
channels = os.getenv('TARGET_CHANNELS', '').split(',')

map_path = 'data/channel_map.json'
if os.path.exists(map_path):
    with open(map_path, 'r') as f:
        cmap = json.load(f)
else:
    cmap = {}

missing = []
for c in channels:
    username = c.replace('@', '')
    if username not in cmap:
        # Case insensitive check
        found = False
        for k in cmap:
            if k.lower() == username.lower():
                cmap[username] = cmap[k]
                found = True
                break
        if not found:
            missing.append(c)

print(f"Missing: {missing}")

# Save updated map (case sensitive normalization)
with open(map_path, 'w') as f:
    json.dump(cmap, f, indent=2)

"""Scene Parser test v2."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from search.scene_parser import SceneParser

test_img = "data/images/1005812857_118.jpg"
print(f"Test gorseli: {test_img}")

parser = SceneParser(confidence_threshold=0.1)
detections = parser.detect_objects(test_img)

print(f"\nTespit edilen objeler ({len(detections)}):")
for i, det in enumerate(detections):
    label = det["label"]
    conf = det["confidence"]
    bbox = det["bbox"]
    print(f"  {i+1}. {label:20s} | guven: {conf:.3f} | bbox: {bbox}")

parser.unload()
print("\nTest tamamlandi!")

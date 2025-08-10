from ultralytics import YOLO
import cv2
import yaml
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# load model
model = YOLO("yolov8s.pt")  

# quick test prediction
image_path = "test_image.jpg"
results = model.predict(source=image_path, imgsz=640, conf=0.7)
annotated = results[0].plot(line_width=2)
plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

# dataset paths
dataset_path = Path("data")
yaml_file_path = dataset_path / "dataset.yaml"

# show yaml
with open(yaml_file_path, "r") as f:
    print(yaml.dump(yaml.safe_load(f), default_flow_style=False))

# dataset stats
for split in ["train", "val"]:
    img_dir = dataset_path / split / "images"
    sizes = {Image.open(p).size for p in img_dir.glob("*.jpg")}
    print(f"{split}: {len(list(img_dir.glob('*.jpg')))} images, sizes: {sizes}")

# train
model.train(
    data=str(yaml_file_path),
    epochs=50,
    imgsz=640,
    batch=16,
    device="",
    patience=20,
    optimizer="auto",
    lr0=0.0005,
    lrf=0.01,
    warmup_epochs=3,
    seed=0
)

# predict video
video_path = "video.mp4"
pred_results = model.predict(source=video_path, save=True, conf=0.5)
print(f"Predictions saved to: {pred_results[0].save_dir}")

# view results
results_dir = Path("runs/train/exp")
plt.imshow(cv2.cvtColor(cv2.imread(str(results_dir / "confusion_matrix_normalized.png")), cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

plt.imshow(cv2.cvtColor(cv2.imread(str(results_dir / "results.png")), cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

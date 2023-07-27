#for serving 
import bentoml
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = bentoml.pytorch.save_model(
    "yolov8-seg-finetune", 
    YOLO('./yolo/coco-person-1280-es50-m.pt').model,
    signatures={ 
                "predict": {
                    "batchable": True,
                    "batch_dim": 0,
                }
    }
    )
print("model is saved at", model)

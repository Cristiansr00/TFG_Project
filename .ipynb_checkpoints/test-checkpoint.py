from ultralytics import YOLO

print("hola que tal estamos")

# Load the model
model = YOLO("")

# Validate the model
metrics = model.val()
print(metrics.box.map)  # mAP50-95
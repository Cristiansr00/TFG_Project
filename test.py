from ultralytics import YOLO
import cv2

model = YOLO(r"models\trained\MODEL_SynRGB_entropy_mean_nc3_5kFold\kfold_training(basic split_1 100epochs)\train\weights\best.pt")
model = YOLO(r"models\trained\MODEL_2026-01-18_5-Fold_Cross-val---\kfold_training(basic split_1 100epochs)\train\weights\best.pt")
image_path = "2Gy-011 copy.JPG"
img = cv2.imread(image_path)

results = model.predict(
    source=image_path,
    conf=0.7,
    iou=0.5
)

result = results[0]

for box in result.boxes:
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
    cls_id = int(box.cls[0].cpu().numpy())
    conf = float(box.conf[0].cpu().numpy())

    class_name = model.names[cls_id]

    if cls_id == 1:
        color = (0, 255, 0)   # verde para dicéntrico
    else:
        color = (0, 0, 255)   # rojo para normal

    label = f"{class_name}: {conf:.2f}"

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        img,
        label,
        (x1, max(y1 - 5, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1
    )

cv2.imwrite("resultado_con_cajas.jpg", img)
cv2.imshow("Resultado", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
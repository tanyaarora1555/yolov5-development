import cv2
import numpy as np
import onnxruntime as ort
import yaml

# Load class names from YAML
yaml_path = "./data.yaml"
with open(yaml_path, "r") as f:
    data_yaml = yaml.safe_load(f)
class_names = data_yaml.get("names", [])
#print("Loaded class names:", class_names)


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]  # current shape (h, w)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


onnx_path = "./segmentation_weights/best1.onnx"
session = ort.InferenceSession(onnx_path)
input_name = session.get_inputs()[0].name
output_names = [o.name for o in session.get_outputs()]


video_path = "papaya.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video source at {video_path}")


colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
conf_thres = 0.25
iou_thres = 0.65


while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig = frame.copy()
    h0, w0 = orig.shape[:2]

   
    img, r, dwdh = letterbox(orig, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)

    
    outputs = session.run(output_names, {input_name: img})
    pred, proto = outputs[0][0], outputs[1][0]

    boxes, scores, class_ids, masks = [], [], [], []

    for det in pred:
        obj_conf = det[4]
        cls_scores = det[5:-32]
        cls_id = int(np.argmax(cls_scores))
        cls_conf = cls_scores[cls_id] * obj_conf

        if cls_conf < conf_thres:
            continue

        # Bounding box (undo letterbox transform)
        x, y, w, h = det[:4]
        x0 = int((x - w / 2 - dwdh[0]) / r)
        y0 = int((y - h / 2 - dwdh[1]) / r)
        x1 = int((x + w / 2 - dwdh[0]) / r)
        y1 = int((y + h / 2 - dwdh[1]) / r)

        
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w0, x1), min(h0, y1)

        boxes.append([x0, y0, x1, y1])
        scores.append(float(cls_conf))
        class_ids.append(int(cls_id))

        
        mask_data = det[-32:] @ proto.reshape(32, -1)
        mask = mask_data.reshape(proto.shape[1], proto.shape[2]) 

        
        mask = cv2.resize(mask, (640, 640))
        mask = (mask > 0.5).astype(np.uint8)

        
        top, left = int(dwdh[1]), int(dwdh[0])
        mask = mask[top:640 - top, left:640 - left]
        mask = cv2.resize(mask, (w0, h0))

        
        big_mask = np.zeros((h0, w0), dtype=np.uint8)
        big_mask[y0:y1, x0:x1] = mask[y0:y1, x0:x1]

        
        kernel = np.ones((3, 3), np.uint8)
        big_mask = cv2.morphologyEx(big_mask, cv2.MORPH_OPEN, kernel)
        big_mask = cv2.morphologyEx(big_mask, cv2.MORPH_CLOSE, kernel)

        masks.append(big_mask)

    # NMS
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)
    final_boxes, final_scores, final_class_ids, final_masks = [], [], [], []
    if len(indices) > 0:
        for i in indices.flatten():
            final_boxes.append(boxes[i])
            final_scores.append(scores[i])
            final_class_ids.append(class_ids[i])
            final_masks.append(masks[i])

    final_image = orig.copy()

    
    for idx, mask in enumerate(final_masks):
        color = colors[idx % len(colors)]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(final_image, contours, -1, color, 2)

       
        colored_mask = np.zeros_like(final_image, np.uint8)
        for i in range(3):
            colored_mask[:, :, i] = mask * color[i]
        final_image = cv2.addWeighted(final_image, 1, colored_mask, 0.5, 0)

   
    for idx, (box, score, cls_id) in enumerate(zip(final_boxes, final_scores, final_class_ids)):
        color = colors[idx % len(colors)]
        class_name = class_names[cls_id] if 0 <= cls_id < len(class_names) else f"id:{cls_id}"
        label = f"{class_name} {score:.2f}"

        #cv2.rectangle(final_image, (box[0], box[1]), (box[2], box[3]), color, 2)
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(final_image, (box[0], box[1] - text_h - 4), (box[0] + text_w, box[1]), color, -1)
        cv2.putText(final_image, label, (box[0], box[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Segmented Result", final_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Finished video processing.")


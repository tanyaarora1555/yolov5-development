import cv2
import numpy as np
import argparse
import time

def load_names(names_path):
    with open(names_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def detect(onnx_path, source, names_path, img_size):
    print(f"Loading ONNX model from: {onnx_path}")
    net = cv2.dnn.readNetFromONNX(onnx_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{source}'")
        return

    class_names = load_names(names_path)
    
    
    start_total_time = time.time()
    total_frames = 0
    total_detections = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or error reading frame.")
            break

        img0 = frame.copy()
        
        blob = cv2.dnn.blobFromImage(img0, 1/255.0, (img_size, img_size), swapRB=True, crop=False)
        net.setInput(blob)

        outputs = net.forward(net.getUnconnectedOutLayersNames())
        outputs = np.squeeze(outputs[0]).T

        rows, cols = outputs.shape
        boxes, scores, class_ids = [], [], []

        for i in range(cols):
            cx, cy, w, h = outputs[0:4, i]
            conf = outputs[4, i]

            if conf < 0.25:
                continue

            class_scores = outputs[5:, i]
            class_id = np.argmax(class_scores)
            score = class_scores[class_id] * conf

            if score < 0.6:
                continue

            x1 = cx - w / 2
            y1 = cy - h / 2
            
            x_factor = img0.shape[1] / img_size
            y_factor = img0.shape[0] / img_size
            
            x1_orig = int(x1 * x_factor)
            y1_orig = int(y1 * y_factor)
            w_orig = int(w * x_factor)
            h_orig = int(h * y_factor)
            
            boxes.append([x1_orig, y1_orig, w_orig, h_orig])
            scores.append(score)
            class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.6, 0.4)
        
        # Add the number of detections in the current frame to the total count
        if len(indices) > 0:
            total_detections += len(indices)
            
            for i in indices.flatten():
                box = boxes[i]
                x1, y1, w, h = box
                x2 = x1 + w
                y2 = y1 + h

                label = class_names[class_ids[i]] if class_ids[i] < len(class_names) else str(class_ids[i])
                color = (0, 255, 0)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                text_x = x1
                text_y = y1 - 10
                if text_y < 10:
                    text_y = y1 + 20

                (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (text_x, text_y - text_h - baseline), (text_x + text_w, text_y + baseline), color, -1)
                cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        cv2.imshow("Detections", frame)
        
        total_frames += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Calculate and print total metrics after the loop ends
    end_total_time = time.time()
    total_time_elapsed = end_total_time - start_total_time
    
    avg_fps = total_frames / total_time_elapsed if total_time_elapsed > 0 else 0

   
    print(f"Total frames processed: {total_frames}")
    print(f"Total detections: {total_detections}")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Total time elapsed: {total_time_elapsed:.2f} seconds")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', required=True, help='Path to ONNX model')
    parser.add_argument('--source', required=True, help='Path to video file or 0 for webcam')
    parser.add_argument('--names', required=True, help='Path to class names file')
    parser.add_argument('--img-size', type=int, default=640, help='Image size for inference (height and width)')
    args = parser.parse_args()

    detect(args.onnx, args.source, args.names, args.img_size)

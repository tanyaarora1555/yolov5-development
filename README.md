# Yolov5 inferencing and training for obejct dtection and segmentation
# Folder structure
## Project Structure

This project is organized into two main sections: one for object detection and one for image segmentation.

```text
.
├── detection/
│   ├── detection_weights/
│   │   └── yolov5m_best.pt  
│   ├── classes.names
│   ├── detect_object.py
│   └── strawberry.mp4
├── segmentation/
│   ├── segmentation_weights/
│   │   └── yolov5m_best.pt
│   ├── data.yaml
│   ├── mask_video.py
│   └── papaya.mp4
├── export_pt.py
└── README.md
```
# Converting .pt to .onnx
 * We need to ensure that the pt model is inside the yolov5 directory
 * The command to run this
   ``` bash
   python3 export_pt.py <path of pytorch model> <path of onnx model> 416
   ```
   In my case it was
   ``` bash
   python3 export_pt.py /home/tanya/yolov5/runs/train/exp23/weights/best.pt /home/tanya/yolov5/runs/train/exp23/weights/best.onnx 416
   ```
   Here 416 is the image size because the model was trained for 416 image size
# Object Detection
## Inferencing 
* To go the detection directory
  ``` bash
  cd detection
  ```
* The command for inferencing using onnx model
  ``` bash
  python detect_object.py --onnx /path/to/your/best.onnx --source <path_to_your_video> --names /path/to/your/classes.names --img-size 416
  ```
```bash
python3 detect_object.py --onnx ./detection_weights/best.onnx --source "./strawberry.mp4" --names ./classes.names --img-size 416
```
# Segmentation

## Training
* To train the model yolov5s-seg on custom data we need to run this command:
```bash
 python3 segment/train.py --weights yolov5s-seg.pt --data /home/tanya/seg/data.yaml --epochs 25 --img 640 --batch-size 16
 ```
 ## Inferencing
 * To go the segmentation directory
  ``` bash
  cd segmentation
  ```
* For inferencing we need to run
  ``` bash
  python3 mask_video.py
  ```
For changing the video and model paths in mask_video.py we need to give the path in the code.
   

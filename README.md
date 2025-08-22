#Yolov5 inferencing and training for obejct dtection and segmentation
## To train the model yolov5s-seg on custom data 
```bash
 python3 segment/train.py --weights yolov5s-seg.pt --data /home/tanya/seg/data.yaml --epochs 25 --img 640 --batch-size 16
 ```
 ## To run convert model from .pt to .onnx
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
   

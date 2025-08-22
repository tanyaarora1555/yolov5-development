# To train the model on yolov5s-seg
``bash
 python3 segment/train.py --weights yolov5s-seg.pt --data /home/tanya/seg/data.yaml --epochs 25 --img 640 --batch-size 16
 ``
 ## To run convert model from .pt to .onnx
 * We need to ensure that the pt model is inside the yolov5 directory
 * The command to run this
   `` bash
   python3 export_pt.py <path of pytorch model> <path of onnx model> 416
   ``
   In my case it was
   `` bash
   python3 export_pt.py /home/tanya/yolov5/runs/train/exp23/weights/best.pt /home/tanya/yolov5/runs/train/exp23/weights/best.onnx 416
   ``
   

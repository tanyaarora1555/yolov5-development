import torch
import sys
import os
from pathlib import Path
import onnx

def add_yolov5_to_path(start_path):
    folder = Path(start_path).resolve().parent
    while folder.parent != folder:
        if (folder / "models").is_dir() and (folder / "utils").is_dir():
            if str(folder) not in sys.path:
                sys.path.append(str(folder))
                print(f"YOLOv5 repo detected and added to sys.path: {folder}")
            return True
        folder = folder.parent
    
    print("Could not auto-detect YOLOv5 repo — assuming model class is already importable.")
    return False

def convert_to_onnx(model_path, output_path, img_size=640, opset_version=12):
    if not add_yolov5_to_path(model_path):
        print("Cannot proceed without YOLOv5 modules. Please ensure this script is run within or alongside the YOLOv5 repo.")
        sys.exit(1)

    
    print(f"Loading PyTorch model from {model_path}")
    
    try:
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        model = ckpt['model'] if 'model' in ckpt else ckpt
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)
        
    model.float()
    model.eval()

    dummy_input = torch.randn(1, 3, img_size, img_size, dtype=torch.float32)
    
    
    try:

        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version
        )
        print("ONNX export successful.")
    except Exception as e:
        print(f"ONNX export failed: {e}")
        sys.exit(1)

    print(f"Model converted to ONNX at: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python export_pt.py <model.pt> <model.onnx> [img_size] [opset_version]")
        sys.exit(1)

    model_path = sys.argv[1]
    output_path = sys.argv[2]
    img_size = int(sys.argv[3]) if len(sys.argv) > 3 else 640
    opset_version = int(sys.argv[4]) if len(sys.argv) > 4 else 12

    if not Path(model_path).exists():
        print(f"Input file not found: {model_path}")
        sys.exit(1)

    convert_to_onnx(model_path, output_path, img_size, opset_version)


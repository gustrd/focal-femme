# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "requests",
#     "tqdm",
#     "facenet-pytorch",
#     "torch",
#     "torchvision"
# ]
# ///
import os
import sys
from pathlib import Path
import requests
from tqdm import tqdm

def download_file(url: str, destination: Path):
    """Download a file with a progress bar."""
    if destination.exists():
        print(f"Already exists: {destination.name}")
        return

    print(f"Downloading {destination.name}...")
    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        chunk_size = 8192
        
        with open(destination, 'wb') as f, tqdm(
            desc=destination.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    size = f.write(chunk)
                    bar.update(size)
    except Exception as e:
        if destination.exists():
            destination.unlink() # Cleanup partial download
        raise e

def setup():
    """Download all required models for focal-femme."""
    cache_dir = Path.home() / ".cache" / "focal-femme"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    models = [
        {
            "name": "retinaface_resnet50.pth",
            "#url_comment": "OpenVINO Model Zoo mirror for RetinaFace ResNet50",
            "url": "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/1/retinaface-resnet50-pytorch/retinaface-resnet50-pytorch/Resnet50_Final.pth",
            "desc": "RetinaFace face detector (ResNet50 backbone)"
        },
        {
            "name": "beauty_resnet18.pth",
            "url": "https://huggingface.co/Gustrd/SCUT-FBP5500-PyTorch-Model/resolve/main/resnet18_py3.pth",
            "desc": "SCUT-FBP5500 beauty predictor (ResNet-18 backbone)"
        },
        {
            "name": "gender_googlenet.onnx",
            "url": "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/age_gender/models/gender_googlenet.onnx",
            "desc": "Gender classifier (GoogleNet ONNX)"
        }
    ]
    
    print("=== focal-femme Model Setup ===")
    print(f"Target directory: {cache_dir}\n")
    
    for model in models:
        dest = cache_dir / model["name"]
        try:
            download_file(model["url"], dest)
        except Exception as e:
            print(f"Error downloading {model['name']}: {e}")
            if "manual_instruction" in model:
                print(f"ACTION REQUIRED: {model['manual_instruction']}")
                print(f"Target Path: {dest}")
            
    print("\nTriggering automatic download for InceptionResnetV1 (VGGFace2)...")
    try:
        from facenet_pytorch import InceptionResnetV1
        import torch
        # This will trigger download if not present
        InceptionResnetV1(pretrained='vggface2').eval()
        print("InceptionResnetV1 ready.")
    except Exception as e:
        print(f"Note: Could not pre-trigger InceptionResnetV1 download: {e}")
        print("This will be downloaded automatically during first run.")

    print("\nAll models processed.")

if __name__ == "__main__":
    try:
        setup()
    except KeyboardInterrupt:
        print("\nDownload interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)

import os
import sys
from pathlib import Path
import requests
from tqdm import tqdm
import click

def download_file(url: str, destination: Path):
    """Download a file with a progress bar."""
    if destination.exists():
        click.echo(f"Already exists: {destination.name}")
        return

    click.echo(f"Downloading {destination.name}...")
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

def setup_models():
    """Download all required models for focal-femme."""
    cache_dir = Path.home() / ".cache" / "focal-femme"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    models = [
        {
            "name": "retinaface_resnet50.pth",
            "url": "https://huggingface.co/shilongz/FlashFace-SD1.5/resolve/main/retinaface_resnet50.pth",
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
    
    click.echo("=== focal-femme Model Setup ===")
    click.echo(f"Target directory: {cache_dir}\n")
    
    for model in models:
        dest = cache_dir / model["name"]
        
        # Check if file exists but is suspiciously small (likely an error page like a 404 HTML)
        if dest.exists() and dest.stat().st_size < 1024 * 1024: # Less than 1MB
            click.echo(f"File {dest.name} is suspiciously small ({dest.stat().st_size} bytes). Triggering re-download.")
            dest.unlink()

        try:
            download_file(model["url"], dest)
        except Exception as e:
            click.echo(f"Error downloading {model['name']}: {e}", err=True)
            if "manual_instruction" in model:
                click.echo(f"ACTION REQUIRED: {model['manual_instruction']}", err=True)
                click.echo(f"Target Path: {dest}", err=True)
            
    click.echo("\nTriggering automatic download for InceptionResnetV1 (VGGFace2)...")
    try:
        from facenet_pytorch import InceptionResnetV1
        import torch
        # This will trigger download if not present
        InceptionResnetV1(pretrained='vggface2').eval()
        click.echo("InceptionResnetV1 ready.")
    except Exception as e:
        click.echo(f"Note: Could not pre-trigger InceptionResnetV1 download: {e}", err=True)
        click.echo("This will be downloaded automatically during first run.", err=True)

    click.echo("\nAll models processed.")

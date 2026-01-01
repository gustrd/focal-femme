# focal-femme

A Python CLI tool for automatic face clustering of photo collections with focus on the primary female subject in each image.

## Features

- Detects all faces in images and filters for female faces
- Selects the "primary" female face (largest, most central)
- Clusters similar faces using DBSCAN algorithm
- **Beauty scoring**: Predicts attractiveness score for each face using SCUT-FBP5500 (ResNet-18)
- Renames files with cluster-based prefixes including normalized beauty score (e.g., `person_001_85_IMG_1234.jpg`)
- Persists embeddings to avoid reprocessing on subsequent runs
- Supports incremental processing of photo collections

## Installation

Requires Python 3.11+

```bash
# Clone the repository
git clone https://github.com/yourusername/focal-femme.git
cd focal-femme

# Install with uv
uv sync

# Or install with pip
pip install -e .
```

### Model Downloads

First run will automatically download models (~400MB total, cached locally). However, you can pre-download everything using the provided script:

```bash
# Windows
.\install_models.bat

# Linux/Mac
chmod +x install_models.sh
./install_models.sh

# Or directly with uv
uv run scripts/setup_models.py
```

**Included Models:**
- **RetinaFace ResNet50** (~100MB) - Face Detection
- **SCUT-FBP5500 ResNet-18** (~90MB) - Beauty Prediction
- **VGGFace2 InceptionResnetV1** (~110MB) - Face Embeddings
- **Gender Classifier GoogleNet** (~50MB) - Gender Classification (ONNX)

### Notes

- All processing is local; no data leaves your machine
- Uses PyTorch for face detection/embeddings/beauty scoring, ONNX for gender classification

## GPU Acceleration

The tool automatically detects and uses the best available device in this order:

| Priority | Device | Platform | Requirements |
|----------|--------|----------|--------------|
| 1 | CUDA | Windows/Linux | NVIDIA GPU + CUDA drivers |
| 2 | XPU | Windows/Linux | Intel Arc GPU + XPU PyTorch |
| 3 | MPS | macOS | Apple Silicon (M1/M2/M3) |
| 4 | CPU | All | Fallback (slower) |

Use `-v` flag to see which device is being used:
```bash
focal-femme -v /path/to/photos
# DEBUG: Using CUDA device: NVIDIA GeForce RTX 4090
```

### Intel XPU Setup (Intel Arc GPUs)

Intel XPU requires the nightly PyTorch build with XPU support:

```bash
# 1. Install XPU-enabled PyTorch (replaces standard torch)
uv pip install -U --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu

# 2. Run with --no-sync to prevent uv from overwriting XPU packages
uv run --no-sync focal-femme /path/to/photos -v
# DEBUG: Using XPU device: Intel(R) Arc(TM) ...
```

**Important:** Always use `uv run --no-sync` when using XPU PyTorch, otherwise `uv run` will reinstall the standard CPU PyTorch from pyproject.toml dependencies.

### ONNX Runtime Acceleration

The gender classifier uses ONNX Runtime, which supports GPU acceleration via several providers:

| Provider | Platform | Requirements |
|----------|----------|--------------|
| CUDA | Windows/Linux | `onnxruntime-gpu` package |
| **OpenVINO** | Windows/Linux | `onnxruntime-openvino` (Intel CPU/GPU optimized) |
| DirectML | Windows | AMD/Intel/NVIDIA via DirectX 12 |
| CoreML | macOS | Apple Neural Engine |
| CPU | All | Default fallback |

We recommend **OpenVINO** for Intel Core Ultra and Intel Arc users as it provides significant acceleration for gender classification. The tool automatically configures the OpenVINO DLL search path on Windows.

**Note:** Version compatibility is critical - `onnxruntime-openvino 1.23.0` requires `openvino 2025.3.x`. The correct version is specified in `pyproject.toml`.

## Usage

### Basic Usage

```bash
focal-femme /path/to/photos
```

### Options

```
focal-femme [OPTIONS] FOLDER

Arguments:
  FOLDER                  Directory containing images to process

Options:
  --eps FLOAT            DBSCAN epsilon parameter (default: 0.5)
                         Lower values = stricter clustering
  --min-samples INT      DBSCAN min_samples (default: 2)
                         Minimum images to form a cluster
  --dry-run              Preview changes without renaming
  --reset                Clear existing embeddings cache
  --verbose, -v          Show detailed processing info
  --female-threshold     Minimum confidence for female classification (0-1)
  --version              Show version and exit
  --help                 Show this message and exit
```

### Examples

Preview what would be renamed:
```bash
focal-femme --dry-run /path/to/photos
```

Stricter clustering (fewer false matches):
```bash
focal-femme --eps 0.4 /path/to/photos
```

Reset and reprocess all images:
```bash
focal-femme --reset /path/to/photos
```

## How It Works

1. **Face Detection**: Uses RetinaFace (ResNet50 backbone) for accurate face detection
2. **Gender Classification**: Uses ONNX gender model (GoogleNet) to filter for female faces
3. **Primary Selection**: Selects the largest female face as the primary subject
4. **Beauty Scoring**: Predicts attractiveness score (1-5 scale) using SCUT-FBP5500 ResNet-18
5. **Embedding Extraction**: Generates 512-dimensional VGGFace2 embeddings via InceptionResnetV1
6. **Clustering**: Groups similar embeddings using DBSCAN with cosine distance
7. **Renaming**: Adds cluster prefix with normalized beauty score

## File Naming Format

Files are renamed with the format: `person_XXX_YY_originalname.ext`

- `XXX` = Cluster ID (000-999)
- `YY` = Normalized beauty score (00-99, relative to other clusters in the dataset)

Example output:
```
Cluster breakdown:
  person_000_99: 15 images, avg beauty: 3.78 (norm: 99)
  person_001_45: 8 images, avg beauty: 3.12 (norm: 45)
  person_002_00: 3 images, avg beauty: 2.85 (norm: 00)
```

The beauty score is normalized using min-max scaling across all clusters:
- Cluster with highest average beauty score gets `99`
- Cluster with lowest average beauty score gets `00`
- Other clusters are scaled proportionally

## Supported Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- GIF (.gif)
- WebP (.webp)

## Development

```bash
# Install dev dependencies
uv sync --extra dev

# Run tests
pytest

# Run tests with coverage
pytest --cov=focal_femme
```

## Future Improvements

### Face Embedding Upgrade
Currently uses InceptionResnetV1 (VGGFace2) for face embeddings in clustering. Consider upgrading to:
- **ArcFace embeddings** via InsightFace for better clustering accuracy
- **Integration**: RetinaFace already provides face landmarks, making ArcFace alignment easier
- **Benefits**: State-of-the-art face recognition accuracy, better cluster separation
- **Trade-off**: Additional dependency, slightly larger models

See [InsightFace documentation](https://github.com/deepinsight/insightface) for implementation details.

## License

MIT License

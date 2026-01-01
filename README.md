# focal-femme

A Python CLI tool for automatic face clustering of photo collections with focus on the primary female subject in each image.

## Features

- Detects all faces in images and filters for female faces
- Selects the "primary" female face (largest, most central)
- Clusters similar faces using DBSCAN algorithm
- Renames files with cluster-based prefixes (e.g., `person_001_IMG_1234.jpg`)
- Persists embeddings to avoid reprocessing on subsequent runs
- Supports incremental processing of new images

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

### Notes

- First run will download models (~100MB for MTCNN + VGGFace2 + gender classifier)
- All processing is local; no data leaves your machine
- Uses PyTorch for face detection/embeddings, ONNX for gender classification
- GPU acceleration available if CUDA is installed

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

1. **Face Detection**: Uses MTCNN (facenet-pytorch) to detect faces
2. **Gender Classification**: Uses ONNX gender model (GoogleNet) to filter for female faces
3. **Primary Selection**: Selects the largest female face as the primary subject
4. **Embedding Extraction**: Generates 512-dimensional VGGFace2 embeddings
5. **Clustering**: Groups similar embeddings using DBSCAN
6. **Renaming**: Adds cluster prefix to organize by person

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

## License

MIT License

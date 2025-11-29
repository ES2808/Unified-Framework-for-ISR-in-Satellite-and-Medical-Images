# Unified Super-Resolution Project

This project implements a Unified Super-Resolution model capable of handling both Medical (X-ray) and Satellite imagery. It uses a shared backbone with domain-specific heads to achieve high-quality upscaling.

## Features
- **Unified Architecture**: Single model for multiple domains.
- **Domain Auto-Detection**: Automatically detects if an image is Medical or Satellite.
- **Tiled Inference**: Handles large images (like satellite maps) without running out of memory.
- **Self-Ensemble**: Optional test-time augmentation for higher quality.

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended) or CPU

### 1. Clone the Repository
```bash
git clone https://github.com/ES2808/Unified-Framework-for-ISR-in-Satellite-and-Medical-Images.git

cd unified_sr_project
```

### 2. Install Dependencies
You can use `pip` or `conda`.

**Using Pip:**
```bash
pip install -r requirements.txt
```

**Using Conda:**
```bash
conda create -n unified_sr python=3.8
conda activate unified_sr
pip install -r requirements.txt
```

## Usage

### Quick Start (Demo Notebook)
The easiest way to use the model is via the Jupyter Notebook:
1. Start Jupyter:
   ```bash
   jupyter notebook
   ```
2. Open `demo.ipynb`.
3. Set your `IMAGE_PATH` and run the cells to see the result.

### Command Line Inference
You can also run inference directly from the terminal:

```bash
python inference.py --input path/to/image.png --output results/ --scale 4
```

**Options:**
- `--domain`: 'medical' or 'satellite' (Optional, auto-detected if omitted)
- `--self_ensemble`: Enable for slightly better quality (slower)

### Training
To train the model from scratch:
```bash
python train.py --medical_data data/medical --satellite_data data/satellite
```

## Repository Structure

```
unified_sr_project/
├── checkpoints/          # Trained model weights
├── data/                 # Dataset directory (Medical/Satellite)
├── src/                  # Source code
│   ├── models/           # Model architectures (UnifiedModel, Heads, etc.)
│   └── utils/            # Utilities (Dataset, Metrics, Preprocessing)
├── demo.ipynb            # Interactive demo notebook
├── inference.py          # Script for running inference
├── train.py              # Main training script
├── app.py                # Streamlit Web Interface (Legacy)
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```



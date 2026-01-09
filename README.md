## 📁 Structure
```
SemanticTrainSys/
├── semantic_adapter.py       # Main training script # Please Change your name on PERSON_ID on Configuration Line#640-650 IMPORTANT
├── dataset/
│   ├── images_color/         # Original color images
│   ├── images_gray/          # Auto-generated grayscale 
│   └── seg_maps/             # Auto-generated segmentation maps
├── checkpoints/              # Trained model checkpoints
└── README.md
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch torchvision diffusers transformers accelerate
pip install ultralytics opencv-python pillow numpy tqdm
```

### Training

1. **Add your images** to `dataset/images_color/`

2. **Run training**:
```bash
   python semantic_adapter.py
   # Choose option 1: Train on your own dataset
```

3. **Monitor progress** - Checkpoints saved to `checkpoints/`

## 👥 Distributed Training

Each team member can train independently:
```bash
# Set your unique ID
PERSON_ID = "yourname"  # in semantic_adapter.py

# Train on your dataset
python semantic_adapter.py
```

### Merging Checkpoints

Collect all team checkpoints and merge:
```bash
python semantic_adapter.py
# Choose option 2: Merge checkpoints
```

## 📊 Checkpoints

- `adapter_YOURNAME_latest.pt` - Individual checkpoints
- `merged_adapter_YYYYMMDD_HHMMSS.pt` - Merged team checkpoint

## 🔗 Production App

The trained model is used in the production app on the **main** branch.

## 📝 Notes

- Dataset with color images should be committed
- Processed images (gray/seg) are auto-generated
- Checkpoints can be large - share via cloud storage if needed

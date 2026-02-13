# Deep Learning Computer Vision Exercises

PyTorch project with 5 exercises:
1. CIFAR-100 classification (Vanilla CNN)
2. Oxford Pet transfer learning
3. SBD semantic segmentation (U-Net)
4. Oxford Pet object detection (Faster R-CNN)
5. CNN vs ViT comparison on CIFAR-10

## Project Structure

```text
ai-images/
├── cifar_data_loaders.py
├── pet_data_loaders.py
├── training_utils.py
├── visualization_utils.py
├── ex1_vanilla_cnn.py
├── ex1_main_experiments.py
├── ex1_quick_experiment.py
├── ex1_analyze_results.py
├── ex2_transfer_learning_models.py
├── ex2_transfer_learning_experiments.py
├── ex3_sbd_data_loader.py
├── ex3_unet_model.py
├── ex3_segmentation_experiments.py
├── ex4_detection_model.py
├── ex4_detection_experiments.py
├── ex5_cnn_models.py
├── ex5_vit_model.py
├── ex5_comparative_experiments.py
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Run Experiments

### Exercise 1: CIFAR-100 Vanilla CNN

```bash
# Full experiment suite
python ex1_main_experiments.py

# Quick test (forces 5 epochs)
python ex1_main_experiments.py --quick_test

# One quick standalone run
python ex1_quick_experiment.py --epochs 10
```

Useful flags:
- `--epochs`
- `--batch_size`
- `--results_dir`
- `--quick_test`

Analyze Ex1 results:

```bash
python ex1_analyze_results.py <path_to_ex1_experiments_dir> --plot
```

### Exercise 2: Transfer Learning (Oxford Pet)

```bash
# Full experiment suite
python ex2_transfer_learning_experiments.py

# Quick test (forces 5 epochs)
python ex2_transfer_learning_experiments.py --quick_test
```

Useful flags:
- `--epochs`
- `--batch_size`
- `--results_dir`
- `--quick_test`

### Exercise 3: Semantic Segmentation (SBD + U-Net)

```bash
# Compact suite (2 optimizers + 2 learning rates)
python ex3_segmentation_experiments.py

# Quick test (forces 3 epochs)
python ex3_segmentation_experiments.py --quick_test

# Example with lower background class weight
python ex3_segmentation_experiments.py --background_weight 0.2
```

Useful flags:
- `--epochs`
- `--batch_size`
- `--num_workers`
- `--image_size`
- `--results_dir`
- `--quick_test`
- `--background_weight`

### Exercise 4: Object Detection (Oxford Pet + Faster R-CNN)

```bash
# Compact suite (default, 4 experiments)
python ex4_detection_experiments.py

# Full grid (6 experiments)
python ex4_detection_experiments.py --full_grid

# Single experiment
python ex4_detection_experiments.py --single --backbone mobilenet --lr 0.001 --epochs 4
```

Useful flags:
- `--quick_test`
- `--full_grid`
- `--single`
- `--backbone {resnet50,mobilenet}`
- `--lr`
- `--epochs`
- `--batch_size`
- `--results_dir`
- `--data_dir`
- `--no_pretrained`

### Exercise 5: CNN vs Transformer (CIFAR-10)

```bash
# Compact suite (default, 4 experiments)
python ex5_comparative_experiments.py

# Full grid
python ex5_comparative_experiments.py --full_grid

# Single experiment
python ex5_comparative_experiments.py --single --architecture vit --model tiny --lr 0.001 --epochs 10 --batch_size 64
```

Useful flags:
- `--quick_test`
- `--full_grid`
- `--single`
- `--architecture {cnn,vit}`
- `--model`
- `--lr`
- `--epochs`
- `--batch_size`
- `--results_dir`

## Outputs

By default, scripts save outputs under these roots:
- `results`
- `results_transfer`
- `results_segmentation`
- `results_detection`
- `results_comparative`

Most runs produce:
- per-experiment folders with `results.json` (or `experiment_results.json`)
- `training_curves.png`
- prediction visualizations (`sample_predictions.png`, `segmentation_results.png`, `detections.png`, etc.)
- summary/report files (`experiments_summary.json`, `experiment_report.txt`, `comparative_report.txt`)

## Notes

- Datasets are downloaded automatically via TorchVision/custom loaders.
- GPU is optional but strongly recommended for Ex4 and Ex5.
- This is an educational project and scripts are optimized for clarity and report generation.

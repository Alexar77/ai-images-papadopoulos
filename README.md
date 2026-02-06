# Deep Learning Exercises - Computer Vision Tasks

Υλοποίηση 5 ασκήσεων βαθιάς μάθησης για Computer Vision με PyTorch

## 📁 Δομή Project

```
ai-images/
│
├── 📂 Unified Utilities (4 files)
│   ├── cifar_data_loaders.py        # CIFAR-10 & CIFAR-100 data loading
│   ├── pet_data_loaders.py          # Oxford Pet (classification & detection)
│   ├── training_utils.py            # All trainers (Classification, Segmentation, Detection)
│   └── visualization_utils.py       # All plotting & report generation
│
├── 📂 Exercise #1: CIFAR-100 Classification (Vanilla CNN)
│   ├── ex1_vanilla_cnn.py           # Custom 4-layer CNN architecture
│   ├── ex1_main_experiments.py      # Hyperparameter comparison experiments
│   ├── ex1_quick_experiment.py      # Quick test script
│   └── ex1_analyze_results.py       # Results analysis tool
│
├── 📂 Exercise #2: Transfer Learning (Oxford Pet)
│   ├── ex2_transfer_learning_models.py      # Pretrained models (ResNet, VGG, EfficientNet)
│   └── ex2_transfer_learning_experiments.py # Transfer learning experiments
│
├── 📂 Exercise #3: Semantic Segmentation (SBD)
│   ├── ex3_sbd_data_loader.py       # SBD dataset with 21 Pascal VOC classes
│   ├── ex3_unet_model.py            # U-Net architecture
│   └── ex3_segmentation_experiments.py # Segmentation experiments
│
├── 📂 Exercise #4: Object Detection (Oxford Pet)
│   ├── ex4_detection_model.py       # Faster R-CNN (ResNet50/MobileNet)
│   └── ex4_detection_experiments.py # Detection experiments
│
├── 📂 Exercise #5: CNN vs Transformer (CIFAR-10)
│   ├── ex5_cnn_models.py            # VGG, ResNet18, ResNet50
│   ├── ex5_vit_model.py             # Vision Transformer (from scratch)
│   └── ex5_comparative_experiments.py # Comparative study
│
├── requirements.txt                  # Dependencies
└── README.md                         # This file
```

## 🎯 Περιγραφή Ασκήσεων

### Άσκηση #1: Vanilla CNN για CIFAR-100
**Dataset**: CIFAR-100 (100 classes, 50K train, 10K test)  
**Task**: Image classification  
**Model**: Custom 4-layer CNN (964K parameters)  
**Experiments**: Loss functions, optimizers, learning rates (9 total)  
**Best Result**: 33.43% test accuracy (LR=0.1, SGD)

```
Architecture: Input(3×32×32) → Conv→Pool×4 → FC→Dropout → Output(100)
Parameters: 964,516 trainable
```

### Άσκηση #2: Transfer Learning
**Dataset**: Oxford-IIIT Pet (37 breeds, 7K images)  
**Task**: Image classification with pretrained models  
**Models**: ResNet18/50, AlexNet, VGG16, EfficientNet-B0  
**Experiments**: Architecture comparison, frozen vs fine-tuned, learning rates  
**Expected**: 85-92% test accuracy

### Άσκηση #3: Semantic Segmentation
**Dataset**: SBD - Semantic Boundaries Dataset (21 Pascal VOC classes)  
**Task**: Pixel-level segmentation  
**Model**: U-Net (encoder-decoder with skip connections)  
**Experiments**: Model sizes (base channels: 32/64/128), optimizers, learning rates  
**Metrics**: Pixel Accuracy, Mean IoU  
**Expected**: 40-75% mIoU

```
U-Net Architecture: Encoder (4 down) → Bottleneck → Decoder (4 up + skip connections)
```

### Άσκηση #4: Object Detection
**Dataset**: Oxford-IIIT Pet (37 breeds)  
**Task**: Object detection with bounding boxes  
**Model**: Faster R-CNN with FPN (ResNet50/MobileNet backbone)  
**Experiments**: Backbones, optimizers, learning rates (8 total)  
**Metrics**: Training loss components (classifier, box regression, RPN)  
**Expected**: Final loss ~0.4-0.7

### Άσκηση #5: Συγκριτική Αξιολόγηση CNN vs Transformer
**Dataset**: CIFAR-10 (10 classes, 60K images)  
**Task**: Comparative study of architectures  
**CNN Models**: VGG11, ResNet18, ResNet50 (32×32 images)  
**Transformer Models**: ViT-Tiny, ViT-Small (224×224 images, from scratch)  
**Experiments**: 8 configurations comparing architectures and hyperparameters  
**Expected**: CNNs 85-93%, ViTs 75-87%

```
ViT Architecture: Patch Embedding → Transformer Encoder (Multi-Head Attention + MLP) × 12 → Classification Head
ViT-Tiny: 192 embed_dim, 3 heads, 5.7M params
ViT-Small: 384 embed_dim, 6 heads, 22M params
```

## 📦 Εγκατάσταση

### Προαπαιτούμενα
- Python 3.8+
- CUDA (προαιρετικό, για GPU acceleration)

### Βήματα Εγκατάστασης

1. Κλωνοποίηση/Λήψη του project

2. Δημιουργία virtual environment (προτείνεται):
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

3. Εγκατάσταση dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Εκτέλεση

### Άσκηση #1: CIFAR-100 Vanilla CNN

```bash
# Full experiments (9 experiments, ~9 hours)
python ex1_main_experiments.py

# Quick test (5 epochs, ~10 minutes)
python ex1_main_experiments.py --quick_test

# Single experiment
python ex1_quick_experiment.py --epochs 10 --lr 0.01
```

### Άσκηση #2: Transfer Learning

```bash
# Full experiments (~1-2 hours)
python ex2_transfer_learning_experiments.py

# Quick test (10 epochs)
python ex2_transfer_learning_experiments.py --quick_test

# Single model
python ex2_transfer_learning_experiments.py --single --model resnet18 --epochs 20
```

### Άσκηση #3: Semantic Segmentation

```bash
# Full experiments (~2-3 hours)
python ex3_segmentation_experiments.py

# Quick test (10 epochs, small batch)
python ex3_segmentation_experiments.py --quick_test --batch_size 4

# Single experiment
python ex3_segmentation_experiments.py --single --base_channels 64 --epochs 30
```

### Άσκηση #4: Object Detection

```bash
# Full experiments (8 experiments, ~2-3 hours)
python ex4_detection_experiments.py

# Quick test (3 epochs)
python ex4_detection_experiments.py --quick_test

# Single experiment
python ex4_detection_experiments.py --single --backbone resnet50 --lr 0.005 --epochs 5
```

### Άσκηση #5: Comparative Study (CNN vs Transformer)

```bash
# Full comparative study (8 experiments, ~4-6 hours)
python ex5_comparative_experiments.py

# Quick test (5 epochs)
python ex5_comparative_experiments.py --quick_test

# Single architecture
python ex5_comparative_experiments.py --single --architecture cnn --model resnet18 --epochs 20
python ex5_comparative_experiments.py --single --architecture vit --model tiny --epochs 30
```

## 📊 Αποτελέσματα

Τα αποτελέσματα αποθηκεύονται σε ξεχωριστούς φακέλους:

```
results_cifar100/           # Άσκηση #1
results_transfer/           # Άσκηση #2
results_segmentation/       # Άσκηση #3
results_detection/          # Άσκηση #4
results_comparative/        # Άσκηση #5
```

Κάθε πείραμα δημιουργεί:
- `results.json` - Metrics και configuration
- `training_curves.png` - Loss & accuracy plots
- `predictions.png` / `detections.png` / `segmentation.png` - Sample results
- `experiments_summary.json` - Σύνοψη όλων των experiments
- `comparative_report.txt` - Detailed report (Άσκηση #5)

## 📈 Αναμενόμενα Benchmarks

| Exercise | Dataset | Metric | Expected |
|----------|---------|--------|----------|
| #1 | CIFAR-100 | Test Accuracy | 30-35% |
| #2 | Oxford Pet | Test Accuracy | 85-92% |
| #3 | SBD | Mean IoU | 40-75% |
| #4 | Oxford Pet | Final Loss | 0.4-0.7 |
| #5 (CNN) | CIFAR-10 | Test Accuracy | 85-93% |
| #5 (ViT) | CIFAR-10 | Test Accuracy | 75-87% |

## 🔧 Τεχνικές Λεπτομέρειες

### Βιβλιοθήκες που Χρησιμοποιούνται

**Core Deep Learning:**
- PyTorch 2.0+ (neural networks, optimization)
- TorchVision (datasets, pretrained models, transforms)

**Data Processing:**
- NumPy (numerical computations, array operations)

**Visualization:**
- Matplotlib (plotting graphs, images)
- Seaborn (enhanced styling)

**Utilities:**
- tqdm (progress bars)

### Datasets

Όλα τα datasets κατεβαίνουν αυτόματα:
- **CIFAR-10/100**: `torchvision.datasets.CIFAR10/100`
- **Oxford Pet**: `torchvision.datasets.OxfordIIITPet`
- **SBD**: Custom loader με automatic download

### Hardware Requirements

**Minimum:**
- CPU: 4+ cores
- RAM: 8GB
- Disk: 2GB για datasets

**Recommended:**
- GPU: NVIDIA with 4GB+ VRAM (για Άσκηση #5)
- RAM: 16GB
- Disk: 5GB

### Training Times (CPU estimates)

| Exercise | Quick Test | Full Experiments |
|----------|------------|------------------|
| #1 | 10 min | 9 hours |
| #2 | 15 min | 1-2 hours |
| #3 | 30 min | 2-3 hours |
| #4 | 20 min | 2-3 hours |
| #5 | 45 min | 4-6 hours |

## 🎓 Υλοποίηση

Το project αναπτύχθηκε σύμφωνα με τις οδηγίες των ασκήσεων:

✅ Χρήση PyTorch για deep learning  
✅ Χρήση NumPy για data processing  
✅ Χρήση Matplotlib για visualization  
✅ Όλες οι αρχιτεκτονικές υλοποιημένες από την αρχή (εκτός pretrained backbones)  
✅ Συγκριτική αξιολόγηση υπερ-παραμέτρων  
✅ Αναλυτικά αποτελέσματα και reports

## 📝 Δομή Κώδικα

### Unified Utilities (Optimized)

Τα shared utilities βελτιστοποιήθηκαν για:
- **Code Reuse**: Κοινές συναρτήσεις training/visualization
- **Consistency**: Ενιαίο API σε όλες τις ασκήσεις
- **Maintainability**: Single source of truth

### Trainers

- `ClassificationTrainer`: Για Ex1, Ex2, Ex5
- `SegmentationTrainer`: Για Ex3 με IoU metrics
- `DetectionTrainer`: Για Ex4 με Faster R-CNN loss components

### Data Loaders

- `cifar_data_loaders.py`: CIFAR-10 & CIFAR-100 με configurable augmentation
- `pet_data_loaders.py`: Oxford Pet για classification & detection
- `ex3_sbd_data_loader.py`: SBD με VOC classes

## 📞 Support

Για ερωτήσεις ή προβλήματα, ανατρέξτε στα comments μέσα στον κώδικα.

## 📄 License

Εκπαιδευτικό project για μαθησιακούς σκοπούς.

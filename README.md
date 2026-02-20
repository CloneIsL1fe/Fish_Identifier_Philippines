# Dataset
https://drive.google.com/drive/folders/1c1ra54iYNwY6148EBrQfZSdKGQ2jF6wo?usp=sharing

# Fish Identifier Philippines 🐟

Deep learning image classifier for identifying 5 common fish species in Philippine markets using Transfer Learning with MobileNetV2.

## Overview

This project uses transfer learning with MobileNetV2 to classify images of fish commonly found in Philippine wet markets. The model achieves high accuracy in identifying 5 species, making it useful for consumers, vendors, and researchers.

## Identified Fish Species

1. **Bangus** (Milkfish) - *Chanos chanos*
2. **Dalagang Bukid** (Yellow-tail Fusilier) - *Caesio cuning*
3. **Tilapia** (Nile Tilapia) - *Oreochromis niloticus*
4. **Hiwas** (Moonfish/Lookdown) - *Mene maculata*
5. **Tulingan** (Skipjack Tuna) - *Katsuwonus pelamis*

## Features

- 🧠 **Transfer Learning** - Uses pre-trained MobileNetV2 for efficient training
- 🎯 **5-Class Classification** - Identifies 5 common Philippine market fish
- 📊 **High Accuracy** - Validated on real market fish images
- 📸 **Image Preprocessing** - Handles various lighting conditions and angles
- 🚫 **Confidence Threshold** - Rejects low-confidence predictions
- 📓 **Jupyter Notebook** - Complete training and evaluation pipeline
- 📈 **Visualization** - Training metrics, confusion matrix, sample predictions

## Technologies Used

- **Python 3.8+**
- **TensorFlow 2.x** - Deep learning framework
- **MobileNetV2** - Pre-trained CNN for transfer learning
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation
- **Matplotlib** - Data visualization
- **Scikit-learn** - Metrics and evaluation
- **Google Colab** - GPU-accelerated training

## Why MobileNetV2?

MobileNetV2 was chosen for this project because:
- ✅ **Lightweight** - Fast inference, suitable for mobile deployment
- ✅ **Pre-trained** - Transfer learning reduces training time
- ✅ **Efficient** - Excellent accuracy-to-size ratio
- ✅ **Mobile-ready** - Can be deployed as TensorFlow Lite for Android/iOS apps

## Model Architecture

```
Input Image (224x224x3)
        ↓
MobileNetV2 Base (pre-trained on ImageNet)
   - Frozen layers (feature extraction)
   - Inverted residual blocks
   - Depthwise separable convolutions
        ↓
Global Average Pooling
        ↓
Dense Layer (128 units) + ReLU + Dropout(0.5)
        ↓
Dense Layer (5 units) + Softmax
        ↓
Output (5 fish species)
```

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/CloneIsL1fe/Fish_Identifier_Philippines.git
cd Fish_Identifier_Philippines
```

### 2. Install dependencies
```bash
pip install tensorflow numpy pandas matplotlib scikit-learn jupyter
```

### 3. For Google Colab (Recommended)
Upload `CommonFishPhilippines.ipynb` to Google Colab and run with GPU runtime:
1. Open Google Colab: https://colab.research.google.com
2. Upload notebook: `File` → `Upload notebook`
3. Enable GPU: `Runtime` → `Change runtime type` → `GPU (T4)`
4. Run all cells

## Dataset Structure

```
Philippines_Fish_Market_Dataset/
├── Bangus/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Dalagang bukid/
│   ├── image1.jpg
│   └── ...
├── Hiwas/
│   ├── image1.jpg
│   └── ...
├── Tilapia/
│   ├── image1.jpg
│   └── ...
└── Tulingan/
    ├── image1.jpg
    └── ...

## Training Configuration

### Data Augmentation
- **Rotation**: ±20 degrees
- **Width/Height shift**: ±20%
- **Zoom**: ±20%
- **Horizontal flip**: Yes
- **Fill mode**: Nearest

### Training Parameters
- **Image size**: 224 x 224 pixels
- **Batch size**: 32
- **Epochs**: 20-30 (with early stopping)
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss function**: Categorical Crossentropy
- **Metrics**: Accuracy, Precision, Recall, F1-Score

### Transfer Learning Strategy
1. **Load MobileNetV2** pre-trained on ImageNet
2. **Freeze base layers** (use as feature extractor)
3. **Add custom classification head** (Dense layers)
4. **Train only new layers** (fast convergence)
5. **Optional fine-tuning** (unfreeze top layers for better accuracy)

## Performance

### Model Metrics
- **Training Accuracy**: ~96%
- **Validation Accuracy**: ~93%
- **Test Accuracy**: ~91%
- **Model Size**: ~9 MB (lightweight!)
- **Inference Time**: ~50ms per image (on GPU)

### Per-Class Performance
| Fish Species | Precision | Recall | F1-Score |
|--------------|-----------|--------|----------|
| Bangus | 0.94 | 0.93 | 0.93 |
| Dalagang Bukid | 0.92 | 0.91 | 0.91 |
| Hiwas | 0.90 | 0.89 | 0.89 |
| Tilapia | 0.93 | 0.94 | 0.93 |
| Tulingan | 0.91 | 0.92 | 0.91 |

## Confidence Threshold Feature

The model includes a **confidence threshold** (default: 0.5):
- If prediction confidence < 50% → Returns "None of the common fishes"
- This prevents false classifications on unknown fish species
- Threshold can be adjusted based on use case

```python
# Strict threshold (fewer false positives)
classify_fish(img_path, threshold=0.8)

# Lenient threshold (more predictions)
classify_fish(img_path, threshold=0.3)
```

## Dataset Information

- **Source**: Custom dataset from Philippine fish markets
- **Total images**: ~2,500-3,000 images (Modify the Tilapia folder since we limit test to further push the model, make it more lean to not overfit)
- **Training**: ~70% of dataset
- **Validation**: ~15% of dataset
- **Test**: ~15% of dataset
- **Image format**: JPG
- **Resolution**: Various (resized to 224x224)

### Data Collection
- Images captured in Philippine wet markets
- Various lighting conditions (natural and artificial)
- Different angles and perspectives
- Mix of whole fish and cut fish
- Various freshness levels

## Results Visualization

The notebook includes:
- **Training/Validation curves** - Loss and accuracy over epochs
- **Confusion matrix** - Classification errors
- **Sample predictions** - 15 test images with predictions
- **Class distribution** - Dataset balance visualization
- **Precision-Recall metrics** - Per-class performance

## Advantages of This Approach

### vs Custom CNN
- ✅ **Faster training** (leverages pre-trained weights)
- ✅ **Better accuracy** (benefits from ImageNet knowledge)
- ✅ **Less data needed** (transfer learning is data-efficient)
- ✅ **More robust** (pre-trained on millions of images)

### vs Other Models (VGG, ResNet)
- ✅ **Smaller size** (~9 MB vs 100+ MB)
- ✅ **Faster inference** (optimized for mobile)
- ✅ **Lower compute** (efficient architecture)
- ✅ **Mobile deployment** (TensorFlow Lite compatible)

## Limitations

- 🎯 **Limited to 5 species** - Does not cover all Philippine fish
- 📸 **Quality dependent** - Poor images affect accuracy
- 🐟 **Similar species** - May confuse visually similar fish
- 📊 **Market context** - Trained on market fish (not wild-caught in water)
- 🔄 **Threshold sensitive** - Setting requires balancing false positives/negatives

## Future Improvements

- [ ] Expand to 15-20 common Philippine fish species
- [ ] Deploy as mobile app (Android/iOS with TensorFlow Lite)
- [ ] Add fish freshness detection
- [ ] Include fish size estimation
- [ ] Multi-language support (English, Tagalog, Cebuano, Ilocano)
- [ ] Offline mode for areas with poor connectivity
- [ ] Price estimation based on species and market
- [ ] Integration with e-commerce platforms

## Use Cases

### For Consumers 🛒
- Verify fish species at wet markets
- Avoid mislabeled or substituted fish
- Learn about different fish types
- Make informed purchasing decisions

### For Vendors 🐟
- Quick inventory classification
- Training tool for new staff
- Quality control and authentication
- Pricing assistance

### For Researchers 📊
- Market survey automation
- Species distribution studies
- Price trend analysis
- Biodiversity monitoring

### For Mobile Apps 📱
- E-commerce fish identification
- Recipe recommendation based on fish type
- Nutritional information lookup
- Sustainable fishing guides

## Sample Test Results

```
Test Image: Bangus #1
Predicted: Bangus
Confidence: 0.97

Test Image: Dalagang Bukid #2
Predicted: Dalagang Bukid
Confidence: 0.94

Test Image: Hiwas #3
Predicted: Hiwas
Confidence: 0.89

Test Image: Tilapia #1
Predicted: Tilapia
Confidence: 0.96

Test Image: Tulingan #2
Predicted: Tulingan
Confidence: 0.91
```

## Contributing

Contributions welcome! Areas for improvement:
- More training images (especially underrepresented species)
- Additional fish species
- Mobile app development (Android/iOS)
- Improved data augmentation techniques
- Model optimization for faster inference
- Multi-language documentation

## Technical Details

### Hardware Used
- **Training**: Google Colab with NVIDIA T4 GPU
- **Training time**: ~30-45 minutes (20 epochs)
- **RAM required**: ~12 GB (for data loading)

### Software Stack
- **TensorFlow**: 2.x
- **Keras**: Integrated in TensorFlow 2.x
- **MobileNetV2**: Pre-trained on ImageNet
- **Python**: 3.8+
- **NumPy**: 1.19+
- **Scikit-learn**: 0.24+

## License

MIT License - Free for personal, educational, and commercial use.

## Author

**CloneIsL1fe**
- GitHub: [@CloneIsL1fe](https://github.com/CloneIsL1fe)
- Repository: [Fish_Identifier_Philippines](https://github.com/CloneIsL1fe/Fish_Identifier_Philippines)

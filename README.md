# Skin Lesion Classification

This project is my first deep learning project, focused on classifying dermatoscopic skin lesion images from the HAM10000 dataset into seven diagnostic categories. I built the work in stages: first understanding the data, then training a custom CNN as a baseline, and finally improving performance with transfer learning using EfficientNetV2B0.

The project is notebook-based and is meant to show the full learning workflow, not just the final model. It includes data exploration, preprocessing, model development, evaluation, and saved model artifacts.

## Project Goal

The main goal of this project is to build an image classification pipeline that can predict the type of skin lesion from an input image. Along the way, the project also explores common deep learning challenges such as class imbalance, data preprocessing, validation strategy, and model generalization.

This is an educational project and not a medical tool. The predictions should not be used for clinical decision-making.

## Dataset

The dataset used here is **HAM10000** (Human Against Machine with 10,000 training images), a well-known benchmark dataset for pigmented skin lesion analysis.

### Dataset summary

- Total images: 10,015
- Number of classes: 7
- Classes: `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`
- Metadata file: `0_data/HAM10000_metadata.csv`
- Processed dataset with image paths: `0_data/basedata.csv`

### Class distribution

The dataset is highly imbalanced, with the `nv` class dominating the samples. This had a clear effect on the baseline model and is one of the main reasons class weighting and transfer learning were important in this project.

| Class | Count |
|---|---:|
| `nv` | 6705 |
| `mel` | 1113 |
| `bkl` | 1099 |
| `bcc` | 514 |
| `akiec` | 327 |
| `vasc` | 142 |
| `df` | 115 |

## Project Workflow

The work is organized into three stages:

### 1. Data analysis

Notebook: `1_data anlysis/1.ipynb`

In this stage, I explored the metadata, checked the lesion categories, looked at class imbalance, reviewed patient-related fields like sex and localization, and linked each image ID to its image file path. I also sampled images from different classes to understand the visual differences between lesion types.

### 2. Baseline CNN model

Notebook: `2_base_model/1.ipynb`

For the first model, I built a simple CNN using TensorFlow/Keras with:

- Three convolution + max-pooling blocks
- A dense layer with dropout
- Softmax output for 7-class classification

The images were resized to `224 x 224`, normalized to `[0, 1]`, and loaded using a `tf.data` pipeline.

To make the validation split more reliable, I used `GroupShuffleSplit` with `lesion_id` as the grouping key. This helps reduce leakage by keeping related images from the same lesion from being split across training and validation.

### 3. Transfer learning with EfficientNetV2B0

Notebook: `3_transfer_learning/01.ipynb`

After the baseline, I moved to transfer learning using **EfficientNetV2B0** pretrained on ImageNet. In this version, I added:

- Transfer learning with a pretrained backbone
- Data augmentation
- Class weights to handle imbalance
- Early stopping and learning rate reduction in the training workflow

This model performed much better than the baseline and gave more meaningful predictions across multiple classes.

## Train/Validation Split

The dataset was split using `GroupShuffleSplit`:

- Training samples: 7,991
- Validation samples: 2,024
- Train lesions: 5,976 unique `lesion_id`
- Validation lesions: 1,494 unique `lesion_id`

This split strategy is important for medical image tasks because it helps avoid overly optimistic validation results.

## Results

### Baseline CNN

The custom CNN served as a useful starting point, but it struggled with the class imbalance and did not generalize well.

- Best training accuracy: `58.58%`
- Best validation accuracy: `57.66%`
- Final validation accuracy: `4.89%`

From the classification report, the baseline model performed poorly on most classes and heavily failed to separate minority classes. This made it clear that a stronger feature extractor was needed.

### Transfer learning model

The transfer learning model gave a clear improvement and produced much more reasonable class-wise performance.

- Final training accuracy: `61.54%`
- Final validation accuracy: `60.77%`
- Final validation loss: `1.0183`

Some validation class metrics from the final report:

- `nv`: precision `0.91`, recall `0.66`
- `bkl`: precision `0.43`, recall `0.53`
- `mel`: precision `0.37`, recall `0.37`
- `vasc`: precision `0.71`, recall `0.85`

Overall validation accuracy reached about **61%**, which is a solid improvement for a first deep learning project and shows the benefit of transfer learning over a basic CNN on this dataset.

## Saved Artifacts

The project includes trained model files and training histories:

- `0_models/model.h5` - baseline CNN model
- `0_models/history.pkl` - baseline training history
- `0_models/fine_tune_model.h5` - transfer learning model
- `0_models/fine_tune_history.pkl` - transfer learning history

## Repository Structure

```text
skin lesion/
├── 0_data/
│   ├── HAM10000_metadata.csv
│   ├── basedata.csv
│   ├── ham10000_images_part_1/
│   └── ham10000_images_part_2/
├── 0_models/
│   ├── model.h5
│   ├── history.pkl
│   ├── fine_tune_model.h5
│   └── fine_tune_history.pkl
├── 1_data anlysis/
│   └── 1.ipynb
├── 2_base_model/
│   └── 1.ipynb
└── 3_transfer_learning/
    └── 01.ipynb
```

## Tools and Libraries

This project uses:

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Pillow
- scikit-learn

## How To Run

At the moment, the project is mainly organized around notebooks. A practical way to run it is:

1. Create a Python environment with the required libraries installed.
2. Make sure the HAM10000 images are available inside `0_data/ham10000_images_part_1/` and `0_data/ham10000_images_part_2/`.
3. Open the notebooks in order:
   - `1_data anlysis/1.ipynb`
   - `2_base_model/1.ipynb`
   - `3_transfer_learning/01.ipynb`

If you want to reproduce the project on another machine, one thing to update first is the file paths inside the notebooks, since they currently use absolute local paths.

## What I Learned

This project helped me understand:

- How to work with image datasets and metadata together
- Why validation strategy matters, especially in medical imaging
- How class imbalance can affect model behavior
- The difference between a baseline CNN and a pretrained model
- How transfer learning can improve performance with limited effort

## Future Improvements

There are several ways this project can be improved further:

- Replace absolute paths with relative paths for better portability
- Add a `requirements.txt` or `environment.yml`
- Save label mappings explicitly for inference
- Try fine-tuning more of the EfficientNet backbone
- Add better experiment tracking
- Explore stronger augmentation and sampling strategies
- Build a small inference app for single-image prediction

## Final Note

This project is a meaningful first step into deep learning and medical image classification. The baseline model helped establish the problem, and the transfer learning model showed a clear improvement. More than just the final accuracy, this project reflects the full process of learning how to prepare data, choose a validation strategy, train models, and evaluate results in a practical way.

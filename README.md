# Vietnamese Handwritten Line Text Recognition using CRNN
This project implements a ``Convolutional Recurrent Neural Network (CRNN)`` combined with ``Connectionist Temporal Classification (CTC) Loss`` to recognize lines of Vietnamese handwritten text from images. The model is built using ``PyTorch`` and trained on Vietnamese handwriting line images derived from HANDS-VNOnDB.

## Table of Contents
- Model Architecture
- Dataset
    - Description
    - Preparing training data locally
    - Image Preprocessing
- Installation
- Usage
    - Training
- Improve Performance of pre-trained CRNN Model in Vietnamese Handwritten Text Recognition task
    - Overview
    - The Core Challenge & Solution
    - Setup
    - How to Use

## Model Architecture
The ``CRNN`` is a standard architecture for Vietnamese Handwritten text sequence (text line) recognition tasks in images. It consists of three main components:

### 1. Convolutional Neural Network (CNN) Backbone: 

- **Purpose:** To extract robust visual features from the input image.

- **Structure:** A deep CNN with 7 ``Conv2d`` layers, where the number of channels increases progressively (64 → 128 → 256 → 512 → 1024). These are combined with ``BatchNorm2d``, ``ReLU``, ``MaxPool2d``, and ``Dropout`` for regularization.

- **Function:** This network acts as a feature extractor, transforming the input image into a sequence of feature vectors along the width of the image. The height dimension is significantly reduced through the ``convolutional`` and ``pooling layers``.

### 2. Recurrent Neural Network (RNN) Sequence Encoder:

- **Purpose:** To model the contextual and sequential relationships between the feature vectors extracted by the ``CNN``.

- **Structure:** A 2-layer ``nn.RNN`` with a ``hidden_size`` of 512 is used.

- **Function:** The ``RNN`` processes the feature sequence from the ``CNN``, capturing dependencies between characters in the text line to make context-aware predictions for each step in the sequence.

### 3. Connectionist Temporal Classification (CTC) Loss:

- **Purpose:** To train the network without requiring precise alignment between the input image segments and their corresponding character labels.

- **Function:** ``CTC Loss`` computes the probability of an output sequence by summing over all possible alignments of the RNN's predictions. This is highly effective for sequence recognition tasks where the input and output lengths can vary.

## Dataset
### Description

The line images used for training were derived from HANDS-VNOnDB, maintained by the Nakagawa Laboratory at Tokyo University of Agriculture and Technology (TUAT). Request access through the [official Zenodo record](https://zenodo.org/records/23864366). Complete the dataset usage agreement and wait for approval before accessing the data. The Conditions of Use v2.0 limit use to non-commercial research and prohibit redistribution of the dataset and its derivatives, including rendered images and transcriptions.

- **Vocabulary:** The character set is comprehensive, including all Vietnamese alphabet characters (uppercase, lowercase, and with diacritics), digits, and common punctuation marks.

### Preparing training data locally

After access is approved, render the line-level InkML strokes to images and use the matching ground-truth transcriptions to create UTF-8 annotation files. Preserve the dataset's train and test split. For each image, write one row with its filename (including the extension), a tab, and the transcription. Omit the header.

By default, `Training_CRNN.ipynb` reads this layout from the repository folder.

```text
data/VietOCR/
├── InkData_line_processed/
│   └── <line images>
├── train_line_annotation.txt
└── test_line_annotation.txt
```

To store the dataset outside the checkout, set `OCR_DATA_DIR` to the folder containing both annotation files and `InkData_line_processed` before starting Jupyter.

Keep dataset files, rendered examples, transcriptions, checkpoints trained on the dataset, and outputs containing examples local. Clear outputs from notebooks run with this dataset before committing them. Cite the [HANDS-VNOnDB paper](https://doi.org/10.1016/j.patcog.2018.01.013) when reporting research with the dataset.

### Image Preprocessing
Each input image undergoes a series of preprocessing steps to normalize and enhance its quality before being fed into the model:

- Convert to ``grayscale``.

- Apply ``CLAHE (Contrast Limited Adaptive Histogram Equalization)`` to improve local contrast.

- Use ``Adaptive Thresholding`` to binarize the image.

- Apply morphological operations (``MORPH_OPEN``, ``DILATE``) to remove noise and sharpen text strokes.

- Resize the image to a fixed height (``img_h=64``) and a dynamic width (``img_w = label_len * char_w``) to maintain the aspect ratio of the characters.

- Normalize pixel values to the range ``[0.0, 1.0]``.
## Installation
- **Clone this repository:**
    ```
    git clone <YOUR_REPOSITORY_URL>
    cd <REPOSITORY_NAME>
    ```
- **Create a virtual environment (recommended):**

    ```
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

## Usage
### Training
- Prepare your local dataset as described above, then open and run `Training_CRNN.ipynb`.

- The training script utilizes key techniques such as:

    - ``Adam Optimizer`` with a ``ReduceLROnPlateau`` learning rate scheduler.

    - ``Automatic Mixed Precision (AMP)`` for faster training on compatible GPUs.

    - ``Early Stopping`` (``patience=70``) to prevent overfitting.

- The best performing model checkpoint will be saved to ``./trained_model/best_model_line.pth``.

## Improve Performance of pre-trained CRNN Model in Vietnamese Handwritten Text Recognition task

This section explains the process used in the ``Testing.ipynb`` notebook to improve performance of the ``pre-trained CRNN model``.

### Overview

- The purpose of this section is to provide a practical demonstration of how to use the ``trained CRNN model`` for real-world prediction tasks. Unlike the training script, which relies on a dataset with known labels, this inference script is engineered to handle single, unlabeled images.

- The key innovation in this notebook is a heuristic-based method for estimating the number of characters in an image, which is a crucial step for preparing the image for my specific ``CRNN model``.

### The Core Challenge & Solution
- The Challenge: The ``CRNN model`` was trained on images that were resized to a dynamic width proportional to the length of their corresponding text label (i.e., ``image_width = character_count * character_width``). During inference on a new image, the ground-truth text and its length are unknown. Passing an improperly sized image to the model would lead to poor performance.
- The Solution: To overcome this, a character estimation function, ``estimate_character_count_with_spaces``, has been implemented. This function analyzes the input image using image processing techniques to approximate the number of characters (including spaces). This estimated count is then used to resize the image to an appropriate width that the ``CRNN model`` expects, significantly improving prediction accuracy.

### Setup
- **Clone this repository:** Skip this if you cloned
- **Create a virtual environment (recommended):** Skip this if you created
- **Place the Pre-trained Model:**
Ensure your trained model file (``best_model_line.pth`` from the training notebook) is located in the ``./trained_model/`` directory.

### How to Use
- Place the image file you want to test inside a directory or you can use my own self-handwritten dataset in this repository (``./data/Self-handwritten/``).

- Open the ``Testing.ipynb`` notebook and update the ``image_path`` variable to point to your image file.

- Run all cells in the notebook.

- The output will display the estimated character count, the final predicted text, and the input image.



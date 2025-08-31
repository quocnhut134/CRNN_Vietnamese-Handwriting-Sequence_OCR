# Vietnamese Handwritten Line Text Recognition using CRNN
This project implements a ``Convolutional Recurrent Neural Network (CRNN)`` combined with ``Connectionist Temporal Classification (CTC) Loss`` to recognize lines of Vietnamese handwritten text from images. The model is built using ``PyTorch`` and trained on a custom ``Vietnamese handwriting dataset``.

## Table of Contents
- Model Architecture
- Dataset
    - Description
    - Image Preprocessing
- Installation
- Usage
    - Training

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

***Note:** You may need to adjust the data paths in the Jupyter Notebook if your structure differs.*

## Usage
### Training
- Open and run the cells in the Jupyter notebook file.

- The training script utilizes key techniques such as:

    - ``Adam Optimizer`` with a ``ReduceLROnPlateau`` learning rate scheduler.

    - ``Automatic Mixed Precision (AMP)`` for faster training on compatible GPUs.

    - ``Early Stopping`` (``patience=70``) to prevent overfitting.

- The best performing model checkpoint will be saved to ``./trained_model/best_model_line.pth``.



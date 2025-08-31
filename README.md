# Improve Performance of CRNN Model in Vietnamese Handwritten Text Recognition task
This document explains the process used in the ``Testing.ipynb`` notebook to improve performance of the ``pre-trained CRNN model``.

## Table of Contents
- Overview

- The Core Challenge & Solution

- Setup & Installation

- How to Use

## Overview

The purpose of this notebook is to provide a practical demonstration of how to use the ``trained CRNN model`` for real-world prediction tasks. Unlike the training script, which relies on a dataset with known labels, this inference script is engineered to handle single, unlabeled images.

The key innovation in this notebook is a heuristic-based method for estimating the number of characters in an image, which is a crucial step for preparing the image for my specific ``CRNN model``.

## The Core Challenge & Solution
- The Challenge: The ``CRNN model`` was trained on images that were resized to a dynamic width proportional to the length of their corresponding text label (i.e., ``image_width = character_count * character_width``). During inference on a new image, the ground-truth text and its length are unknown. Passing an improperly sized image to the model would lead to poor performance.
- The Solution: To overcome this, a character estimation function, ``estimate_character_count_with_spaces``, has been implemented. This function analyzes the input image using image processing techniques to approximate the number of characters (including spaces). This estimated count is then used to resize the image to an appropriate width that the ``CRNN model`` expects, significantly improving prediction accuracy.

## Setup & Installation
- **Clone the repository:**

    ```
    git clone <YOUR_REPOSITORY_URL>
    cd <REPOSITORY_NAME>
    ```
- **Create a virtual environment (Recommended):**
    ```
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
- **Place the Pre-trained Model:**
Ensure your trained model file (``best_model_line.pth`` from the training notebook) is located in the ``./trained_model/`` directory.

## How to Use
- Place the image file you want to test inside a directory or you can use my own self-handwritten dataset in this repository (``./data/Self-handwritten/``).

- Open the ``Testing.ipynb`` notebook and update the ``image_path`` variable to point to your image file.

- Run all cells in the notebook.

- The output will display the estimated character count, the final predicted text, and the input image.

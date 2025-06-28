# 🚧 Hate Speech Detection Model 🚧

## Overview

Welcome to the Hate Speech Detection Model repository! This project leverages state-of-the-art Natural Language Processing (NLP) and Machine Learning (ML) techniques to identify and classify hate speech in text, specifically focusing on social media content like tweets. Our goal is to contribute to safer online environments by developing a robust, efficient, and fair hate speech detection system.

The model is built upon the powerful DistilBERT architecture, fine-tuned for the specific task of hate speech classification. We've incorporated advanced optimization techniques such as Quantization-Aware Training (QAT) and pruning to create a highly efficient model suitable for deployment in resource-constrained environments. Furthermore, we emphasize fairness in our model, utilizing AIF360 to assess and mitigate biases across different sensitive attributes like gender and race-related topics.

This README provides a comprehensive guide to understanding the project's structure, the underlying concepts of hate speech detection, the machine learning methodologies employed, and detailed explanations of the codebase and its various components.

## 🎯 Project Goals

* **Accurate Hate Speech Classification:** Develop a model that can accurately distinguish between normal, offensive, and hate speech content.
* **Performance Optimization:** Implement techniques like quantization and pruning to reduce model size and inference time without significant performance degradation.
* **Fairness and Bias Mitigation:** Evaluate and address potential biases in the model's predictions concerning different demographic groups or sensitive topics.
* **Deployment Readiness:** Export the optimized model to ONNX format for efficient deployment across various platforms.

## 🧐 What is Hate Speech?

Hate speech is generally defined as any form of expression that attacks a person or group on the basis of attributes such as race, ethnicity, national origin, religion, sex, disability, or sexual orientation. It often aims to degrade, intimidate, or incite violence or discrimination against specific groups.

In the context of this project, we categorize text into three labels:

* **Normal (0):** Content that is benign and does not contain any offensive or hateful language.
* **Offensive (1):** Content that is aggressive, insulting, or profane but may not directly target a protected attribute or incite hatred.
* **Hate (2):** Content that explicitly promotes hatred, discrimination, or violence against a protected group.

Understanding these distinctions is crucial for building an effective and nuanced hate speech detection system.

## 🤖 Machine Learning in Hate Speech Detection

Machine learning plays a pivotal role in automating the identification of hate speech. Instead of relying on manual rule-based systems, which are often rigid and difficult to scale, ML models can learn complex patterns and relationships within vast amounts of text data.

Here's how ML contributes to this project:

* **Natural Language Processing (NLP):** At its core, hate speech detection is an NLP task. Models are trained to understand the semantics, syntax, and context of human language.
* **Feature Extraction:** Text data is transformed into numerical representations (embeddings) that ML models can process.
* **Classification:** The model learns to classify these numerical representations into predefined categories (Normal, Offensive, Hate).
* **Deep Learning (DistilBERT):** We utilize DistilBERT, a powerful transformer-based deep learning model, known for its ability to capture intricate linguistic nuances and contextual relationships, making it highly effective for text classification tasks.

### ⚡ Model Optimization: Quantization & Pruning

To ensure our hate speech detection model is not only accurate but also highly efficient, especially for deployment on devices with limited computational resources, we employ two powerful optimization techniques: **Quantization** and  **Pruning** .

#### What is Quantization?

Quantization is a technique that reduces the precision of the numbers used to represent a neural network's weights and activations. Most deep learning models are trained using 32-bit floating-point numbers (FP32). Quantization typically converts these to lower-precision formats, such as 8-bit integers (INT8).

* **Why use it?**
  * **Reduced Model Size:** Storing numbers with fewer bits requires less memory. An INT8 model can be 4x smaller than its FP32 counterpart.
  * **Faster Inference:** Operations on lower-precision integers are computationally less intensive, leading to faster execution times and lower power consumption. This is particularly beneficial for deploying models on mobile devices or edge computing platforms.
* **Quantization-Aware Training (QAT):** Instead of quantizing a model after it's fully trained (post-training quantization), QAT performs quantization *during* the training process. This allows the model to "learn" to be robust to the effects of quantization, often leading to higher accuracy compared to post-training methods. In our code, `torch.quantization.prepare_qat(model, inplace=True)` sets up the model for this process.

#### What is Pruning?

Pruning is a technique that removes "unnecessary" connections (weights) in a neural network. It's based on the idea that many weights in a trained neural network contribute very little to its final output, and removing them can reduce the model's complexity without significantly impacting its performance.

* **Why use it?**
  * **Reduced Model Size:** Removing weights leads to a sparser network, requiring less memory to store.
  * **Faster Inference:** While not always directly speeding up computation on standard hardware (due to sparse matrix operations), it can enable specialized hardware or frameworks to achieve speedups. More importantly, it complements quantization by further reducing the model footprint.
* **How it works (in our code):** `prune.random_unstructured(module, name="weight", amount=0.5)` randomly removes 50% of the weights from specified layers. `prune.remove(module, 'weight')` then permanently removes these pruned weights, making the model truly smaller.

### ⚖️ Fairness Metrics with AIF360

In developing AI systems, it's crucial to ensure they are fair and do not perpetuate or amplify existing societal biases. Hate speech detection models, in particular, must be carefully evaluated for fairness, as they can disproportionately affect certain groups.

* **AIF360:** We use IBM's AI Fairness 360 (AIF360) toolkit to assess the fairness of our model. This toolkit provides a comprehensive set of metrics for measuring bias and algorithms for mitigating it.
* **Protected Attributes:** In our evaluation, we define 'gender' and 'race' as protected attributes by categorizing tweets based on their `TopicID`. This allows us to examine if the model performs differently for tweets related to these sensitive topics.
* **Key Metrics Evaluated:**
  * **Statistical Parity Difference:** Measures the difference in the proportion of favorable outcomes (e.g., correctly classified as 'Normal' or 'Non-Hate') between unprivileged and privileged groups. An ideal value is 0.
  * **Disparate Impact:** The ratio of the proportion of favorable outcomes for the unprivileged group to that of the privileged group. An ideal value is 1.
  * **Average Odds Difference:** The average of the difference in False Positive Rates and True Positive Rates between unprivileged and privileged groups. An ideal value is 0.

By analyzing these metrics, we gain insights into potential biases and can work towards creating a more equitable model.

## 📂 Repository Structure

This repository is organized to provide a clear and logical flow for understanding and interacting with the Hate Speech Detection Model. Folders are listed in order of their typical importance in the development and understanding workflow.

* `hate-speech-ml.py`
  * **Purpose:** This is the heart of the project. It contains the complete Python script for the entire machine learning pipeline. This includes:
    * Data loading and preprocessing.
    * Dataset and DataLoader setup using DistilBERT tokenizer.
    * Model initialization with DistilBERT for Sequence Classification.
    * **Quantization-Aware Training (QAT)** preparation and execution.
    * Training and evaluation loops.
    * Fairness metric calculation using AIF360.
    * **Post-training optimization (Pruning)** .
    * Conversion of the optimized model to **ONNX** format for deployment.
    * Verification and prediction using the ONNX model.
  * **Importance:** This is the executable code that brings all concepts to life, demonstrating the model's training, optimization, and evaluation.
* `processed-data/`
  * **Purpose:** This folder stores the dataset that has undergone thorough preprocessing using AI-Data engineering and data analysis techniques. This means the raw data has been cleaned, transformed, and made ready for direct consumption by the machine learning model.
  * **Importance:** Contains the clean, ready-to-use data that feeds directly into the `hate-speech-ml.py` script for training and evaluation. It represents the crucial step of preparing high-quality input for the model.
* `docs/`
  * **Purpose:** This folder serves as the documentation hub for the project. It contains three important files:
    * **Progressive Roadmap:** Outlines the development approach taken, divided into 5 blocks, providing insight into the project's evolution.
    * **Deprecated Model Evaluation Metric Report:** Contains the performance metrics and analysis for older, no-longer-used model versions.
    * **Current Model Evaluation Metric Report:** Provides the latest performance metrics and detailed analysis of the current, optimized hate speech detection model.
  * **Importance:** Provides critical insights into the project's planning, development history, and current performance, essential for tracking progress and understanding model effectiveness.
* `data-preprocessing/`
  * **Purpose:** This folder houses the scripts responsible for preparing the raw data. This includes:
    * **AI Engineering/Analysis File:** Scripts for advanced data cleaning, feature engineering, and exploratory data analysis.
    * **Data Scrapper File:** Code used to collect raw data from its source (e.g., social media APIs).
  * **Importance:** These scripts are vital for transforming raw, unorganized data into the clean, structured format found in `processed-data/`, which is then used by the main model.
* `raw-data/`
  * **Purpose:** This folder contains the original, untouched dataset as it was initially collected. It serves as an archival copy of the data before any preprocessing or cleaning steps were applied.
  * **Importance:** Acts as the foundational source for all data processing. Keeping raw data separate ensures reproducibility and allows for re-evaluation of preprocessing steps if needed.
* `deprecated-model/`
  * **Purpose:** This folder specifically contains the previous, deprecated version of the hate speech detection model, along with the metric scoring code used for that model.
  * **Importance:** Provides a historical record of past model iterations, useful for comparison, understanding improvements, and referencing previous approaches.

## 🛠️ Technologies Used

The development of this project relies on a robust stack of Python libraries and frameworks:

* **Python:** The primary programming language.
* **PyTorch:** A leading open-source machine learning framework, extensively used for building and training neural networks.
* **Hugging Face Transformers:** Provides the pre-trained DistilBERT model and tokenizer, simplifying the implementation of transformer-based architectures.
* **pandas:** Essential for data manipulation and analysis, especially for handling tabular data.
* **numpy:** Fundamental package for numerical computation in Python, widely used for array operations.
* **scikit-learn:** Provides a wide range of machine learning algorithms and utilities, including `train_test_split` and `classification_report` for model evaluation.
* **AIF360:** IBM's AI Fairness 360 toolkit for measuring and mitigating bias in machine learning models.
* **ONNX Runtime:** An open-source inference engine for ONNX models, enabling efficient cross-platform deployment.
* **re (Regular Expression):** Used for advanced text pattern matching and cleaning.
* **emoji:** Library for handling and demojizing emoji characters in text.
* **tqdm:** Provides a fast, extensible progress bar for loops, enhancing user experience during training and evaluation.
* **os:** Python's standard library for interacting with the operating system, used for file and directory operations.

## 🚀 Getting Started

To get a copy of the project up and running on your local machine for development and testing purposes, follow these steps:

### Prerequisites

Ensure you have Python 3.8+ installed.

### Installation

1. **Clone the repository:**

   ```
   git clone https://github.com/your-username/hate-speech-detection.git
   cd hate-speech-detection

   ```
2. **Create a virtual environment (recommended):**

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: `venv\Scripts\activate`

   ```
3. **Install dependencies:**

   ```
   pip install -r requirements.txt

   ```

   *(Note: You might need to create a `requirements.txt` file by running `pip freeze > requirements.txt` after installing the listed libraries manually, or populate it based on the technologies section.)*

### Running the Model

After installation, you can run the main script:

```
python hate-speech-ml.py

```

This script will:

1. Load the dataset (or generate dummy data if `processed_dataset.csv` is not found).
2. Preprocess the text data.
3. Set up the DistilBERT model and prepare it for Quantization-Aware Training.
4. Train the model.
5. Evaluate its performance using a classification report and fairness metrics (Statistical Parity Difference, Disparate Impact, Average Odds Difference) for gender and race-related topics.
6. Apply pruning and finalize quantization.
7. Export the optimized model to `hate_speech_model_quantized_pruned.onnx`.
8. Verify the ONNX model and demonstrate a prediction on a sample tweet.

## 👋 Contributing

We welcome contributions to this project! If you have suggestions for improvements, new features, or bug fixes, please feel free to:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature/YourFeature`).
6. Open a Pull Request.

Please ensure your code adheres to the existing style and includes relevant tests if applicable.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://mit-license.org/) file for details.

Made with ❤️ by [Marshallcodes](https://github.com/marshallcodes), [Praise Oladejo](https://github.com/luwanise), [Oritsemisan Omare](https://github.com/MisanOmar3)/REACHVERSE

# PTSD Emotion Recognition System

## Overview

This project leverages a *Transformer-based architecture* for *emotion recognition* from speech. The system uses the *Wav2Vec2 model* for *speech feature extraction, followed by a **Transformer-based Emotion Classifier. The system is designed to classify emotions in speech, specifically for **Post-Traumatic Stress Disorder (PTSD)* detection.

---

## Features

### Wav2Vec2 Feature Extraction

- *Wav2Vec2 Model*: A *pretrained model* used for speech feature extraction.
- *Input: Raw **WAV files* containing emotional speech.
- *Output*: 768D feature vectors representing:
  - *Tone*: Captures variation in pitch and melody.
  - *Frequency*: Represents speech's tonal characteristics.
  - *Wavelength*: Encodes speech duration and rhythm.
  - *Stress*: Captures emphasis and stress in speech.

Wav2Vec2 processes the speech to extract meaningful features for emotion recognition.

---

### Transformer-based Emotion Classifier

The *Transformer-based model* utilizes features from Wav2Vec2 and performs emotion classification using the following architecture:
  
- *Feature Projection*: A linear layer projects the 768D features into a larger hidden space.
- *Transformer Encoder*: A multi-layer self-attention mechanism captures complex relationships in the speech data.
- *Feed-Forward Network (FNN): Multi-layer fully connected network (FNN) with **ReLU* activation.
- *Softmax Output*: Softmax activation provides probabilities for each emotion class.

---

### Model Enhancements and Regularization

Key enhancements have been made to the Transformer-based architecture:

- *Larger Transformer*: More layers and attention heads for capturing complex speech patterns.
- *Feature Projection*: Projects the Wav2Vec2 features to a larger hidden space to increase capacity.
- *Larger Feedforward Networks*: Helps capture richer representations of input data.

---

Regularization techniques applied:
- *Dropout*: Prevents overfitting by randomly dropping units during training.
- *Batch Normalization*: Normalizes activations to speed up training and improve generalization.
- *Weight Initialization: Uses **Xavier uniform initialization* for better convergence.

---

### RAVDESS Dataset

The *RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)* dataset is used for training and testing the model:
- *8 distinct emotions*: Calm, Angry, Sad, Happy, Disgust, Fearful, Surprised, Neutral.
- *24 Actors*: Includes 24 different actors performing these emotions.
- *1,440 WAV Files*: Each file contains a speech recording with an emotional label.

---

#### Dataset Split:
- *Training*: 80% (1152 files).
- *Testing*: 20% (288 files).

---

### Model Training and Evaluation

The model is trained using the following setup:

- *Batch Processing*: Mini-batches (32 samples per batch).
- *Optimizer: **Adam optimizer* used for weight updates.
- *Loss Function: **Cross-Entropy Loss* for multi-class classification.
- *Epochs: Trained for **10 epochs* with shuffling at the start of each epoch.

*Evaluation Metrics*:
- *Accuracy*: 70.8% (73.6% during validation).
- *Balanced Accuracy*: 70.0%.
- *F1-Score*: 70.0%.
- *Average Confidence*: 95.2%.

---

#### Confusion Matrix & Confidence Score:
A confusion matrix is used to show the performance of the classifier across different emotions.
Confidence Score is calculated based on Probabiity-Weights, Distribution and Certainity
Risk Score is evaulated on Hyper-Arousal, Emotional Numbing, Negative Effect and Vocal Indicators.

---

## 📸 Screenshots

Screenshots are available in the `/screenshots` folder.  
Example:

![Img1](screenshots/Screenshot%202025-10-14%20151006.png)
![Img2](screenshots/Screenshot%202025-10-14%20151022.png)
![Img3](screenshots/Screenshot%202025-10-14%20151034.png)
![Img4](screenshots/Screenshot%202025-10-14%20151045.png)
![Img5](screenshots/Screenshot%202025-10-14%20151133.png)


---


## Acknowledgments
- Wav2Vec2: Hugging Face and Facebook AI Research.
- RAVDESS dataset: Ryerson University.


   ```bash
   streamlit app.py

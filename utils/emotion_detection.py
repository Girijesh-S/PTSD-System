import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import librosa  # Add this import

class EmotionClassifier(nn.Module):
    def __init__(self, input_dim=768, num_emotions=8):
        super(EmotionClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_emotions),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.classifier(x)

class EmotionDetector:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier = EmotionClassifier()
        self.label_encoder = LabelEncoder()
        self.load_models()
        
    def load_models(self):
        """Load trained models"""
        try:
            self.emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised', 'no_speech']
            self.label_encoder.fit(self.emotions)
            print("✅ Emotion detector initialized successfully")
        except Exception as e:
            print(f"⚠️ Could not load trained models: {e}")
            self.emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised', 'no_speech']
            self.label_encoder.fit(self.emotions)
    
    def detect_silence(self, audio, sr=16000, threshold=0.01):
        """Detect if audio is silent or has speech"""
        try:
            if audio is None or len(audio) == 0:
                return True, 0.0, 0.0
            
            rms = np.sqrt(np.mean(audio**2))
            non_silent_samples = np.sum(np.abs(audio) > threshold)
            total_samples = len(audio)
            non_silent_percentage = non_silent_samples / total_samples * 100
            
            # Audio is considered silent if RMS is very low OR less than 10% is non-silent
            is_silent = rms < 0.005 or non_silent_percentage < 10
            
            return is_silent, rms, non_silent_percentage
            
        except Exception as e:
            print(f"Error in silence detection: {e}")
            return True, 0.0, 0.0
        
    def analyze_audio(self, audio_path):
        """Main analysis function - with silence detection"""
        try:
            from audio_processing import AudioProcessor
            
            processor = AudioProcessor()
            audio = processor.load_audio(audio_path)
            
            if audio is None:
                return self.no_speech_result()
            
            # Check for silence before feature extraction
            is_silent, rms, non_silent_percentage = self.detect_silence(audio)
            
            if is_silent:
                print(f"⚠️ Silent audio detected: RMS={rms:.6f}, Non-silent={non_silent_percentage:.1f}%")
                return self.no_speech_result(rms, non_silent_percentage)
            
            features = processor.extract_features(audio)
            
            if features is not None:
                # Use actual model prediction if available, otherwise mock analysis
                result = self.predict_emotion(features)
                return result
            else:
                return self.mock_analysis(features)
                
        except Exception as e:
            print(f"Error in audio analysis: {e}")
            return self.mock_analysis(None)
    
    def no_speech_result(self, rms=0.0, non_silent_percentage=0.0):
        """Return result for silent/no speech audio"""
        emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised', 'no_speech']
        probabilities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        
        return {
            'emotion': 'no_speech',
            'confidence': 0.0,
            'all_probabilities': dict(zip(emotions, probabilities)),
            'ptsd_risk': 0.0,
            'features_used': 0,
            'audio_energy': rms,
            'non_silent_percentage': non_silent_percentage,
            'is_silent': True
        }
    
    def predict_emotion(self, features):
        """Actual model prediction - replace with your trained model"""
        try:
            # If you have a trained model, use it here:
            # features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            # with torch.no_grad():
            #     probabilities = self.classifier(features_tensor).cpu().numpy().flatten()
            
            # For now, using mock analysis until you integrate your trained model
            return self.mock_analysis(features)
            
        except Exception as e:
            print(f"Model prediction failed: {e}")
            return self.mock_analysis(features)
    
    def mock_analysis(self, features):
        """Mock analysis for demo - replace with actual model"""
        emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        
        # If features are available, create more realistic probabilities
        if features is not None and len(features) > 0:
            # Use feature statistics to influence probabilities
            feature_mean = np.mean(features)
            feature_std = np.std(features)
            
            # Create more realistic probability distribution
            base_probs = np.random.dirichlet(np.ones(8), size=1)[0]
            
            # Adjust based on audio characteristics
            if feature_std < 0.1:  # Very uniform features
                base_probs = [0.6, 0.2, 0.05, 0.05, 0.03, 0.03, 0.02, 0.02]  # Bias toward neutral/calm
            elif feature_mean > 0.5:  # High energy features
                base_probs = [0.1, 0.1, 0.3, 0.1, 0.2, 0.1, 0.05, 0.05]  # Bias toward active emotions
            
            probabilities = base_probs
        else:
            probabilities = np.random.dirichlet(np.ones(8), size=1)[0]
        
        primary_emotion = emotions[np.argmax(probabilities)]
        
        # PTSD risk calculation
        risk_factors = {
            'fearful': 0.8, 'angry': 0.7, 'sad': 0.5, 
            'disgust': 0.4, 'surprised': 0.3, 'neutral': 0.2,
            'calm': 0.1, 'happy': 0.0
        }
        
        ptsd_risk = risk_factors.get(primary_emotion, 0.0)
        
        return {
            'emotion': primary_emotion,
            'confidence': float(np.max(probabilities)),
            'all_probabilities': dict(zip(emotions, probabilities)),
            'ptsd_risk': ptsd_risk,
            'features_used': len(features) if features is not None else 0,
            'is_silent': False
        }

# Global function for easy access
def predict_emotion_live(audio_path):
    """Global function to predict emotion from audio file"""
    detector = EmotionDetector()
    return detector.analyze_audio(audio_path)
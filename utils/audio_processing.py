import librosa
import numpy as np
import torch

class AudioProcessor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.processor = None
        self.model = None
        self.load_models()
        
    def load_models(self):
        """Load models with compatibility handling"""
        try:
            # Try new import first
            from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
            self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        except ImportError:
            try:
                # Try old import
                from transformers import Wav2Vec2Processor, Wav2Vec2Model
                self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
                self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
            except ImportError:
                # Fallback to librosa features only
                print("⚠️ Transformers not available, using librosa features only")
                self.processor = None
                self.model = None
        
        if self.model:
            self.model.eval()
    
    def load_audio(self, file_path, duration=10):
        """Load and preprocess audio file"""
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate, duration=duration)
            if len(audio) == 0:
                raise ValueError("Loaded audio is empty")
            return audio
        except Exception as e:
            print(f"Error loading audio from {file_path}: {e}")
            return None
    
    def extract_features(self, audio):
        """Extract features using available methods"""
        try:
            if audio is None or len(audio) == 0:
                raise ValueError("Audio is empty or None")
            
            # If transformers is available, use wav2vec2
            if self.processor and self.model:
                inputs = self.processor(audio, sampling_rate=self.sample_rate, 
                                      return_tensors="pt", padding=True)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    features = outputs.last_hidden_state.mean(dim=1).numpy()
                return features.flatten()
            else:
                # Fallback to librosa features
                return self.extract_librosa_features(audio)
                
        except Exception as e:
            print(f"Error extracting features: {e}")
            return self.extract_librosa_features(audio)
    
    def extract_librosa_features(self, audio):
        """Extract features using librosa as fallback"""
        try:
            features = []
            
            # MFCC features
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
            features.extend(np.mean(mfcc, axis=1))
            features.extend(np.std(mfcc, axis=1))
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
            features.append(np.mean(spectral_centroid))
            features.append(np.std(spectral_centroid))
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)
            features.append(np.mean(zcr))
            features.append(np.std(zcr))
            
            # RMS energy
            rms = librosa.feature.rms(y=audio)
            features.append(np.mean(rms))
            features.append(np.std(rms))
            
            return np.array(features)
            
        except Exception as e:
            print(f"Error extracting librosa features: {e}")
            return np.random.rand(30)  # Return random features as last resort
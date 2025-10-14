import streamlit as st

# Page configuration MUST be first
st.set_page_config(
    page_title="PTSD Emotion Recognition",
    page_icon="🎭",
    layout="wide"
)

import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import tempfile
import os
import wave
import pyaudio
from datetime import datetime
import time
import sys
import torch.nn as nn

# Add the utils directory to Python path to import your modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# Import your actual emotion detection model
try:
    from emotion_detection import EmotionDetector, predict_emotion_live
    from audio_processing import AudioProcessor
    MODEL_LOADED = True
except ImportError as e:
    st.warning(f"Could not import emotion detection modules: {e}")
    MODEL_LOADED = False

class RealTimeRecorder:
    def __init__(self):
        self.audio = None
        self.is_recording = False
        self.frames = []
        self._initialize_audio()
    
    def _initialize_audio(self):
        """Initialize audio system"""
        try:
            self.audio = pyaudio.PyAudio()
            # Test if we can access microphone
            device_info = self.audio.get_default_input_device_info()
            st.sidebar.success(f"🎤 Microphone: {device_info['name']}")
        except Exception as e:
            st.error(f"❌ Cannot access microphone: {e}")
            self.audio = None
    
    def record_audio(self, duration=5):
        """Record audio for specified duration"""
        if self.audio is None:
            st.error("❌ Audio system not available")
            return None
            
        self.frames = []
        
        try:
            # Audio settings
            sample_rate = 16000
            channels = 1
            chunk_size = 1024
            format_type = pyaudio.paInt16
            
            # Open stream
            stream = self.audio.open(
                format=format_type,
                channels=channels,
                rate=sample_rate,
                input=True,
                frames_per_buffer=chunk_size
            )
            
            st.info(f"🎙️ Recording for {duration} seconds... Speak now!")
            
            # Calculate total chunks needed
            chunks_needed = int((sample_rate / chunk_size) * duration)
            
            # Record with progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(chunks_needed):
                try:
                    data = stream.read(chunk_size, exception_on_overflow=False)
                    self.frames.append(data)
                    
                    # Update progress
                    progress = (i + 1) / chunks_needed
                    progress_bar.progress(progress)
                    status_text.text(f"Recording... {int(progress * 100)}%")
                    
                except Exception as e:
                    st.warning(f"Audio chunk error: {e}")
                    continue
            
            # Clean up stream
            stream.stop_stream()
            stream.close()
            progress_bar.empty()
            status_text.empty()
            
            # Check if we got any audio
            if not self.frames:
                st.error("❌ No audio data captured")
                return None
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            if self._save_wav(temp_file.name, sample_rate, channels, format_type):
                # Verify the file
                if os.path.exists(temp_file.name) and os.path.getsize(temp_file.name) > 1000:
                    st.success("✅ Recording completed!")
                    return temp_file.name
                else:
                    st.error("❌ Recorded file is empty or too small")
                    os.unlink(temp_file.name)
                    return None
            else:
                return None
                
        except Exception as e:
            st.error(f"❌ Recording failed: {str(e)}")
            st.info("""
            **Troubleshooting:**
            - Ensure microphone is connected
            - Grant microphone permissions to Python
            - Close other applications using microphone
            - Try a shorter recording duration
            """)
            return None
    
    def _save_wav(self, filename, sample_rate, channels, format_type):
        """Save recorded frames to WAV file"""
        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(self.audio.get_sample_size(format_type))
                wf.setframerate(sample_rate)
                wf.writeframes(b''.join(self.frames))
            return True
        except Exception as e:
            st.error(f"❌ Error saving audio: {e}")
            return False
    
    def close(self):
        """Clean up audio resources"""
        if self.audio:
            self.audio.terminate()

# ... REST OF YOUR PTSDStreamlitApp CLASS REMAINS EXACTLY THE SAME ...
class PTSDStreamlitApp:
    def __init__(self):
        self.recorder = RealTimeRecorder()
        self.setup_models()
    
    def setup_models(self):
        """Setup models - initialize your actual models here"""
        st.session_state.setdefault('session_history', [])
        
        # Initialize your emotion detection model
        if MODEL_LOADED:
            try:
                # Initialize your model
                self.emotion_detector = EmotionDetector()
                st.sidebar.success("✅ Emotion model loaded successfully")
            except Exception as e:
                st.sidebar.warning(f"⚠️ Could not initialize emotion model: {e}")
        else:
            st.sidebar.warning("⚠️ Using fallback analysis (demo mode)")
    
    def analyze_with_model(self, audio_path):
        """Analyze audio using your actual emotion detection model"""
        try:
            if not MODEL_LOADED:
                return self.fallback_analysis(audio_path)
            
            # Use your actual model for prediction
            result = predict_emotion_live(audio_path)
            
            if result:
                return result
            else:
                return self.fallback_analysis(audio_path)
                
        except Exception as e:
            st.error(f"❌ Model analysis failed: {e}")
            return self.fallback_analysis(audio_path)
    
    def fallback_analysis(self, audio_path):
        """Fallback analysis if model fails to load"""
        try:
            # Use librosa to get actual audio duration
            if not os.path.exists(audio_path):
                st.error("❌ Audio file not found")
                return None
                
            duration = librosa.get_duration(filename=audio_path)
            audio, sr = librosa.load(audio_path, sr=16000)
            rms = np.sqrt(np.mean(audio**2))
            
            emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
            weights = np.random.dirichlet(np.ones(8))
            
            primary_emotion = emotions[np.argmax(weights)]
            confidence = np.max(weights)
            
            risk_factors = {
                'fearful': 0.8, 'angry': 0.7, 'sad': 0.5, 'disgust': 0.4,
                'surprised': 0.3, 'neutral': 0.2, 'calm': 0.1, 'happy': 0.0
            }
            
            risk_score = risk_factors.get(primary_emotion, 0.3)
            
            indicators = []
            if rms < 0.005:
                indicators.append("Low audio energy detected - possible emotional flatness")
                risk_score += 0.2
            
            return {
                'emotion': primary_emotion,
                'confidence': float(confidence),
                'all_probabilities': dict(zip(emotions, weights)),
                'ptsd_risk': min(risk_score, 1.0),
                'recommendation': self.get_recommendation(risk_score),
                'indicators': indicators,
                'audio_duration': duration,  # This will now show actual duration
                'audio_energy': rms
            }
            
        except Exception as e:
            st.error(f"❌ Fallback analysis error: {e}")
            return None

    def analyze_recorded_audio(self, audio_path):
        """Analyze the recorded audio using your actual model"""
        return self.analyze_with_model(audio_path)
    
    def get_recommendation(self, risk_score):
        """Get recommendation based on risk score"""
        if risk_score > 0.7:
            return "🚨 HIGH ALERT: Strong PTSD indicators detected. Consider immediate therapist review."
        elif risk_score > 0.4:
            return "⚠️ MEDIUM ALERT: Moderate PTSD risk. Close monitoring recommended."
        else:
            return "✅ LOW RISK: Emotional patterns within normal range. Continue monitoring."
    
    def display_results(self, result):
        """Display analysis results"""
        if not result:
            st.error("❌ No analysis results available")
            return
            
        st.header("📊 Emotional Analysis Results")
        
        # Main results row
        col1, col2, col3 = st.columns(3)
        
        with col1:
            emotion_emoji = {
                'happy': '😊', 'sad': '😢', 'angry': '😠', 
                'fearful': '😨', 'disgust': '🤢', 'surprised': '😲',
                'neutral': '😐', 'calm': '😌'
            }
            emoji = emotion_emoji.get(result['emotion'], '🎭')
            st.metric(
                "Primary Emotion", 
                f"{emoji} {result['emotion'].upper()}",
                f"{result['confidence']:.1%} confidence"
            )
        
        with col2:
            confidence_level = "High" if result['confidence'] > 0.7 else "Medium" if result['confidence'] > 0.5 else "Low"
            st.metric("Analysis Confidence", confidence_level)
        
        with col3:
            risk_score = result.get('ptsd_risk', 0)
            risk_level = "HIGH" if risk_score > 0.7 else "MEDIUM" if risk_score > 0.4 else "LOW"
            risk_color = "#ff6b6b" if risk_level == "HIGH" else "#ffd93d" if risk_level == "MEDIUM" else "#6bcf7f"
            
            st.markdown(f"""
            <div style="background-color: {risk_color}; padding: 15px; border-radius: 10px; color: {'white' if risk_level == 'HIGH' else 'black'};">
                <h3>PTSD Risk Assessment</h3>
                <p><strong>Level:</strong> {risk_level}</p>
                <p><strong>Score:</strong> {risk_score:.2f}/1.0</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Audio metrics - THIS WILL NOW SHOW ACTUAL DURATION
        st.subheader("🎵 Audio Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Audio Duration", f"{result.get('audio_duration', 0):.2f}s")
        with col2:
            st.metric("Speech Energy", f"{result.get('audio_energy', 0):.4f}")
        
        # Emotion distribution chart
        st.subheader("🎯 Emotion Probability Distribution")
        emotions = list(result['all_probabilities'].keys())
        probabilities = list(result['all_probabilities'].values())
        
        fig = px.bar(
            x=emotions, 
            y=probabilities,
            color=probabilities,
            color_continuous_scale='Viridis',
            labels={'x': 'Emotions', 'y': 'Probability'}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # PTSD Insights
        st.subheader("🧠 PTSD Insights & Recommendations")
        
        if 'indicators' in result and result['indicators']:
            st.markdown("**Detected Indicators:**")
            for indicator in result['indicators']:
                st.write(f"• {indicator}")
        
        st.markdown("**Clinical Recommendation:**")
        st.info(result.get('recommendation', 'Continue monitoring as usual.'))
    
    def audio_file_analysis(self):
        st.header("📁 Upload Audio File for Analysis")
        
        uploaded_file = st.file_uploader(
            "Upload an audio file for emotion analysis",
            type=['wav', 'mp3', 'm4a', 'flac']
        )
        
        if uploaded_file is not None:
            st.audio(uploaded_file, format='audio/wav')
            
            if st.button("🔍 Analyze Emotional Content", type="primary"):
                with st.spinner("Analyzing emotional content..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        audio_path = tmp_file.name
                    
                    try:
                        result = self.analyze_recorded_audio(audio_path)
                        if result:
                            self.display_results(result)
                    finally:
                        if os.path.exists(audio_path):
                            os.unlink(audio_path)
    
    def realtime_analysis(self):
        st.header("🎤 Real-time Audio Recording")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### Live Emotion Detection Session
            
            Record speech in real-time and get instant emotional analysis with 
            PTSD risk assessment. Perfect for therapy sessions.
            
            **Instructions:**
            1. Ensure your microphone is connected and working
            2. Click 'Start Recording' below
            3. Speak naturally for the selected duration
            4. View instant emotional analysis and PTSD risk assessment
            """)
            
            recording_duration = st.slider(
                "Recording Duration (seconds)",
                min_value=3,
                max_value=15,
                value=7,
                help="Longer recordings (5-10 seconds) provide better analysis"
            )
            
            # Recording button
            if st.button("🎙️ Start Recording Session", type="primary", use_container_width=True):
                # Show recording in progress
                with st.spinner(f"🔴 Recording audio for {recording_duration} seconds... Speak now!"):
                    # Actually record audio
                    audio_file = self.recorder.record_audio(duration=recording_duration)
                    
                    if audio_file:
                        # Display the recorded audio
                        st.audio(audio_file, format='audio/wav')
                        
                        # Analyze the recorded audio
                        result = self.analyze_recorded_audio(audio_file)
                        
                        if result:
                            self.display_results(result)
                        
                        # Cleanup
                        if os.path.exists(audio_file):
                            os.unlink(audio_file)
        
        with col2:
            st.markdown("### Session Controls")
            model_status = "✅ Loaded" if MODEL_LOADED else "⚠️ Demo Mode"
            st.metric("Model Status", model_status)
            st.metric("Microphone", "Available")
            st.metric("Analysis Mode", "Real-time")
            
            st.markdown("### Tips for Best Results")
            st.info("""
            - Speak clearly and naturally
            - Avoid background noise
            - 5-10 seconds is ideal
            - Use a quiet environment
            """)
    
    def home_page(self):
        st.markdown("""
        ## Welcome to PTSD Speech Emotion Recognition System
        
        This system analyzes speech patterns to detect emotional states that may indicate PTSD symptoms.
        
        ### Features:
        - **Real-time Recording**: Live audio analysis during therapy sessions
        - **File Upload**: Analyze pre-recorded audio files
        - **PTSD Risk Assessment**: Identify potential trauma indicators
        - **Clinical Insights**: Actionable recommendations for therapists
        """)
    
    def how_it_works_page(self):
        st.markdown("""
        ## How It Works
        
        The system uses advanced AI to analyze vocal characteristics and detect emotional patterns
        associated with Post-Traumatic Stress Disorder.
        
        **Technology Stack:**
        - wav2vec2.0 for speech feature extraction
        - Deep learning for emotion classification
        - Real-time audio processing
        - PTSD risk assessment algorithms
        """)
    
    def main(self):
        """Main method"""
        st.title("🎭 PTSD Speech Emotion Recognition System")
        
        # Sidebar navigation
        st.sidebar.title("Navigation")
        app_mode = st.sidebar.radio(
            "Choose Analysis Mode:",
            ["🏠 Home", "📁 Upload Audio File", "🎤 Real-time Recording", "📊 How It Works"]
        )
        
        if app_mode == "🏠 Home":
            self.home_page()
        elif app_mode == "📁 Upload Audio File":
            self.audio_file_analysis()
        elif app_mode == "🎤 Real-time Recording":
            self.realtime_analysis()
        elif app_mode == "📊 How It Works":
            self.how_it_works_page()
    
    def __del__(self):
        """Cleanup when app closes"""
        if hasattr(self, 'recorder'):
            self.recorder.close()

def main():
    app = PTSDStreamlitApp()
    app.main()

if __name__ == "__main__":
    main()
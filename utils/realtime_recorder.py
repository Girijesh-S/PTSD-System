import librosa
import pyaudio
import wave
import numpy as np
import threading
import time
from datetime import datetime
import tempfile
import os

class RealTimeRecorder:
    def __init__(self, rate=16000, chunksize=1024, channels=1):
        self.rate = rate
        self.chunksize = chunksize
        self.channels = channels
        self.audio = pyaudio.PyAudio()
        self.frames = []
        self.is_recording = False
        
    def record_audio(self, duration):
        """Record audio for specified duration"""
        self.frames = []
        
        try:
            stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunksize
            )
            
            print(f"Recording for {duration} seconds...")
            
            for i in range(0, int(self.rate / self.chunksize * duration)):
                data = stream.read(self.chunksize)
                self.frames.append(data)
            
            stream.stop_stream()
            stream.close()
            
            return self.frames
            
        except Exception as e:
            print(f"Recording error: {e}")
            return None
    
    def record_to_file(self, duration):
        """Record audio and save directly to temporary file"""
        frames = self.record_audio(duration)
        if frames:
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            self.save_audio(frames, temp_file.name)
            return temp_file.name
        return None
    
    def save_audio(self, frames, filename):
        """Save recorded audio to file"""
        try:
            wf = wave.open(filename, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(frames))
            wf.close()
            return True
        except Exception as e:
            print(f"Error saving audio: {e}")
            return False
        
    def close(self):
        """Clean up"""
        self.audio.terminate()


class RealTimeAudioProcessor:
    """Additional class for real-time audio processing if needed"""
    def __init__(self, rate=16000):
        self.rate = rate
        
    def analyze_audio_chunk(self, audio_chunk):
        """Analyze a single audio chunk in real-time"""
        try:
            # Convert byte data to numpy array
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            # Basic audio analysis
            rms_energy = np.sqrt(np.mean(audio_float**2))
            zero_crossing = np.mean(librosa.feature.zero_crossing_rate(audio_float))
            
            return {
                'rms_energy': rms_energy,
                'zero_crossing_rate': zero_crossing,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Audio processing error: {e}")
            return None


# For backward compatibility
PTSDEmotionPredictor = RealTimeAudioProcessor
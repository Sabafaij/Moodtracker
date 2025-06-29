import librosa
import librosa.display
import numpy as np
import soundfile as sf
import os


class VoiceAnalyzer:
    """
    Analyzes voice patterns to detect indicators that may relate to mental health states.
    """
    
    def __init__(self):
        """Initialize the voice analyzer with default parameters."""
        self.sample_rate = 22050  # Default sample rate
    
    def analyze_audio(self, audio_file):
        """
        Analyze an audio file to extract voice features.
        
        Args:
            audio_file: Path to the audio file
            
        Returns:
            dict: Analysis results including pitch, energy, and other voice features
        """
        # Check if file exists
        if not os.path.exists(audio_file):
            return None
        
        try:
            # Load the audio file
            y, sr = librosa.load(audio_file, sr=self.sample_rate)
            
            # Basic sanity check - make sure we have audio data
            if len(y) < sr:  # Less than 1 second of audio
                return None
            
            # Analyze pitch (fundamental frequency)
            pitch_data = self._analyze_pitch(y, sr)
            
            # Analyze energy/volume
            energy_data = self._analyze_energy(y)
            
            # Analyze speech rate
            speech_rate = self._estimate_speech_rate(y, sr)
            
            # Extract additional voice features
            voice_features = self._extract_voice_features(y, sr)
            
            return {
                'pitch': pitch_data,
                'energy': energy_data,
                'speech_rate': speech_rate,
                'voice_features': voice_features
            }
            
        except Exception as e:
            print(f"Error analyzing audio: {str(e)}")
            return None
    
    def _analyze_pitch(self, y, sr):
        """
        Analyze pitch (fundamental frequency) variation in the audio.
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            dict: Pitch analysis results
        """
        # Use librosa to extract pitch
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        
        # Get the pitch with highest magnitude for each frame
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:  # Only include valid pitch values
                pitch_values.append(pitch)
        
        if not pitch_values:
            return {'mean': 0, 'variance': 0, 'values': []}
        
        # Calculate statistics
        mean_pitch = np.mean(pitch_values)
        pitch_variance = np.var(pitch_values)
        
        return {
            'mean': float(mean_pitch),
            'variance': float(pitch_variance),
            'values': pitch_values
        }
    
    def _analyze_energy(self, y):
        """
        Analyze energy/volume variation in the audio.
        
        Args:
            y: Audio time series
            
        Returns:
            dict: Energy analysis results
        """
        # Calculate the RMS energy
        energy = librosa.feature.rms(y=y)[0]
        
        # Convert to dB scale
        energy_db = librosa.amplitude_to_db(energy, ref=np.max)
        
        # Calculate statistics
        if len(energy_db) > 0:
            mean_energy = np.mean(energy_db)
            energy_variance = np.var(energy_db)
        else:
            mean_energy = 0
            energy_variance = 0
        
        return {
            'mean': float(mean_energy),
            'variance': float(energy_variance),
            'values': energy_db.tolist()
        }
    
    def _estimate_speech_rate(self, y, sr):
        """
        Estimate speech rate by detecting syllable-like events.
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            float: Estimated speech rate (syllables per second)
        """
        # Calculate onset envelope
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        
        # Detect onsets (roughly correlates with syllables)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        
        # Count onsets and divide by duration
        num_onsets = len(onsets)
        duration = len(y) / sr  # Duration in seconds
        
        if duration > 0:
            speech_rate = num_onsets / duration
        else:
            speech_rate = 0
        
        return float(speech_rate)
    
    def _extract_voice_features(self, y, sr):
        """
        Extract additional voice features that may correlate with mental states.
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            dict: Additional voice features
        """
        features = {}
        
        # Spectral centroid - related to perceived brightness of sound
        # (lower in depressed speech)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = float(np.mean(cent))
        
        # Spectral bandwidth - related to perceived richness of sound
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features['spectral_bandwidth_mean'] = float(np.mean(bandwidth))
        
        # Zero crossing rate - related to perceived noisiness/voice quality
        # (can indicate tension)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zero_crossing_rate_mean'] = float(np.mean(zcr))
        
        # MFCCs - overall spectral shape, related to voice quality
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfccs, axis=1)
        
        # Add first few MFCCs (most important for voice quality)
        for i in range(min(5, len(mfcc_means))):
            features[f'mfcc{i+1}_mean'] = float(mfcc_means[i])
        
        # Jitter - frequency variation in voice (increased in some conditions)
        # This is a simplified approximation
        if len(y) > sr//10:  # At least 0.1 seconds
            chunks = np.array_split(y[:sr], 10)  # Split first second into 10 chunks
            chunk_pitches = []
            
            for chunk in chunks:
                if len(chunk) > 0:
                    pitches, _ = librosa.piptrack(y=chunk, sr=sr)
                    pitch = np.mean(pitches) if pitches.size > 0 else 0
                    chunk_pitches.append(pitch)
            
            if chunk_pitches:
                jitter = np.std(chunk_pitches) / max(np.mean(chunk_pitches), 1)
                features['jitter'] = float(jitter)
        
        return features

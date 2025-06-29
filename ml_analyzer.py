import numpy as np
from sklearn.preprocessing import MinMaxScaler


class MentalHealthAnalyzer:
    """
    Analyzes combined facial and voice data to detect potential indicators
    related to mental health states.
    """
    
    def __init__(self):
        """Initialize the mental health analyzer."""
        # Initialize scalers for normalizing features
        self.face_scaler = MinMaxScaler()
        self.voice_scaler = MinMaxScaler()
        
        # Define feature weights for different indicators
        # These would ideally be learned from training data
        self.mood_feature_weights = {
            # Facial features
            'mouth_curvature': 0.3,
            'eye_openness': 0.2,
            'eyebrow_position': 0.1,
            # Emotions
            'happy': 0.3,
            'sad': -0.3,
            # Voice features
            'pitch_variance': 0.2,
            'energy_mean': 0.2,
            'spectral_centroid_mean': 0.2
        }
        
        self.anxiety_feature_weights = {
            # Facial features
            'face_symmetry': -0.2,
            # Emotions
            'fearful': 0.3,
            'surprised': 0.1,
            'neutral': -0.2,
            # Voice features
            'speech_rate': 0.3,
            'jitter': 0.2,
            'zero_crossing_rate_mean': 0.2
        }
        
        self.energy_feature_weights = {
            # Facial features
            'eye_openness': 0.2,
            # Emotions
            'happy': 0.2,
            'sad': -0.3,
            # Voice features
            'energy_mean': 0.3,
            'speech_rate': 0.3,
            'pitch_variance': 0.2
        }
    
    def analyze(self, facial_analysis, voice_analysis):
        """
        Perform combined analysis of facial and voice data.
        
        Args:
            facial_analysis: Results from face analyzer
            voice_analysis: Results from voice analyzer
            
        Returns:
            dict: Combined analysis results
        """
        # Extract relevant features from facial analysis
        facial_features = facial_analysis.get('facial_features', {})
        emotions = facial_analysis.get('emotions', {})
        
        # Extract relevant features from voice analysis
        pitch_data = voice_analysis.get('pitch', {})
        energy_data = voice_analysis.get('energy', {})
        speech_rate = voice_analysis.get('speech_rate', 0)
        voice_features = voice_analysis.get('voice_features', {})
        
        # Combine features into a unified representation
        combined_features = self._combine_features(
            facial_features, emotions, pitch_data, energy_data, speech_rate, voice_features
        )
        
        # Calculate mental health indicators
        mood_indicators = self._calculate_mood_indicators(combined_features)
        anxiety_indicators = self._calculate_anxiety_indicators(combined_features)
        energy_indicators = self._calculate_energy_indicators(combined_features)
        
        # Generate an overall assessment
        overall_assessment = self._generate_assessment(
            mood_indicators, anxiety_indicators, energy_indicators
        )
        
        return {
            'mood_indicators': mood_indicators,
            'anxiety_indicators': anxiety_indicators,
            'energy_indicators': energy_indicators,
            'overall_assessment': overall_assessment
        }
    
    def _combine_features(self, facial_features, emotions, pitch_data, energy_data, speech_rate, voice_features):
        """
        Combine facial and voice features into a unified representation.
        
        Returns:
            dict: Combined feature set
        """
        combined = {}
        
        # Add facial features
        for key, value in facial_features.items():
            combined[key] = value
        
        # Add emotions
        for emotion, value in emotions.items():
            combined[emotion] = value
        
        # Add pitch features
        combined['pitch_mean'] = pitch_data.get('mean', 0)
        combined['pitch_variance'] = pitch_data.get('variance', 0)
        
        # Add energy features
        combined['energy_mean'] = energy_data.get('mean', 0)
        combined['energy_variance'] = energy_data.get('variance', 0)
        
        # Add speech rate
        combined['speech_rate'] = speech_rate
        
        # Add voice features
        for key, value in voice_features.items():
            combined[key] = value
        
        return combined
    
    def _calculate_mood_indicators(self, features):
        """
        Calculate indicators related to mood state.
        
        Args:
            features: Combined feature set
            
        Returns:
            dict: Mood indicators
        """
        indicators = {}
        
        # Calculate positive mood indicator
        positive_score = 0
        positive_weights = 0
        
        for feature, weight in self.mood_feature_weights.items():
            if weight > 0 and feature in features:
                positive_score += features[feature] * weight
                positive_weights += weight
        
        if positive_weights > 0:
            indicators['positive_affect'] = positive_score / positive_weights
        else:
            indicators['positive_affect'] = 0.5  # Neutral default
        
        # Calculate negative mood indicator
        negative_score = 0
        negative_weights = 0
        
        for feature, weight in self.mood_feature_weights.items():
            if weight < 0 and feature in features:
                negative_score += features[feature] * abs(weight)
                negative_weights += abs(weight)
        
        if negative_weights > 0:
            indicators['negative_affect'] = negative_score / negative_weights
        else:
            indicators['negative_affect'] = 0.5  # Neutral default
        
        # Calculate overall mood balance
        indicators['mood_balance'] = indicators['positive_affect'] / max(indicators['negative_affect'], 0.1)
        
        # Normalize to 0-1 range
        indicators['mood_balance'] = min(max(indicators['mood_balance'] / 2, 0), 1)
        
        # Calculate expression variability (can indicate mood regulation)
        if 'pitch_variance' in features and 'energy_variance' in features:
            indicators['expression_variability'] = (
                features['pitch_variance'] / 500 +  # Normalize pitch variance
                features['energy_variance'] / 100   # Normalize energy variance
            ) / 2
            indicators['expression_variability'] = min(max(indicators['expression_variability'], 0), 1)
        else:
            indicators['expression_variability'] = 0.5
        
        return indicators
    
    def _calculate_anxiety_indicators(self, features):
        """
        Calculate indicators related to anxiety state.
        
        Args:
            features: Combined feature set
            
        Returns:
            dict: Anxiety indicators
        """
        indicators = {}
        
        # Calculate tension indicator
        tension_score = 0
        tension_weights = 0
        
        for feature, weight in self.anxiety_feature_weights.items():
            if feature in features:
                # For negative weights, we invert the feature value (1 - feature)
                if weight < 0:
                    tension_score += (1 - features[feature]) * abs(weight)
                else:
                    tension_score += features[feature] * weight
                tension_weights += abs(weight)
        
        if tension_weights > 0:
            indicators['tension'] = tension_score / tension_weights
        else:
            indicators['tension'] = 0.5  # Neutral default
        
        # Calculate speech pattern irregularity
        if 'jitter' in features and 'speech_rate' in features:
            # High jitter and very fast or very slow speech can indicate anxiety
            speech_rate_deviation = abs(features['speech_rate'] - 4.5) / 4.5  # Deviation from average
            indicators['speech_irregularity'] = (
                min(features.get('jitter', 0) * 5, 1) * 0.6 +  # Weight jitter more heavily
                min(speech_rate_deviation, 1) * 0.4
            )
        else:
            indicators['speech_irregularity'] = 0.5
        
        # Calculate facial tension
        if 'face_symmetry' in features and 'eyebrow_position' in features:
            indicators['facial_tension'] = (
                (1 - features['face_symmetry']) * 0.7 +  # Low symmetry indicates tension
                min(max(50 - features['eyebrow_position'], 0) / 50, 1) * 0.3  # Lowered eyebrows
            )
        else:
            indicators['facial_tension'] = 0.5
        
        return indicators
    
    def _calculate_energy_indicators(self, features):
        """
        Calculate indicators related to energy/activation state.
        
        Args:
            features: Combined feature set
            
        Returns:
            dict: Energy indicators
        """
        indicators = {}
        
        # Calculate overall energy level
        energy_score = 0
        energy_weights = 0
        
        for feature, weight in self.energy_feature_weights.items():
            if feature in features:
                if weight < 0:
                    # For negative weights, we invert the feature
                    energy_score += (1 - features[feature]) * abs(weight)
                else:
                    energy_score += features[feature] * weight
                energy_weights += abs(weight)
        
        if energy_weights > 0:
            indicators['energy_level'] = energy_score / energy_weights
        else:
            indicators['energy_level'] = 0.5  # Neutral default
        
        # Calculate voice energy
        if 'energy_mean' in features and 'spectral_centroid_mean' in features:
            # Normalize values to reasonable ranges
            norm_energy = min(max((features['energy_mean'] + 30) / 30, 0), 1)  # Typical range -30 to 0 dB
            norm_centroid = min(features['spectral_centroid_mean'] / 5000, 1)  # Normalize to typical max
            
            indicators['voice_energy'] = (norm_energy * 0.7 + norm_centroid * 0.3)
        else:
            indicators['voice_energy'] = 0.5
        
        # Calculate facial animation
        if 'eye_openness' in features and 'mouth_curvature' in features:
            indicators['facial_animation'] = (
                features['eye_openness'] * 0.4 +
                abs(features['mouth_curvature']) * 0.6  # Absolute value - both smiles and frowns show animation
            )
            indicators['facial_animation'] = min(indicators['facial_animation'], 1)
        else:
            indicators['facial_animation'] = 0.5
        
        return indicators
    
    def _generate_assessment(self, mood_indicators, anxiety_indicators, energy_indicators):
        """
        Generate an overall textual assessment based on the indicators.
        
        Args:
            mood_indicators: Dict of mood-related indicators
            anxiety_indicators: Dict of anxiety-related indicators
            energy_indicators: Dict of energy-related indicators
            
        Returns:
            str: Overall assessment text
        """
        # Get primary indicators
        mood_balance = mood_indicators.get('mood_balance', 0.5)
        tension = anxiety_indicators.get('tension', 0.5)
        energy_level = energy_indicators.get('energy_level', 0.5)
        
        # Categorize primary indicators into low/moderate/high
        mood_category = self._categorize_indicator(mood_balance)
        tension_category = self._categorize_indicator(tension)
        energy_category = self._categorize_indicator(energy_level)
        
        # Build assessment text
        assessment = "## Expression Pattern Analysis\n\n"
        
        # Mood assessment
        assessment += "### Mood Expression Patterns\n"
        if mood_category == "low":
            assessment += "Your facial expressions and voice patterns show characteristics that are sometimes associated with lower mood states. "
            assessment += "This includes less facial animation and reduced vocal variability.\n\n"
        elif mood_category == "high":
            assessment += "Your expressions show patterns often associated with positive mood states. "
            assessment += "This includes animated facial expressions and varied vocal patterns.\n\n"
        else:
            assessment += "Your expressions show balanced mood patterns with moderate facial animation and vocal variability.\n\n"
        
        # Tension assessment
        assessment += "### Tension Expression Patterns\n"
        if tension_category == "high":
            assessment += "Your expressions show some patterns that can be associated with tension or stress. "
            assessment += "This includes specific facial muscle patterns and speech characteristics.\n\n"
        elif tension_category == "low":
            assessment += "Your expressions show relatively relaxed patterns with minimal tension indicators in both face and voice.\n\n"
        else:
            assessment += "Your expressions show moderate levels of the patterns sometimes associated with tension.\n\n"
        
        # Energy assessment
        assessment += "### Energy Expression Patterns\n"
        if energy_category == "low":
            assessment += "Your expressions suggest lower energy patterns in both facial activity and vocal characteristics. "
            assessment += "This may reflect fatigue or reduced activation.\n\n"
        elif energy_category == "high":
            assessment += "Your expressions demonstrate energetic patterns in facial animation and voice characteristics, "
            assessment += "suggesting higher levels of activation.\n\n"
        else:
            assessment += "Your expressions show moderate energy levels with balanced facial animation and vocal projection.\n\n"
        
        # Add important disclaimer
        assessment += "### Important Note\n"
        assessment += "This analysis is based solely on computer detection of expression patterns and is for educational purposes only. "
        assessment += "It is not a clinical assessment and should not be used for diagnosis or treatment decisions. "
        assessment += "Many factors can influence expressions, including cultural background, individual differences, and current context."
        
        return assessment
    
    def _categorize_indicator(self, value, low_threshold=0.4, high_threshold=0.6):
        """
        Categorize a numerical indicator into low/moderate/high.
        
        Args:
            value: Indicator value (0-1 range)
            low_threshold: Threshold for "low" category
            high_threshold: Threshold for "high" category
            
        Returns:
            str: Category ("low", "moderate", or "high")
        """
        if value < low_threshold:
            return "low"
        elif value > high_threshold:
            return "high"
        else:
            return "moderate"

import face_recognition
import cv2
import numpy as np
from sklearn.preprocessing import normalize


class FaceAnalyzer:
    """
    Analyzes facial expressions to detect emotions and other facial features
    that may be indicators of mental health states.
    """
    
    def __init__(self):
        # Emotion classification model is a simple rule-based system
        # in this implementation. In a production system, this would
        # likely be a trained ML model.
        self.emotion_labels = ['neutral', 'happy', 'sad', 'angry', 'surprised', 'fearful']
        
    def analyze_face(self, image):
        """
        Analyze a face image to detect emotions and facial features.
        
        Args:
            image: RGB image containing a face
            
        Returns:
            dict: Analysis results including emotions and facial features
        """
        # Detect face locations in the image
        face_locations = face_recognition.face_locations(image)
        
        # If no face is found, return None
        if not face_locations:
            return None
            
        # For simplicity, we'll analyze only the first face found
        face_location = face_locations[0]  # (top, right, bottom, left)
        
        # Get facial landmarks
        face_landmarks = face_recognition.face_landmarks(image, [face_location])
        
        if not face_landmarks:
            return None
            
        landmarks = face_landmarks[0]
        
        # Extract features from landmarks
        facial_features = self._extract_facial_features(landmarks)
        
        # Determine emotions based on facial features
        emotions = self._determine_emotions(facial_features)
        
        return {
            'emotions': emotions,
            'facial_features': facial_features
        }
    
    def _extract_facial_features(self, landmarks):
        """
        Extract meaningful features from facial landmarks.
        
        Args:
            landmarks: Dictionary of facial landmarks
            
        Returns:
            dict: Extracted facial features
        """
        features = {}
        
        # Eye openness (ratio of eye height to width)
        left_eye = landmarks.get('left_eye', [])
        right_eye = landmarks.get('right_eye', [])
        
        if left_eye and right_eye:
            # Calculate eye openness
            left_eye_height = self._get_vertical_distance(left_eye)
            left_eye_width = self._get_horizontal_distance(left_eye)
            right_eye_height = self._get_vertical_distance(right_eye)
            right_eye_width = self._get_horizontal_distance(right_eye)
            
            left_eye_ratio = left_eye_height / left_eye_width if left_eye_width > 0 else 0
            right_eye_ratio = right_eye_height / right_eye_width if right_eye_width > 0 else 0
            
            features['eye_openness'] = (left_eye_ratio + right_eye_ratio) / 2
        
        # Eyebrow position (relative to eye)
        left_eyebrow = landmarks.get('left_eyebrow', [])
        right_eyebrow = landmarks.get('right_eyebrow', [])
        
        if left_eyebrow and right_eyebrow and left_eye and right_eye:
            # Calculate average eyebrow height
            left_eyebrow_height = np.mean([p[1] for p in left_eyebrow])
            right_eyebrow_height = np.mean([p[1] for p in right_eyebrow])
            
            # Calculate average eye height
            left_eye_height = np.mean([p[1] for p in left_eye])
            right_eye_height = np.mean([p[1] for p in right_eye])
            
            # Distance between eyebrow and eye (smaller value = eyebrows lowered, might indicate concern/anger)
            left_distance = left_eye_height - left_eyebrow_height
            right_distance = right_eye_height - right_eyebrow_height
            
            features['eyebrow_position'] = (left_distance + right_distance) / 2
        
        # Mouth curvature (smile/frown detection)
        top_lip = landmarks.get('top_lip', [])
        bottom_lip = landmarks.get('bottom_lip', [])
        
        if top_lip and bottom_lip:
            # Get the corners and middle points of the lips
            left_corner = top_lip[0]
            right_corner = top_lip[6]
            top_middle = top_lip[3]
            bottom_middle = bottom_lip[3]
            
            # Calculate the average height of the corners
            corner_height = (left_corner[1] + right_corner[1]) / 2
            
            # Calculate mouth curvature (positive for smile, negative for frown)
            # Simplified model: difference between corner height and middle height
            top_curvature = corner_height - top_middle[1]
            bottom_curvature = corner_height - bottom_middle[1]
            
            features['mouth_curvature'] = (top_curvature + bottom_curvature) / 2
        
        # Face symmetry (can indicate stress when asymmetrical)
        features['face_symmetry'] = self._calculate_face_symmetry(landmarks)
        
        return features
    
    def _determine_emotions(self, facial_features):
        """
        Determine emotions based on facial features.
        This is a simplified rule-based approach.
        
        Args:
            facial_features: Dictionary of facial features
            
        Returns:
            dict: Emotion probabilities
        """
        # Initialize emotion scores
        emotion_scores = {emotion: 0.0 for emotion in self.emotion_labels}
        
        # Extract features
        eye_openness = facial_features.get('eye_openness', 0)
        eyebrow_position = facial_features.get('eyebrow_position', 0)
        mouth_curvature = facial_features.get('mouth_curvature', 0)
        face_symmetry = facial_features.get('face_symmetry', 1)
        
        # Simple rule-based scoring for emotions
        # Happy: high mouth curvature (smile), moderate to high eye openness
        emotion_scores['happy'] += max(0, mouth_curvature * 5)
        emotion_scores['happy'] += max(0, eye_openness * 2)
        
        # Sad: negative mouth curvature (frown), lower eye openness
        emotion_scores['sad'] += max(0, -mouth_curvature * 4)
        emotion_scores['sad'] += max(0, (0.3 - eye_openness) * 3)
        
        # Angry: lowered eyebrows, slight frown
        emotion_scores['angry'] += max(0, (50 - eyebrow_position) * 0.1)
        emotion_scores['angry'] += max(0, -mouth_curvature * 2)
        
        # Surprised: raised eyebrows, wide eyes
        emotion_scores['surprised'] += max(0, eyebrow_position * 0.05)
        emotion_scores['surprised'] += max(0, eye_openness * 4)
        
        # Fearful: raised eyebrows, wide eyes, tense mouth
        emotion_scores['fearful'] += max(0, eyebrow_position * 0.03)
        emotion_scores['fearful'] += max(0, eye_openness * 2)
        emotion_scores['fearful'] += max(0, (1 - face_symmetry) * 5)
        
        # Neutral: balanced features, high symmetry
        emotion_scores['neutral'] += max(0, face_symmetry * 3)
        emotion_scores['neutral'] += max(0, (1 - abs(mouth_curvature)) * 2)
        
        # Normalize scores to get probabilities
        total = sum(emotion_scores.values())
        if total > 0:
            emotion_probs = {k: v/total for k, v in emotion_scores.items()}
        else:
            # If no scores, default to neutral
            emotion_probs = {k: 0.0 for k in emotion_scores}
            emotion_probs['neutral'] = 1.0
        
        return emotion_probs
    
    def _get_vertical_distance(self, points):
        """Calculate the vertical distance (height) of a set of points."""
        if not points:
            return 0
        y_values = [p[1] for p in points]
        return max(y_values) - min(y_values)
    
    def _get_horizontal_distance(self, points):
        """Calculate the horizontal distance (width) of a set of points."""
        if not points:
            return 0
        x_values = [p[0] for p in points]
        return max(x_values) - min(x_values)
    
    def _calculate_face_symmetry(self, landmarks):
        """
        Calculate facial symmetry by comparing left and right sides.
        
        Args:
            landmarks: Facial landmarks dictionary
            
        Returns:
            float: Symmetry score (1 = perfectly symmetrical, 0 = asymmetrical)
        """
        # Get center line of face (approximate)
        nose_tip = landmarks.get('nose_tip', [[0, 0]])[0]
        nose_bridge = landmarks.get('nose_bridge', [[0, 0]])[0]
        
        if nose_tip == [0, 0] or nose_bridge == [0, 0]:
            return 0.5  # Default middle value if can't determine
            
        # Calculate center x-coordinate
        center_x = (nose_tip[0] + nose_bridge[0]) / 2
        
        # Compare left and right sides of key features
        symmetry_scores = []
        
        # Compare eyes
        left_eye = landmarks.get('left_eye', [])
        right_eye = landmarks.get('right_eye', [])
        
        if left_eye and right_eye:
            # Get average positions
            left_eye_center = np.mean(left_eye, axis=0)
            right_eye_center = np.mean(right_eye, axis=0)
            
            # Distance from center
            left_dist = abs(center_x - left_eye_center[0])
            right_dist = abs(right_eye_center[0] - center_x)
            
            # Compare the distances (ratio of smaller to larger)
            eye_symmetry = min(left_dist, right_dist) / max(left_dist, right_dist) if max(left_dist, right_dist) > 0 else 1
            symmetry_scores.append(eye_symmetry)
        
        # Similar comparisons for eyebrows, mouth corners, etc.
        # For simplicity, we'll use just the eyes in this implementation
        
        # Average the symmetry scores
        if symmetry_scores:
            return np.mean(symmetry_scores)
        return 0.5  # Default value

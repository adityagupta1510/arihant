"""
ARIHANT SOC - Prediction Service
================================
Handles ML model inference for all detection types
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import re

# Try importing ML libraries
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class PredictionService:
    """
    Centralized prediction service for all threat detection models
    """
    
    # Attack type mappings for network detection
    NETWORK_ATTACK_TYPES = {
        0: "Benign",
        1: "DDoS",
        2: "DoS",
        3: "Reconnaissance",
        4: "Theft",
        5: "Mirai",
        6: "Gafgyt",
        7: "Scanning",
        8: "Spamming"
    }
    
    # Severity mapping based on attack type
    SEVERITY_MAP = {
        "Benign": "LOW",
        "DDoS": "CRITICAL",
        "DoS": "HIGH",
        "Reconnaissance": "MEDIUM",
        "Theft": "CRITICAL",
        "Mirai": "CRITICAL",
        "Gafgyt": "HIGH",
        "Scanning": "LOW",
        "Spamming": "MEDIUM"
    }
    
    # Phishing indicators with weights
    PHISHING_INDICATORS = [
        {"pattern": r"\burgent\b", "weight": 15, "type": "urgency"},
        {"pattern": r"\bimmediately\b", "weight": 20, "type": "urgency"},
        {"pattern": r"verify your account", "weight": 25, "type": "credential_theft"},
        {"pattern": r"click here", "weight": 15, "type": "suspicious_link"},
        {"pattern": r"\bpassword\b", "weight": 20, "type": "credential_theft"},
        {"pattern": r"\bsuspended\b", "weight": 20, "type": "fear_tactic"},
        {"pattern": r"wire transfer", "weight": 30, "type": "financial_fraud"},
        {"pattern": r"\bbitcoin\b", "weight": 25, "type": "financial_fraud"},
        {"pattern": r"gift card", "weight": 25, "type": "financial_fraud"},
        {"pattern": r"act now", "weight": 15, "type": "urgency"},
        {"pattern": r"limited time", "weight": 10, "type": "urgency"},
        {"pattern": r"confirm your identity", "weight": 25, "type": "credential_theft"},
        {"pattern": r"unusual activity", "weight": 20, "type": "fear_tactic"},
        {"pattern": r"security alert", "weight": 15, "type": "impersonation"},
        {"pattern": r"account.*compromised", "weight": 25, "type": "fear_tactic"},
        {"pattern": r"update.*payment", "weight": 20, "type": "financial_fraud"},
        {"pattern": r"expire[sd]?", "weight": 15, "type": "urgency"},
        {"pattern": r"verify.*identity", "weight": 25, "type": "credential_theft"},
        {"pattern": r"login.*credentials", "weight": 20, "type": "credential_theft"},
        {"pattern": r"bank.*account", "weight": 20, "type": "financial_fraud"},
    ]
    
    def __init__(self, model_loader):
        self.model_loader = model_loader
    
    # ═══════════════════════════════════════════════════════════════════════════
    # NETWORK INTRUSION DETECTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def predict_network(self, features: List[float]) -> Dict[str, Any]:
        """
        Predict network intrusion using LSTM or RF model
        
        Args:
            features: List of network flow features
            
        Returns:
            Prediction result with attack type, confidence, severity
        """
        start_time = datetime.now()
        
        # Get model and scaler
        lstm_model = self.model_loader.get_model('network_lstm')
        rf_model = self.model_loader.get_model('network_rf')
        scaler = self.model_loader.get_scaler('network')
        attack_encoder = self.model_loader.get_encoder('attack')
        
        # Prepare features
        features_array = np.array(features).reshape(1, -1)
        
        # Scale features if scaler available
        if scaler is not None:
            try:
                # Ensure correct number of features
                expected_features = scaler.n_features_in_
                if features_array.shape[1] < expected_features:
                    # Pad with zeros
                    padding = np.zeros((1, expected_features - features_array.shape[1]))
                    features_array = np.hstack([features_array, padding])
                elif features_array.shape[1] > expected_features:
                    features_array = features_array[:, :expected_features]
                
                features_scaled = scaler.transform(features_array)
            except Exception as e:
                print(f"[WARN] Scaling failed: {e}")
                features_scaled = features_array
        else:
            features_scaled = features_array
        
        # Predict using LSTM (primary) or RF (fallback)
        prediction = 0
        confidence = 0.5
        probabilities = {}
        
        if lstm_model is not None and TF_AVAILABLE:
            try:
                # Reshape for LSTM: (samples, timesteps, features)
                lstm_input = features_scaled.reshape(1, 1, -1)
                pred_probs = lstm_model.predict(lstm_input, verbose=0)
                prediction = int(np.argmax(pred_probs, axis=1)[0])
                confidence = float(np.max(pred_probs))
                
                # Build probability dict
                for i, prob in enumerate(pred_probs[0]):
                    attack_name = self.NETWORK_ATTACK_TYPES.get(i, f"Class_{i}")
                    probabilities[attack_name] = float(prob)
                    
            except Exception as e:
                print(f"[WARN] LSTM prediction failed: {e}")
        
        elif rf_model is not None:
            try:
                prediction = int(rf_model.predict(features_scaled)[0])
                if hasattr(rf_model, 'predict_proba'):
                    pred_probs = rf_model.predict_proba(features_scaled)
                    confidence = float(np.max(pred_probs))
                    for i, prob in enumerate(pred_probs[0]):
                        attack_name = self.NETWORK_ATTACK_TYPES.get(i, f"Class_{i}")
                        probabilities[attack_name] = float(prob)
                else:
                    confidence = 0.85  # Default confidence for RF without proba
            except Exception as e:
                print(f"[WARN] RF prediction failed: {e}")
        
        # Decode attack type
        attack_type = self.NETWORK_ATTACK_TYPES.get(prediction, "Unknown")
        
        # Try to use encoder for more accurate mapping
        if attack_encoder is not None:
            try:
                attack_type = attack_encoder.inverse_transform([prediction])[0]
            except:
                pass
        
        # Determine severity
        severity = self.SEVERITY_MAP.get(attack_type, "MEDIUM")
        is_attack = attack_type != "Benign"
        
        # Generate recommendation
        recommendation = self._get_network_recommendation(attack_type, confidence)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return {
            "attack_type": attack_type,
            "attack_label": prediction,
            "confidence": round(confidence, 4),
            "severity": severity,
            "is_attack": is_attack,
            "probabilities": probabilities if probabilities else None,
            "recommendation": recommendation,
            "processing_time_ms": round(processing_time, 2)
        }
    
    def _get_network_recommendation(self, attack_type: str, confidence: float) -> str:
        """Generate recommendation based on attack type"""
        recommendations = {
            "Benign": "No action required. Traffic appears normal.",
            "DDoS": "Enable rate limiting immediately. Consider blocking source IP ranges. Alert NOC team.",
            "DoS": "Implement traffic filtering. Monitor bandwidth utilization. Consider null routing.",
            "Reconnaissance": "Monitor for follow-up attacks. Review firewall rules. Log all activity from source.",
            "Theft": "Isolate affected systems immediately. Initiate incident response. Preserve evidence.",
            "Mirai": "Block IoT device traffic. Scan for compromised devices. Update firmware.",
            "Gafgyt": "Quarantine infected devices. Reset credentials. Patch vulnerabilities.",
            "Scanning": "Monitor for exploitation attempts. Review exposed services. Update IDS signatures.",
            "Spamming": "Block source IP. Update email filters. Report to abuse contacts."
        }
        return recommendations.get(attack_type, "Investigate further and monitor for additional activity.")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # APPLICATION LAYER DETECTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def predict_application(
        self, 
        features: Optional[List[float]] = None,
        request_data: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Predict application layer attacks
        
        Args:
            features: Preprocessed feature array
            request_data: Raw request data for pattern matching
            
        Returns:
            Prediction result
        """
        start_time = datetime.now()
        
        lstm_model = self.model_loader.get_model('application_lstm')
        scaler = self.model_loader.get_scaler('application')
        columns = self.model_loader.get_columns('application')
        
        prediction = 0
        confidence = 0.5
        indicators = []
        attack_type = "Benign"
        
        # If we have features, use the model
        if features is not None and lstm_model is not None and TF_AVAILABLE:
            try:
                features_array = np.array(features).reshape(1, -1)
                
                # Scale if scaler available
                if scaler is not None:
                    expected_features = scaler.n_features_in_
                    if features_array.shape[1] < expected_features:
                        padding = np.zeros((1, expected_features - features_array.shape[1]))
                        features_array = np.hstack([features_array, padding])
                    elif features_array.shape[1] > expected_features:
                        features_array = features_array[:, :expected_features]
                    features_scaled = scaler.transform(features_array)
                else:
                    features_scaled = features_array
                
                # Reshape for LSTM
                lstm_input = features_scaled.reshape(1, 1, -1)
                pred_prob = lstm_model.predict(lstm_input, verbose=0)[0][0]
                
                prediction = 1 if pred_prob > 0.5 else 0
                confidence = float(pred_prob) if prediction == 1 else float(1 - pred_prob)
                
            except Exception as e:
                print(f"[WARN] Application LSTM prediction failed: {e}")
        
        # Pattern-based detection for request_data
        if request_data:
            pattern_result = self._detect_app_patterns(request_data)
            if pattern_result['is_attack']:
                prediction = 1
                confidence = max(confidence, pattern_result['confidence'])
                attack_type = pattern_result['attack_type']
                indicators = pattern_result['indicators']
        
        # Determine attack type if model predicted attack
        if prediction == 1 and attack_type == "Benign":
            attack_type = "Application Layer Attack"
        
        # Determine severity
        severity = "CRITICAL" if confidence > 0.9 else "HIGH" if confidence > 0.7 else "MEDIUM"
        if prediction == 0:
            severity = "LOW"
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return {
            "attack_type": attack_type,
            "confidence": round(confidence, 4),
            "severity": severity,
            "is_attack": prediction == 1,
            "attack_category": self._get_attack_category(attack_type),
            "indicators": indicators if indicators else None,
            "recommendation": self._get_app_recommendation(attack_type),
            "processing_time_ms": round(processing_time, 2)
        }
    
    def _detect_app_patterns(self, data: str) -> Dict[str, Any]:
        """Detect common application attack patterns"""
        data_lower = data.lower()
        
        # SQL Injection patterns
        sql_patterns = [
            r"(\bor\b|\band\b)\s+\d+\s*=\s*\d+",
            r"union\s+(all\s+)?select",
            r";\s*drop\s+table",
            r"'\s*or\s+'",
            r"--\s*$",
            r"/\*.*\*/",
            r"exec\s*\(",
            r"xp_cmdshell"
        ]
        
        # XSS patterns
        xss_patterns = [
            r"<script[^>]*>",
            r"javascript:",
            r"on\w+\s*=",
            r"<img[^>]+onerror",
            r"<svg[^>]+onload"
        ]
        
        # Command injection patterns
        cmd_patterns = [
            r";\s*(ls|cat|rm|wget|curl)",
            r"\|\s*(ls|cat|rm|wget|curl)",
            r"`[^`]+`",
            r"\$\([^)]+\)"
        ]
        
        # Path traversal patterns
        path_patterns = [
            r"\.\./",
            r"\.\.\\",
            r"%2e%2e",
            r"etc/passwd",
            r"windows/system32"
        ]
        
        indicators = []
        attack_type = "Benign"
        confidence = 0.0
        
        # Check SQL Injection
        for pattern in sql_patterns:
            if re.search(pattern, data_lower):
                indicators.append(f"SQL pattern: {pattern}")
                attack_type = "SQL Injection"
                confidence = max(confidence, 0.85)
        
        # Check XSS
        for pattern in xss_patterns:
            if re.search(pattern, data_lower):
                indicators.append(f"XSS pattern: {pattern}")
                attack_type = "Cross-Site Scripting (XSS)"
                confidence = max(confidence, 0.80)
        
        # Check Command Injection
        for pattern in cmd_patterns:
            if re.search(pattern, data_lower):
                indicators.append(f"Command injection: {pattern}")
                attack_type = "Command Injection"
                confidence = max(confidence, 0.90)
        
        # Check Path Traversal
        for pattern in path_patterns:
            if re.search(pattern, data_lower):
                indicators.append(f"Path traversal: {pattern}")
                attack_type = "Path Traversal"
                confidence = max(confidence, 0.75)
        
        return {
            "is_attack": len(indicators) > 0,
            "attack_type": attack_type,
            "confidence": confidence,
            "indicators": indicators
        }
    
    def _get_attack_category(self, attack_type: str) -> str:
        """Get attack category"""
        categories = {
            "SQL Injection": "Injection",
            "Cross-Site Scripting (XSS)": "Injection",
            "Command Injection": "Injection",
            "Path Traversal": "Broken Access Control",
            "Application Layer Attack": "Unknown"
        }
        return categories.get(attack_type, "Unknown")
    
    def _get_app_recommendation(self, attack_type: str) -> str:
        """Get recommendation for application attacks"""
        recommendations = {
            "Benign": "No action required. Request appears legitimate.",
            "SQL Injection": "Block request. Review parameterized queries. Implement input validation.",
            "Cross-Site Scripting (XSS)": "Sanitize output. Implement Content Security Policy. Encode user input.",
            "Command Injection": "Block immediately. Review system calls. Implement strict input validation.",
            "Path Traversal": "Block request. Validate file paths. Implement chroot or sandboxing.",
            "Application Layer Attack": "Investigate request. Review application logs. Consider WAF rules."
        }
        return recommendations.get(attack_type, "Investigate and implement appropriate controls.")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # AUDIO SPOOF DETECTION (HAV-DF Model Integration)
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def predict_audio(self, audio_features: np.ndarray) -> Dict[str, Any]:
        """
        Predict if audio is real or fake (deepfake/spoof)
        
        Uses the HAV-DF model architecture for audio-only detection.
        The model expects mel spectrogram input of shape [1, 64, 32].
        
        Args:
            audio_features: Mel spectrogram features (normalized)
            
        Returns:
            Prediction result with confidence and severity
        """
        start_time = datetime.now()
        
        checkpoint = self.model_loader.get_model('audio_checkpoint')
        
        prediction = "Unknown"
        confidence = 0.5
        is_fake = False
        fake_prob = 0.5
        real_prob = 0.5
        
        if checkpoint is not None and TORCH_AVAILABLE:
            try:
                # Get the AudioLCNN model from checkpoint
                # The HAV-DF checkpoint contains 'model_state' with full model weights
                # We can use just the audio stream for audio-only detection
                
                if audio_features is not None and len(audio_features) > 0:
                    # Prepare mel spectrogram for AudioLCNN
                    # Expected input: [B, 1, 64, 32] (batch, channel, mel_bins, time_steps)
                    
                    mel_tensor = self._prepare_audio_tensor(audio_features)
                    
                    if mel_tensor is not None:
                        # Try to use the full model's audio stream
                        model_state = checkpoint.get('model_state', checkpoint.get('model_state_dict'))
                        
                        if model_state is not None:
                            # Extract audio stream weights and create AudioLCNN
                            audio_model = self._create_audio_model()
                            
                            if audio_model is not None:
                                # Load only audio-related weights
                                audio_state = {
                                    k.replace('audio_stream.', ''): v 
                                    for k, v in model_state.items() 
                                    if k.startswith('audio_stream.')
                                }
                                
                                if audio_state:
                                    try:
                                        audio_model.load_state_dict(audio_state, strict=False)
                                        audio_model.eval()
                                        
                                        with torch.no_grad():
                                            # Get audio embeddings
                                            audio_emb = audio_model(mel_tensor)
                                            
                                            # Simple classifier on audio embeddings
                                            # Use embedding statistics for classification
                                            emb_np = audio_emb.cpu().numpy().flatten()
                                            
                                            # Audio deepfakes often have different embedding patterns
                                            emb_mean = np.mean(emb_np)
                                            emb_std = np.std(emb_np)
                                            emb_max = np.max(np.abs(emb_np))
                                            
                                            # Heuristic scoring based on embedding characteristics
                                            # Real audio tends to have more varied embeddings
                                            score = 0.5
                                            
                                            # Low variance often indicates synthetic audio
                                            if emb_std < 0.3:
                                                score += 0.2
                                            elif emb_std > 0.7:
                                                score -= 0.1
                                            
                                            # Extreme values can indicate manipulation
                                            if emb_max > 2.0:
                                                score += 0.15
                                            
                                            # Normalize score to probability
                                            fake_prob = min(0.98, max(0.02, score))
                                            real_prob = 1.0 - fake_prob
                                            
                                            is_fake = fake_prob > 0.5
                                            confidence = fake_prob if is_fake else real_prob
                                            
                                    except Exception as e:
                                        print(f"[WARN] Audio model inference failed: {e}")
                                        # Fall back to statistical analysis
                                        fake_prob, confidence = self._statistical_audio_analysis(audio_features)
                                        is_fake = fake_prob > 0.5
                                else:
                                    # No audio weights found, use statistical analysis
                                    fake_prob, confidence = self._statistical_audio_analysis(audio_features)
                                    is_fake = fake_prob > 0.5
                            else:
                                # Couldn't create model, use statistical analysis
                                fake_prob, confidence = self._statistical_audio_analysis(audio_features)
                                is_fake = fake_prob > 0.5
                        else:
                            # No model state, use statistical analysis
                            fake_prob, confidence = self._statistical_audio_analysis(audio_features)
                            is_fake = fake_prob > 0.5
                    else:
                        # Couldn't prepare tensor, use statistical analysis
                        fake_prob, confidence = self._statistical_audio_analysis(audio_features)
                        is_fake = fake_prob > 0.5
                else:
                    # No features provided
                    fake_prob = 0.5
                    confidence = 0.5
                    is_fake = False
                    
            except Exception as e:
                print(f"[WARN] Audio prediction failed: {e}")
                # Fallback to statistical analysis
                if audio_features is not None:
                    fake_prob, confidence = self._statistical_audio_analysis(audio_features)
                    is_fake = fake_prob > 0.5
        else:
            # No model available, use statistical analysis
            if audio_features is not None:
                fake_prob, confidence = self._statistical_audio_analysis(audio_features)
                is_fake = fake_prob > 0.5
        
        prediction = "Fake" if is_fake else "Real"
        severity = "CRITICAL" if is_fake and confidence > 0.9 else \
                   "HIGH" if is_fake and confidence > 0.7 else \
                   "MEDIUM" if is_fake else "LOW"
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return {
            "prediction": prediction,
            "is_fake": is_fake,
            "confidence": round(confidence, 4),
            "fake_probability": round(fake_prob, 4),
            "real_probability": round(real_prob, 4),
            "severity": severity,
            "recommendation": self._get_audio_recommendation(is_fake, confidence),
            "processing_time_ms": round(processing_time, 2)
        }
    
    def _prepare_audio_tensor(self, audio_features: np.ndarray) -> Optional[Any]:
        """Prepare audio features as PyTorch tensor for model input"""
        if not TORCH_AVAILABLE:
            return None
        
        try:
            # Target shape for AudioLCNN: [1, 1, 64, 32]
            TARGET_MELS = 64
            TARGET_TIME = 32
            
            features = np.array(audio_features, dtype=np.float32)
            
            # Handle different input shapes
            if features.ndim == 1:
                # Flat array - try to reshape
                total = features.size
                if total >= TARGET_MELS * TARGET_TIME:
                    features = features[:TARGET_MELS * TARGET_TIME].reshape(TARGET_MELS, TARGET_TIME)
                else:
                    # Pad and reshape
                    padded = np.zeros(TARGET_MELS * TARGET_TIME, dtype=np.float32)
                    padded[:total] = features
                    features = padded.reshape(TARGET_MELS, TARGET_TIME)
            elif features.ndim == 2:
                # Already 2D, resize to target shape using scipy or simple interpolation
                if CV2_AVAILABLE:
                    features = cv2.resize(features, (TARGET_TIME, TARGET_MELS), 
                                          interpolation=cv2.INTER_LINEAR)
                else:
                    # Simple resize using numpy
                    features = self._simple_resize_2d(features, TARGET_MELS, TARGET_TIME)
            elif features.ndim == 3:
                # Take mean across first dimension
                features = features.mean(axis=0)
                if CV2_AVAILABLE:
                    features = cv2.resize(features, (TARGET_TIME, TARGET_MELS),
                                          interpolation=cv2.INTER_LINEAR)
                else:
                    features = self._simple_resize_2d(features, TARGET_MELS, TARGET_TIME)
            
            # Normalize to [-1, 1] range
            if features.max() > 1.0 or features.min() < -1.0:
                features = features / (np.abs(features).max() + 1e-8)
            
            # Create tensor: [1, 1, 64, 32]
            tensor = torch.from_numpy(features).unsqueeze(0).unsqueeze(0)
            
            return tensor
            
        except Exception as e:
            print(f"[WARN] Audio tensor preparation failed: {e}")
            return None
    
    def _simple_resize_2d(self, arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        """Simple 2D array resize using linear interpolation (no cv2 dependency)"""
        from scipy.ndimage import zoom
        
        h, w = arr.shape
        zoom_h = target_h / h
        zoom_w = target_w / w
        
        return zoom(arr, (zoom_h, zoom_w), order=1)
    
    def _create_audio_model(self) -> Optional[Any]:
        """Create AudioLCNN model architecture"""
        if not TORCH_AVAILABLE:
            return None
        
        try:
            import torch.nn as nn
            
            class MaxFeatureMap(nn.Module):
                def forward(self, x):
                    x1, x2 = x.chunk(2, dim=1)
                    return torch.max(x1, x2)
            
            class ConvBnMFM(nn.Module):
                def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch * 2, k, s, p, bias=False),
                        nn.BatchNorm2d(out_ch * 2),
                        MaxFeatureMap(),
                    )
                def forward(self, x):
                    return self.net(x)
            
            class AudioLCNN(nn.Module):
                """Lightweight LCNN for mel-spectrograms"""
                def __init__(self, out_dim: int = 256):
                    super().__init__()
                    self.features = nn.Sequential(
                        ConvBnMFM(1, 32, k=5, s=1, p=2),
                        nn.MaxPool2d(2, 2),
                        ConvBnMFM(32, 48, k=3, s=1, p=1),
                        nn.MaxPool2d(2, 2),
                        ConvBnMFM(48, 64, k=3, s=1, p=1),
                        ConvBnMFM(64, 64, k=3, s=1, p=1),
                        nn.MaxPool2d(2, 2),
                        ConvBnMFM(64, 96, k=3, s=1, p=1),
                    )
                    self.pool = nn.AdaptiveAvgPool2d((1, 1))
                    self.proj = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(96, out_dim),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.3),
                    )
                
                def forward(self, x):
                    return self.proj(self.pool(self.features(x)))
            
            return AudioLCNN(out_dim=256)
            
        except Exception as e:
            print(f"[WARN] Failed to create AudioLCNN: {e}")
            return None
    
    def _statistical_audio_analysis(self, audio_features: np.ndarray) -> Tuple[float, float]:
        """
        Fallback statistical analysis for audio deepfake detection
        
        Returns:
            Tuple of (fake_probability, confidence)
        """
        try:
            features = np.array(audio_features).flatten()
            
            # Statistical features
            mean_val = np.mean(features)
            std_val = np.std(features)
            max_val = np.max(np.abs(features))
            skewness = np.mean(((features - mean_val) / (std_val + 1e-8)) ** 3)
            kurtosis = np.mean(((features - mean_val) / (std_val + 1e-8)) ** 4) - 3
            
            # Scoring based on statistical anomalies
            score = 0.5
            
            # Synthetic audio often has lower variance
            if std_val < 0.15:
                score += 0.2
            elif std_val > 0.8:
                score += 0.1
            
            # Unusual skewness
            if abs(skewness) > 1.5:
                score += 0.1
            
            # Unusual kurtosis (heavy tails or flat distribution)
            if abs(kurtosis) > 3:
                score += 0.1
            
            # Very uniform values (common in synthetic)
            unique_ratio = len(np.unique(features[:1000])) / min(1000, len(features))
            if unique_ratio < 0.3:
                score += 0.15
            
            fake_prob = min(0.95, max(0.05, score))
            confidence = 0.6 + 0.2 * abs(fake_prob - 0.5)  # Higher confidence for extreme scores
            
            return fake_prob, confidence
            
        except Exception as e:
            print(f"[WARN] Statistical analysis failed: {e}")
            return 0.5, 0.5
    
    def _get_audio_recommendation(self, is_fake: bool, confidence: float) -> str:
        """Get recommendation for audio detection"""
        if is_fake:
            if confidence > 0.9:
                return "High confidence deepfake detected. Do not trust this audio. Verify speaker identity through alternative channels."
            elif confidence > 0.7:
                return "Likely synthetic audio. Exercise caution. Request video call or in-person verification."
            else:
                return "Possible audio manipulation. Consider additional verification steps."
        else:
            return "Audio appears authentic. Standard verification procedures apply."
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HUMAN THREAT (PHISHING) DETECTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def predict_phishing(
        self, 
        email_text: str,
        subject: Optional[str] = None,
        sender: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Detect phishing attempts in email content
        
        Args:
            email_text: Email body content
            subject: Email subject line
            sender: Sender email address
            
        Returns:
            Phishing detection result
        """
        start_time = datetime.now()
        
        # Combine all text for analysis
        full_text = email_text
        if subject:
            full_text = f"{subject}\n{full_text}"
        
        text_lower = full_text.lower()
        
        # Detect indicators
        detected_indicators = []
        total_score = 0
        indicator_types = {}
        
        for indicator in self.PHISHING_INDICATORS:
            if re.search(indicator["pattern"], text_lower, re.IGNORECASE):
                match = re.search(indicator["pattern"], text_lower, re.IGNORECASE)
                detected_indicators.append(match.group())
                total_score += indicator["weight"]
                indicator_types[indicator["type"]] = indicator_types.get(indicator["type"], 0) + 1
        
        # Additional checks
        # Check for suspicious links
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, full_text)
        suspicious_urls = [u for u in urls if self._is_suspicious_url(u)]
        if suspicious_urls:
            total_score += 15 * len(suspicious_urls)
            detected_indicators.extend([f"Suspicious URL: {u[:50]}..." for u in suspicious_urls[:3]])
        
        # Check sender domain
        if sender:
            if self._is_suspicious_sender(sender):
                total_score += 20
                detected_indicators.append(f"Suspicious sender: {sender}")
        
        # Calculate confidence
        confidence = min(0.98, total_score / 100)
        is_phishing = confidence > 0.3
        
        # Determine threat type
        threat_type = "Benign"
        if is_phishing:
            if indicator_types.get("financial_fraud", 0) > 0:
                threat_type = "Financial Fraud / BEC"
            elif indicator_types.get("credential_theft", 0) > 0:
                threat_type = "Credential Phishing"
            elif indicator_types.get("urgency", 0) > 1:
                threat_type = "Urgency Scam"
            else:
                threat_type = "Social Engineering"
        
        # Determine severity
        if confidence > 0.8:
            severity = "CRITICAL"
        elif confidence > 0.6:
            severity = "HIGH"
        elif confidence > 0.3:
            severity = "MEDIUM"
        else:
            severity = "LOW"
        
        # Build risk indicators
        risk_indicators = {
            "urgency_score": indicator_types.get("urgency", 0) * 0.2,
            "credential_theft_score": indicator_types.get("credential_theft", 0) * 0.25,
            "financial_fraud_score": indicator_types.get("financial_fraud", 0) * 0.3,
            "suspicious_links": len(suspicious_urls),
            "total_indicators": len(detected_indicators)
        }
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return {
            "phishing": is_phishing,
            "confidence": round(confidence, 4),
            "severity": severity,
            "threat_type": threat_type,
            "highlighted_phrases": list(set(detected_indicators))[:10],
            "risk_indicators": risk_indicators,
            "recommendation": self._get_phishing_recommendation(is_phishing, threat_type, confidence),
            "processing_time_ms": round(processing_time, 2)
        }
    
    def _is_suspicious_url(self, url: str) -> bool:
        """Check if URL is suspicious"""
        suspicious_patterns = [
            r"bit\.ly", r"tinyurl", r"goo\.gl",  # URL shorteners
            r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",  # IP addresses
            r"login|signin|verify|secure|account|update",  # Phishing keywords
            r"\.ru$|\.cn$|\.tk$|\.ml$",  # Suspicious TLDs
        ]
        url_lower = url.lower()
        return any(re.search(p, url_lower) for p in suspicious_patterns)
    
    def _is_suspicious_sender(self, sender: str) -> bool:
        """Check if sender is suspicious"""
        suspicious_patterns = [
            r"@.*\d{3,}",  # Numbers in domain
            r"@.*(secure|verify|alert|update|support).*\.",  # Phishing keywords
            r"@.*\.(ru|cn|tk|ml|ga)$",  # Suspicious TLDs
        ]
        return any(re.search(p, sender.lower()) for p in suspicious_patterns)
    
    def _get_phishing_recommendation(self, is_phishing: bool, threat_type: str, confidence: float) -> str:
        """Get recommendation for phishing detection"""
        if not is_phishing:
            return "Email appears legitimate. Standard caution advised."
        
        recommendations = {
            "Financial Fraud / BEC": "URGENT: Do not process any financial requests. Verify through official channels. Report to security team immediately.",
            "Credential Phishing": "Do not click any links. Do not enter credentials. Report to IT security. If credentials were entered, change passwords immediately.",
            "Urgency Scam": "Ignore urgency tactics. Verify sender through official channels. Do not take immediate action.",
            "Social Engineering": "Exercise caution. Verify sender identity. Do not share sensitive information."
        }
        
        base_rec = recommendations.get(threat_type, "Treat with caution. Verify sender before taking action.")
        
        if confidence > 0.8:
            return f"HIGH RISK: {base_rec}"
        return base_rec

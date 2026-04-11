"""
ARIHANT SOC - Model Loader Service
==================================
Handles loading and management of ML models at startup
"""

import os
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ML Libraries
import numpy as np

# Try importing ML libraries
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    print("[WARN] joblib not available")

try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[WARN] TensorFlow not available")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARN] PyTorch not available")


class ModelLoader:
    """
    Centralized model loading and management
    
    Loads models once at startup and provides access throughout the application.
    Supports:
    - Scikit-learn models (.pkl via joblib)
    - TensorFlow/Keras models (.h5)
    - PyTorch models (.pt)
    """
    
    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.loaded_models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.encoders: Dict[str, Any] = {}
        self.columns: Dict[str, Any] = {}
        self.load_times: Dict[str, float] = {}
        self._initialized = False
    
    async def load_all_models(self):
        """Load all models asynchronously"""
        if self._initialized:
            return
        
        print(f"\n[INFO] Loading models from: {self.models_dir}")
        
        # Run model loading in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_models_sync)
        
        self._initialized = True
    
    def _load_models_sync(self):
        """Synchronous model loading"""
        
        # ═══════════════════════════════════════════════════════════════════
        # NETWORK LAYER MODELS
        # ═══════════════════════════════════════════════════════════════════
        
        # LSTM Model for Network Intrusion Detection
        nids_lstm_path = self.models_dir / "nids_lstm_model.h5"
        if nids_lstm_path.exists() and TF_AVAILABLE:
            try:
                start = datetime.now()
                self.loaded_models['network_lstm'] = tf.keras.models.load_model(
                    str(nids_lstm_path), 
                    compile=False
                )
                self.load_times['network_lstm'] = (datetime.now() - start).total_seconds()
                print(f"  ✓ Network LSTM model loaded ({self.load_times['network_lstm']:.2f}s)")
            except Exception as e:
                print(f"  ✗ Failed to load Network LSTM: {e}")
        
        # Random Forest Model (alternative)
        rf_path = self.models_dir / "rf_model.pkl"
        if rf_path.exists() and JOBLIB_AVAILABLE:
            try:
                start = datetime.now()
                self.loaded_models['network_rf'] = joblib.load(rf_path)
                self.load_times['network_rf'] = (datetime.now() - start).total_seconds()
                print(f"  ✓ Network RF model loaded ({self.load_times['network_rf']:.2f}s)")
            except Exception as e:
                print(f"  ✗ Failed to load Network RF: {e}")
        
        # Network Scaler
        nids_scaler_path = self.models_dir / "nids_scaler.pkl"
        if nids_scaler_path.exists() and JOBLIB_AVAILABLE:
            try:
                self.scalers['network'] = joblib.load(nids_scaler_path)
                print(f"  ✓ Network scaler loaded")
            except Exception as e:
                print(f"  ✗ Failed to load Network scaler: {e}")
        
        # Network Encoders
        for encoder_name in ['IPV4_SRC_ADDR_encoder.pkl', 'IPV4_DST_ADDR_encoder.pkl', 'attack_encoder.pkl']:
            encoder_path = self.models_dir / encoder_name
            if encoder_path.exists() and JOBLIB_AVAILABLE:
                try:
                    key = encoder_name.replace('.pkl', '').replace('_encoder', '')
                    self.encoders[key] = joblib.load(encoder_path)
                    print(f"  ✓ Encoder '{key}' loaded")
                except Exception as e:
                    print(f"  ✗ Failed to load encoder {encoder_name}: {e}")
        
        # ═══════════════════════════════════════════════════════════════════
        # APPLICATION LAYER MODELS
        # ═══════════════════════════════════════════════════════════════════
        
        app_lstm_path = self.models_dir / "app_layer_lstm_model.h5"
        if app_lstm_path.exists() and TF_AVAILABLE:
            try:
                start = datetime.now()
                self.loaded_models['application_lstm'] = tf.keras.models.load_model(
                    str(app_lstm_path),
                    compile=False
                )
                self.load_times['application_lstm'] = (datetime.now() - start).total_seconds()
                print(f"  ✓ Application LSTM model loaded ({self.load_times['application_lstm']:.2f}s)")
            except Exception as e:
                print(f"  ✗ Failed to load Application LSTM: {e}")
        
        # Application Scaler
        app_scaler_path = self.models_dir / "app_layer_scaler.pkl"
        if app_scaler_path.exists() and JOBLIB_AVAILABLE:
            try:
                self.scalers['application'] = joblib.load(app_scaler_path)
                print(f"  ✓ Application scaler loaded")
            except Exception as e:
                print(f"  ✗ Failed to load Application scaler: {e}")
        
        # Application Columns
        app_columns_path = self.models_dir / "app_layer_columns.pkl"
        if app_columns_path.exists() and JOBLIB_AVAILABLE:
            try:
                self.columns['application'] = joblib.load(app_columns_path)
                print(f"  ✓ Application columns loaded ({len(self.columns['application'])} features)")
            except Exception as e:
                print(f"  ✗ Failed to load Application columns: {e}")
        
        # ═══════════════════════════════════════════════════════════════════
        # AUDIO SPOOF DETECTION MODEL (HAV-DF)
        # ═══════════════════════════════════════════════════════════════════
        
        audio_model_path = self.models_dir / "best_model.pt"
        if audio_model_path.exists() and TORCH_AVAILABLE:
            try:
                start = datetime.now()
                # Load PyTorch model
                checkpoint = torch.load(
                    str(audio_model_path), 
                    map_location=torch.device('cpu'),
                    weights_only=False
                )
                self.loaded_models['audio_checkpoint'] = checkpoint
                self.load_times['audio'] = (datetime.now() - start).total_seconds()
                print(f"  ✓ Audio spoof model loaded ({self.load_times['audio']:.2f}s)")
            except Exception as e:
                print(f"  ✗ Failed to load Audio model: {e}")
        
        # ═══════════════════════════════════════════════════════════════════
        # LABEL ENCODERS
        # ═══════════════════════════════════════════════════════════════════
        
        label_encoders_path = self.models_dir / "label_encoders.pkl"
        if label_encoders_path.exists() and JOBLIB_AVAILABLE:
            try:
                self.encoders['labels'] = joblib.load(label_encoders_path)
                print(f"  ✓ Label encoders loaded")
            except Exception as e:
                print(f"  ✗ Failed to load label encoders: {e}")
        
        # Train columns
        train_columns_path = self.models_dir / "train_columns.pkl"
        if train_columns_path.exists() and JOBLIB_AVAILABLE:
            try:
                self.columns['train'] = joblib.load(train_columns_path)
                print(f"  ✓ Train columns loaded")
            except Exception as e:
                print(f"  ✗ Failed to load train columns: {e}")
    
    def get_model(self, name: str) -> Optional[Any]:
        """Get a loaded model by name"""
        return self.loaded_models.get(name)
    
    def get_scaler(self, name: str) -> Optional[Any]:
        """Get a loaded scaler by name"""
        return self.scalers.get(name)
    
    def get_encoder(self, name: str) -> Optional[Any]:
        """Get a loaded encoder by name"""
        return self.encoders.get(name)
    
    def get_columns(self, name: str) -> Optional[Any]:
        """Get column list by name"""
        return self.columns.get(name)
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all loaded models"""
        return {
            "models_loaded": list(self.loaded_models.keys()),
            "scalers_loaded": list(self.scalers.keys()),
            "encoders_loaded": list(self.encoders.keys()),
            "columns_loaded": list(self.columns.keys()),
            "load_times": self.load_times,
            "tensorflow_available": TF_AVAILABLE,
            "pytorch_available": TORCH_AVAILABLE,
            "joblib_available": JOBLIB_AVAILABLE
        }
    
    def cleanup(self):
        """Cleanup resources"""
        self.loaded_models.clear()
        self.scalers.clear()
        self.encoders.clear()
        self.columns.clear()
        
        # Clear TensorFlow session if available
        if TF_AVAILABLE:
            tf.keras.backend.clear_session()
        
        print("[INFO] Model resources cleaned up")

"""
ARIHANT SOC - Response Schemas
==============================
Pydantic models for API responses
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class SeverityLevel(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class AttackLayer(str, Enum):
    NETWORK = "NETWORK"
    APPLICATION = "APPLICATION"
    AUDIO = "AUDIO"
    HUMAN = "HUMAN"


# ═══════════════════════════════════════════════════════════════════════════════
# BASE RESPONSE
# ═══════════════════════════════════════════════════════════════════════════════

class BaseResponse(BaseModel):
    """Base response with common fields"""
    success: bool = True
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    processing_time_ms: Optional[float] = None


class ErrorResponse(BaseModel):
    """Error response schema"""
    success: bool = False
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ═══════════════════════════════════════════════════════════════════════════════
# NETWORK DETECTION RESPONSE
# ═══════════════════════════════════════════════════════════════════════════════

class NetworkDetectionResponse(BaseResponse):
    """Response for network intrusion detection"""
    attack_type: str = Field(..., description="Detected attack type")
    attack_label: int = Field(..., description="Numeric attack label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    severity: SeverityLevel = Field(..., description="Threat severity")
    is_attack: bool = Field(..., description="Whether traffic is malicious")
    probabilities: Optional[Dict[str, float]] = Field(
        None, 
        description="Class probabilities"
    )
    recommendation: str = Field(..., description="Recommended action")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "timestamp": "2024-01-15T10:30:00Z",
                "processing_time_ms": 45.2,
                "attack_type": "DDoS",
                "attack_label": 3,
                "confidence": 0.97,
                "severity": "CRITICAL",
                "is_attack": True,
                "probabilities": {"Benign": 0.03, "DDoS": 0.97},
                "recommendation": "Enable rate limiting and block source IP"
            }
        }


class NetworkBatchResponse(BaseResponse):
    """Response for batch network detection"""
    results: List[NetworkDetectionResponse]
    total_analyzed: int
    attacks_detected: int
    attack_summary: Dict[str, int]


# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION DETECTION RESPONSE
# ═══════════════════════════════════════════════════════════════════════════════

class ApplicationDetectionResponse(BaseResponse):
    """Response for application layer attack detection"""
    attack_type: str = Field(..., description="Detected attack type")
    confidence: float = Field(..., ge=0.0, le=1.0)
    severity: SeverityLevel
    is_attack: bool
    attack_category: Optional[str] = Field(None, description="Attack category")
    indicators: Optional[List[str]] = Field(
        None, 
        description="Detected malicious indicators"
    )
    recommendation: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "timestamp": "2024-01-15T10:30:00Z",
                "processing_time_ms": 38.5,
                "attack_type": "SQL Injection",
                "confidence": 0.92,
                "severity": "HIGH",
                "is_attack": True,
                "attack_category": "Injection",
                "indicators": ["OR 1=1", "UNION SELECT", "--"],
                "recommendation": "Block request and review input validation"
            }
        }


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO DETECTION RESPONSE
# ═══════════════════════════════════════════════════════════════════════════════

class AudioDetectionResponse(BaseResponse):
    """Response for audio spoof detection"""
    prediction: str = Field(..., description="Real or Fake")
    is_fake: bool = Field(..., description="Whether audio is synthetic")
    confidence: float = Field(..., ge=0.0, le=1.0)
    severity: SeverityLevel
    audio_features: Optional[Dict[str, Any]] = Field(
        None,
        description="Extracted audio features"
    )
    recommendation: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "timestamp": "2024-01-15T10:30:00Z",
                "processing_time_ms": 125.8,
                "prediction": "Fake",
                "is_fake": True,
                "confidence": 0.95,
                "severity": "HIGH",
                "audio_features": {
                    "duration_seconds": 5.2,
                    "sample_rate": 16000
                },
                "recommendation": "Flag as potential deepfake, verify speaker identity"
            }
        }


# ═══════════════════════════════════════════════════════════════════════════════
# HUMAN THREAT RESPONSE
# ═══════════════════════════════════════════════════════════════════════════════

class HumanThreatResponse(BaseResponse):
    """Response for human threat (phishing) detection"""
    phishing: bool = Field(..., description="Is phishing attempt")
    confidence: float = Field(..., ge=0.0, le=1.0)
    severity: SeverityLevel
    threat_type: str = Field(..., description="Type of social engineering")
    highlighted_phrases: List[str] = Field(
        ..., 
        description="Suspicious phrases detected"
    )
    risk_indicators: Optional[Dict[str, Any]] = Field(
        None,
        description="Risk indicator breakdown"
    )
    recommendation: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "timestamp": "2024-01-15T10:30:00Z",
                "processing_time_ms": 28.3,
                "phishing": True,
                "confidence": 0.89,
                "severity": "HIGH",
                "threat_type": "Credential Phishing",
                "highlighted_phrases": ["urgent", "click here", "verify your account"],
                "risk_indicators": {
                    "urgency_score": 0.85,
                    "suspicious_links": 2,
                    "impersonation_score": 0.72
                },
                "recommendation": "Quarantine email and alert user"
            }
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ALERT RESPONSE
# ═══════════════════════════════════════════════════════════════════════════════

class AlertResponse(BaseModel):
    """Single alert response"""
    id: str
    attack_type: str
    severity: SeverityLevel
    confidence: float
    source: AttackLayer
    timestamp: datetime
    resolved: bool = False
    details: Optional[Dict[str, Any]] = None
    source_ip: Optional[str] = None
    notes: Optional[str] = None
    assigned_to: Optional[str] = None
    report_id: Optional[str] = None  # Link to threat intelligence report
    contextual_insight: Optional[str] = None  # Quick insight from report


class AlertListResponse(BaseResponse):
    """Response for alert listing"""
    alerts: List[AlertResponse]
    total: int
    page: int
    limit: int
    has_more: bool


class AlertCreateResponse(BaseResponse):
    """Response for alert creation"""
    alert: AlertResponse
    message: str = "Alert created successfully"


class AlertStatsResponse(BaseResponse):
    """Response for alert statistics"""
    total_alerts: int
    by_severity: Dict[str, int]
    by_source: Dict[str, int]
    resolved_count: int
    unresolved_count: int
    last_24h_count: int


# ═══════════════════════════════════════════════════════════════════════════════
# WEBSOCKET MESSAGE
# ═══════════════════════════════════════════════════════════════════════════════

class WebSocketAlert(BaseModel):
    """WebSocket alert message format"""
    type: str = "alert"
    data: AlertResponse

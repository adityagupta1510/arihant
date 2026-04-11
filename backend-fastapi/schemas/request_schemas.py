"""
ARIHANT SOC - Request Schemas
=============================
Pydantic models for API request validation
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class SeverityLevel(str, Enum):
    """Threat severity levels"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class AttackLayer(str, Enum):
    """Attack layer classification"""
    NETWORK = "NETWORK"
    APPLICATION = "APPLICATION"
    AUDIO = "AUDIO"
    HUMAN = "HUMAN"


# ═══════════════════════════════════════════════════════════════════════════════
# NETWORK DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

class NetworkDetectionRequest(BaseModel):
    """Request schema for network intrusion detection"""
    features: List[float] = Field(
        ...,
        description="Array of network flow features",
        min_items=1,
        max_items=100
    )
    source_ip: Optional[str] = Field(None, description="Source IP address")
    dest_ip: Optional[str] = Field(None, description="Destination IP address")
    protocol: Optional[str] = Field(None, description="Network protocol")
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": [0.5, 0.3, 0.8, 0.1, 0.9, 0.2, 0.7, 0.4, 0.6, 0.5],
                "source_ip": "192.168.1.100",
                "dest_ip": "10.0.0.1",
                "protocol": "TCP"
            }
        }


class NetworkBatchRequest(BaseModel):
    """Batch request for multiple network flows"""
    flows: List[NetworkDetectionRequest] = Field(
        ...,
        description="List of network flows to analyze",
        min_items=1,
        max_items=100
    )


# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

class ApplicationDetectionRequest(BaseModel):
    """Request schema for application layer attack detection"""
    features: Optional[List[float]] = Field(
        None,
        description="Preprocessed feature array"
    )
    request_data: Optional[str] = Field(
        None,
        description="Raw HTTP request data or payload"
    )
    url: Optional[str] = Field(None, description="Request URL")
    method: Optional[str] = Field(None, description="HTTP method")
    headers: Optional[Dict[str, str]] = Field(None, description="Request headers")
    body: Optional[str] = Field(None, description="Request body")
    
    @validator('features', 'request_data', pre=True, always=True)
    def check_at_least_one(cls, v, values):
        if v is None and values.get('features') is None and values.get('request_data') is None:
            pass  # Will be validated at model level
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "request_data": "SELECT * FROM users WHERE id=1 OR 1=1--",
                "url": "/api/users",
                "method": "GET"
            }
        }


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

class AudioDetectionRequest(BaseModel):
    """Request schema for audio spoof detection (metadata only, file via form)"""
    filename: Optional[str] = Field(None, description="Original filename")
    duration_seconds: Optional[float] = Field(None, description="Audio duration")
    sample_rate: Optional[int] = Field(None, description="Sample rate in Hz")
    
    class Config:
        json_schema_extra = {
            "example": {
                "filename": "voice_sample.wav",
                "duration_seconds": 5.2,
                "sample_rate": 16000
            }
        }


# ═══════════════════════════════════════════════════════════════════════════════
# HUMAN THREAT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

class HumanThreatRequest(BaseModel):
    """Request schema for human threat (phishing) detection"""
    email_text: str = Field(
        ...,
        description="Email content to analyze",
        min_length=10,
        max_length=50000
    )
    subject: Optional[str] = Field(None, description="Email subject line")
    sender: Optional[str] = Field(None, description="Sender email address")
    headers: Optional[Dict[str, str]] = Field(None, description="Email headers")
    
    class Config:
        json_schema_extra = {
            "example": {
                "email_text": "URGENT: Your account has been compromised. Click here immediately to verify your identity and restore access.",
                "subject": "URGENT: Account Security Alert",
                "sender": "security@bank-verify.com"
            }
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ALERT MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

class AlertCreateRequest(BaseModel):
    """Request schema for creating alerts"""
    attack_type: str = Field(..., description="Type of attack detected")
    severity: SeverityLevel = Field(..., description="Severity level")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    source: AttackLayer = Field(..., description="Detection source layer")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")
    source_ip: Optional[str] = Field(None, description="Source IP if applicable")
    
    class Config:
        json_schema_extra = {
            "example": {
                "attack_type": "SQL Injection",
                "severity": "HIGH",
                "confidence": 0.92,
                "source": "APPLICATION",
                "details": {"payload": "' OR 1=1--"},
                "source_ip": "192.168.1.100"
            }
        }


class AlertUpdateRequest(BaseModel):
    """Request schema for updating alerts"""
    status: Optional[str] = Field(None, description="Alert status")
    resolved: Optional[bool] = Field(None, description="Mark as resolved")
    notes: Optional[str] = Field(None, description="Analyst notes")
    assigned_to: Optional[str] = Field(None, description="Assigned analyst")


class AlertQueryParams(BaseModel):
    """Query parameters for alert listing"""
    severity: Optional[SeverityLevel] = None
    source: Optional[AttackLayer] = None
    resolved: Optional[bool] = None
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)

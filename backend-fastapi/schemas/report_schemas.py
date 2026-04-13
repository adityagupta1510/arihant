"""
ARIHANT SOC - Report Schemas
============================
Pydantic models for threat report generation API
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class ReportSeverity(str, Enum):
    """Report severity levels"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ReportSource(str, Enum):
    """Detection source layers"""
    NETWORK = "network"
    APPLICATION = "application"
    AUDIO = "audio"
    HUMAN = "human"


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════

class ReportGenerateRequest(BaseModel):
    """Request schema for generating a threat report"""
    attack_type: str = Field(
        ...,
        description="Type of attack detected (e.g., DDoS, SQL Injection, Phishing)",
        min_length=1,
        max_length=100
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Detection confidence score (0.0 to 1.0)"
    )
    severity: ReportSeverity = Field(
        ...,
        description="Severity level of the threat"
    )
    source: Optional[ReportSource] = Field(
        default=ReportSource.NETWORK,
        description="Detection source layer"
    )
    additional_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional context about the threat (source IP, payload, etc.)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "attack_type": "DDoS",
                "confidence": 0.97,
                "severity": "HIGH",
                "source": "network",
                "additional_context": {
                    "source_ip": "192.168.1.100",
                    "target_port": 443,
                    "packets_per_second": 50000
                }
            }
        }


class ReportBatchRequest(BaseModel):
    """Request schema for generating multiple reports"""
    threats: List[ReportGenerateRequest] = Field(
        ...,
        description="List of threats to generate reports for",
        min_items=1,
        max_items=20
    )


# ═══════════════════════════════════════════════════════════════════════════════
# RESPONSE SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════

class AttackDetails(BaseModel):
    """Attack details in the report"""
    type: str
    description: str
    impact: str
    source_layer: str


class RiskInfo(BaseModel):
    """Risk level information"""
    level: str
    tone: str
    urgency: str
    color: str
    icon: str


class AIEnhanced(BaseModel):
    """AI-enhanced explanations"""
    simple_explanation: Optional[str] = None
    business_impact: Optional[str] = None
    executive_advice: Optional[str] = None
    raw_explanation: Optional[str] = None


class ThreatReportResponse(BaseModel):
    """Complete threat report response"""
    title: str = Field(..., description="Report title")
    summary: str = Field(..., description="Executive summary of the threat")
    attack_details: AttackDetails = Field(..., description="Detailed attack information")
    risk_level: str = Field(..., description="Risk severity level")
    risk_info: RiskInfo = Field(..., description="Risk level metadata")
    confidence_score: str = Field(..., description="Confidence as percentage string")
    confidence_value: float = Field(..., description="Raw confidence value")
    precautions: List[str] = Field(..., description="Preventive measures")
    non_technical_advice: List[str] = Field(..., description="Advice for non-technical users")
    recommended_actions: List[str] = Field(..., description="Recommended response actions")
    detection_source: str = Field(..., description="Source of detection")
    timestamp: str = Field(..., description="Report generation timestamp")
    report_id: str = Field(..., description="Unique report identifier")
    additional_context: Optional[Dict[str, Any]] = Field(None, description="Additional threat context")
    ai_enhanced: Optional[AIEnhanced] = Field(None, description="AI-enhanced explanations")
    is_generic: Optional[bool] = Field(False, description="Whether this is a generic report")
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "Distributed Denial of Service (DDoS) Attack Detected",
                "summary": "A potential cyber attack has been identified that may disrupt system availability.",
                "attack_details": {
                    "type": "DDoS",
                    "description": "A DDoS attack attempts to overwhelm a system with traffic.",
                    "impact": "This can slow down or crash your system.",
                    "source_layer": "NETWORK"
                },
                "risk_level": "HIGH",
                "risk_info": {
                    "level": "HIGH",
                    "tone": "warning",
                    "urgency": "Urgent attention needed",
                    "color": "#FF8C00",
                    "icon": "⚠️"
                },
                "confidence_score": "97%",
                "confidence_value": 0.97,
                "precautions": ["Monitor unusual spikes in network traffic"],
                "non_technical_advice": ["Contact your technical team if website becomes slow"],
                "recommended_actions": ["Block suspicious IP addresses"],
                "detection_source": "network",
                "timestamp": "2024-01-15T10:30:00Z",
                "report_id": "RPT-20240115103000-A1B2C3D4"
            }
        }


class ReportGenerateResponse(BaseModel):
    """API response wrapper for report generation"""
    success: bool = Field(..., description="Whether the request was successful")
    timestamp: datetime = Field(..., description="Response timestamp")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    report: ThreatReportResponse = Field(..., description="Generated threat report")


class ReportBatchResponse(BaseModel):
    """API response for batch report generation"""
    success: bool
    timestamp: datetime
    processing_time_ms: float
    total_reports: int
    reports: List[ThreatReportResponse]


class AttackTypesResponse(BaseModel):
    """Response for supported attack types"""
    success: bool
    attack_types: List[str]
    total: int


class SeverityLevelsResponse(BaseModel):
    """Response for severity level information"""
    success: bool
    severity_levels: Dict[str, Any]

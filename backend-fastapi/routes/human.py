"""
ARIHANT SOC - Human Threat Detection Routes
===========================================
API endpoints for phishing and social engineering detection

Auto-triggers threat intelligence reports on detection
"""

from fastapi import APIRouter, Request, HTTPException
from typing import Optional
from datetime import datetime

from schemas.request_schemas import HumanThreatRequest
from schemas.response_schemas import HumanThreatResponse
from services.prediction_service import PredictionService
from services.elastic_service import elastic_service

router = APIRouter()


@router.post(
    "/detect",
    response_model=HumanThreatResponse,
    summary="Detect Phishing/Social Engineering",
    description="Analyze email content to detect phishing attempts and social engineering"
)
async def detect_phishing(
    request: Request,
    data: HumanThreatRequest
):
    """
    Detect phishing and social engineering attempts in email content
    
    - **email_text**: Email body content (required)
    - **subject**: Email subject line (optional)
    - **sender**: Sender email address (optional)
    - **headers**: Email headers (optional)
    
    Returns phishing probability, threat type, highlighted phrases, and recommendations
    """
    model_loader = request.app.state.model_loader
    ws_manager = request.app.state.ws_manager
    
    if not model_loader:
        raise HTTPException(status_code=503, detail="Model service not available")
    
    prediction_service = PredictionService(model_loader)
    
    # Run prediction
    result = await prediction_service.predict_phishing(
        email_text=data.email_text,
        subject=data.subject,
        sender=data.sender
    )
    
    # Log to Elasticsearch
    await elastic_service.log_prediction(
        index=elastic_service.INDEX_HUMAN,
        input_data={
            "email_length": len(data.email_text),
            "has_subject": data.subject is not None,
            "has_sender": data.sender is not None,
            "sender_domain": data.sender.split("@")[1] if data.sender and "@" in data.sender else None
        },
        prediction=result
    )
    
    # AUTO-TRIGGER: Process through Threat Intelligence Engine if phishing detected
    intelligence_result = None
    if result["phishing"] and result["confidence"] > 0.6:
        threat_intelligence = request.app.state.threat_intelligence
        
        # Map threat type to attack type
        attack_type_map = {
            "Credential Phishing": "Phishing",
            "Financial Fraud / BEC": "BEC",
            "Urgency Scam": "Phishing",
            "Impersonation": "Phishing",
            "Malware Distribution": "Malware"
        }
        attack_type = attack_type_map.get(result["threat_type"], "Phishing")
        
        intelligence_result = await threat_intelligence.process_detection(
            attack_type=attack_type,
            confidence=result["confidence"],
            severity=result["severity"],
            source="human",
            sender_email=data.sender,
            email_subject=data.subject,
            threat_type=result["threat_type"],
            indicators_count=len(result["highlighted_phrases"]),
            highlighted_phrases=result["highlighted_phrases"][:5]  # Top 5 phrases
        )
    
    response = HumanThreatResponse(
        success=True,
        timestamp=datetime.utcnow(),
        processing_time_ms=result["processing_time_ms"],
        phishing=result["phishing"],
        confidence=result["confidence"],
        severity=result["severity"],
        threat_type=result["threat_type"],
        highlighted_phrases=result["highlighted_phrases"],
        risk_indicators=result.get("risk_indicators"),
        recommendation=result["recommendation"]
    )
    
    # Add intelligence data to response if available
    if intelligence_result:
        response_dict = response.model_dump()
        response_dict["intelligence"] = {
            "report_id": intelligence_result["report"]["report_id"],
            "alert_id": intelligence_result["alert"]["id"],
            "contextual_insight": intelligence_result["report"].get("contextual_insight"),
            "email_sent": intelligence_result.get("email_sent", False)
        }
        return response_dict
    
    return response


@router.post(
    "/analyze-quick",
    summary="Quick Phishing Analysis",
    description="Fast analysis of text for phishing indicators without full scoring"
)
async def quick_analyze(
    text: str
):
    """
    Quick analysis for phishing indicators
    
    Returns list of detected suspicious phrases without full confidence scoring
    """
    import re
    
    # Quick pattern matching
    indicators = []
    text_lower = text.lower()
    
    quick_patterns = [
        (r"\burgent\b", "Urgency language"),
        (r"click here", "Suspicious call-to-action"),
        (r"verify your account", "Account verification request"),
        (r"\bpassword\b", "Password mention"),
        (r"wire transfer", "Financial request"),
        (r"gift card", "Gift card scam indicator"),
        (r"act now", "Pressure tactic"),
        (r"limited time", "Artificial urgency"),
        (r"confirm your identity", "Identity verification request"),
        (r"unusual activity", "Fear tactic"),
        (r"account.*suspend", "Account threat"),
        (r"https?://\S+", "Contains links"),
    ]
    
    for pattern, description in quick_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            match = re.search(pattern, text_lower, re.IGNORECASE)
            indicators.append({
                "phrase": match.group(),
                "type": description
            })
    
    risk_level = "LOW"
    if len(indicators) >= 5:
        risk_level = "CRITICAL"
    elif len(indicators) >= 3:
        risk_level = "HIGH"
    elif len(indicators) >= 1:
        risk_level = "MEDIUM"
    
    return {
        "success": True,
        "timestamp": datetime.utcnow().isoformat(),
        "indicators_found": len(indicators),
        "risk_level": risk_level,
        "indicators": indicators,
        "recommendation": "Full analysis recommended" if indicators else "No obvious indicators found"
    }


@router.post(
    "/analyze-sender",
    summary="Analyze Email Sender",
    description="Analyze sender email address for suspicious patterns"
)
async def analyze_sender(
    sender: str
):
    """
    Analyze email sender for suspicious patterns
    
    - **sender**: Email address to analyze
    """
    import re
    
    indicators = []
    risk_score = 0
    
    # Extract domain
    domain = sender.split("@")[1] if "@" in sender else sender
    
    # Check for suspicious patterns
    checks = [
        (r"\d{3,}", "Contains multiple numbers", 15),
        (r"(secure|verify|alert|update|support|admin|help)", "Contains suspicious keywords", 20),
        (r"\.(ru|cn|tk|ml|ga|cf)$", "Suspicious TLD", 25),
        (r"[^a-zA-Z0-9@._-]", "Contains unusual characters", 10),
        (r"^[a-z]{20,}@", "Unusually long local part", 10),
        (r"(paypal|amazon|microsoft|google|apple|bank)", "Impersonation attempt", 30),
    ]
    
    for pattern, description, score in checks:
        if re.search(pattern, sender.lower()):
            indicators.append(description)
            risk_score += score
    
    # Check if domain looks like typosquatting
    legitimate_domains = ["gmail.com", "outlook.com", "yahoo.com", "hotmail.com"]
    for legit in legitimate_domains:
        if domain != legit and domain.replace(".", "").replace("-", "") in legit.replace(".", ""):
            indicators.append("Possible typosquatting")
            risk_score += 25
    
    risk_level = "LOW"
    if risk_score >= 50:
        risk_level = "CRITICAL"
    elif risk_score >= 30:
        risk_level = "HIGH"
    elif risk_score >= 15:
        risk_level = "MEDIUM"
    
    return {
        "success": True,
        "timestamp": datetime.utcnow().isoformat(),
        "sender": sender,
        "domain": domain,
        "risk_score": min(100, risk_score),
        "risk_level": risk_level,
        "indicators": indicators,
        "recommendation": "Block sender" if risk_level in ["CRITICAL", "HIGH"] else "Exercise caution" if risk_level == "MEDIUM" else "Appears legitimate"
    }


@router.get(
    "/stats",
    summary="Human Threat Detection Statistics",
    description="Get statistics for phishing detections"
)
async def get_human_stats():
    """Get human threat detection statistics"""
    stats = await elastic_service.get_stats(
        elastic_service.INDEX_HUMAN,
        time_range="24h"
    )
    
    return {
        "success": True,
        "timestamp": datetime.utcnow().isoformat(),
        "stats": stats
    }


@router.get(
    "/history",
    summary="Human Threat Detection History",
    description="Get recent phishing detection history"
)
async def get_human_history(
    limit: int = 50,
    time_range: str = "24h"
):
    """Get human threat detection history"""
    logs = await elastic_service.search_logs(
        elastic_service.INDEX_HUMAN,
        time_range=time_range,
        size=limit
    )
    
    return {
        "success": True,
        "timestamp": datetime.utcnow().isoformat(),
        "count": len(logs),
        "logs": logs
    }


@router.get(
    "/threat-types",
    summary="Get Threat Types",
    description="List of detectable social engineering threat types"
)
async def get_threat_types():
    """Get list of detectable threat types"""
    return {
        "success": True,
        "threat_types": [
            {
                "type": "Credential Phishing",
                "description": "Attempts to steal login credentials",
                "indicators": ["verify your account", "password expired", "login required"],
                "severity": "HIGH"
            },
            {
                "type": "Financial Fraud / BEC",
                "description": "Business Email Compromise and wire fraud",
                "indicators": ["wire transfer", "urgent payment", "gift card"],
                "severity": "CRITICAL"
            },
            {
                "type": "Urgency Scam",
                "description": "Creates artificial urgency to bypass rational thinking",
                "indicators": ["act now", "limited time", "immediate action"],
                "severity": "MEDIUM"
            },
            {
                "type": "Impersonation",
                "description": "Pretends to be trusted entity",
                "indicators": ["security team", "IT department", "CEO"],
                "severity": "HIGH"
            },
            {
                "type": "Malware Distribution",
                "description": "Attempts to deliver malicious attachments/links",
                "indicators": ["download attachment", "enable macros", "click to view"],
                "severity": "CRITICAL"
            }
        ]
    }

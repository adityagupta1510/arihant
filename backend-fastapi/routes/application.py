"""
ARIHANT SOC - Application Detection Routes
==========================================
API endpoints for application layer attack detection
"""

from fastapi import APIRouter, Request, HTTPException
from typing import Optional
from datetime import datetime

from schemas.request_schemas import ApplicationDetectionRequest
from schemas.response_schemas import ApplicationDetectionResponse
from services.prediction_service import PredictionService
from services.elastic_service import elastic_service

router = APIRouter()


@router.post(
    "/detect",
    response_model=ApplicationDetectionResponse,
    summary="Detect Application Layer Attack",
    description="Analyze request data to detect SQL injection, XSS, command injection, etc."
)
async def detect_application_attack(
    request: Request,
    data: ApplicationDetectionRequest
):
    """
    Detect application layer attacks
    
    - **features**: Preprocessed feature array (optional)
    - **request_data**: Raw HTTP request data or payload (optional)
    - **url**: Request URL (optional)
    - **method**: HTTP method (optional)
    - **headers**: Request headers (optional)
    - **body**: Request body (optional)
    
    At least one of features or request_data should be provided.
    
    Returns attack type, confidence, severity, indicators, and recommendations
    """
    model_loader = request.app.state.model_loader
    ws_manager = request.app.state.ws_manager
    
    if not model_loader:
        raise HTTPException(status_code=503, detail="Model service not available")
    
    # Validate input
    if data.features is None and data.request_data is None:
        raise HTTPException(
            status_code=400, 
            detail="Either 'features' or 'request_data' must be provided"
        )
    
    prediction_service = PredictionService(model_loader)
    
    # Combine request data for analysis
    combined_data = data.request_data or ""
    if data.url:
        combined_data = f"{data.url}\n{combined_data}"
    if data.body:
        combined_data = f"{combined_data}\n{data.body}"
    
    # Run prediction
    result = await prediction_service.predict_application(
        features=data.features,
        request_data=combined_data if combined_data else None
    )
    
    # Log to Elasticsearch
    await elastic_service.log_prediction(
        index=elastic_service.INDEX_APPLICATION,
        input_data={
            "url": data.url,
            "method": data.method,
            "has_features": data.features is not None,
            "request_data_length": len(data.request_data) if data.request_data else 0
        },
        prediction=result
    )
    
    # Broadcast alert if attack detected with high confidence
    if result["is_attack"] and result["confidence"] > 0.7:
        import uuid
        await ws_manager.broadcast_alert(
            alert_id=str(uuid.uuid4()),
            attack_type=result["attack_type"],
            severity=result["severity"],
            confidence=result["confidence"],
            source="APPLICATION",
            details={
                "url": data.url,
                "method": data.method,
                "indicators": result.get("indicators", [])
            }
        )
    
    return ApplicationDetectionResponse(
        success=True,
        timestamp=datetime.utcnow(),
        processing_time_ms=result["processing_time_ms"],
        attack_type=result["attack_type"],
        confidence=result["confidence"],
        severity=result["severity"],
        is_attack=result["is_attack"],
        attack_category=result.get("attack_category"),
        indicators=result.get("indicators"),
        recommendation=result["recommendation"]
    )


@router.post(
    "/analyze-payload",
    summary="Analyze Request Payload",
    description="Quick analysis of a request payload for common attack patterns"
)
async def analyze_payload(
    request: Request,
    payload: str
):
    """
    Quick payload analysis without full model inference
    
    Useful for real-time WAF-like analysis
    """
    model_loader = request.app.state.model_loader
    
    if not model_loader:
        raise HTTPException(status_code=503, detail="Model service not available")
    
    prediction_service = PredictionService(model_loader)
    
    # Use pattern-based detection only
    result = prediction_service._detect_app_patterns(payload)
    
    return {
        "success": True,
        "timestamp": datetime.utcnow().isoformat(),
        "is_attack": result["is_attack"],
        "attack_type": result["attack_type"],
        "confidence": result["confidence"],
        "indicators": result["indicators"],
        "recommendation": prediction_service._get_app_recommendation(result["attack_type"])
    }


@router.get(
    "/stats",
    summary="Application Detection Statistics",
    description="Get statistics for application layer detections"
)
async def get_application_stats():
    """Get application detection statistics"""
    stats = await elastic_service.get_stats(
        elastic_service.INDEX_APPLICATION,
        time_range="24h"
    )
    
    return {
        "success": True,
        "timestamp": datetime.utcnow().isoformat(),
        "stats": stats
    }


@router.get(
    "/history",
    summary="Application Detection History",
    description="Get recent application detection history"
)
async def get_application_history(
    limit: int = 50,
    time_range: str = "24h"
):
    """Get application detection history"""
    logs = await elastic_service.search_logs(
        elastic_service.INDEX_APPLICATION,
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
    "/attack-patterns",
    summary="Get Known Attack Patterns",
    description="List of attack patterns the system can detect"
)
async def get_attack_patterns():
    """Get list of detectable attack patterns"""
    return {
        "success": True,
        "patterns": {
            "sql_injection": {
                "description": "SQL Injection attacks",
                "examples": ["' OR 1=1--", "UNION SELECT", "; DROP TABLE"],
                "severity": "HIGH"
            },
            "xss": {
                "description": "Cross-Site Scripting attacks",
                "examples": ["<script>", "javascript:", "onerror="],
                "severity": "HIGH"
            },
            "command_injection": {
                "description": "OS Command Injection",
                "examples": ["; ls", "| cat /etc/passwd", "`whoami`"],
                "severity": "CRITICAL"
            },
            "path_traversal": {
                "description": "Directory Traversal attacks",
                "examples": ["../", "..\\", "%2e%2e"],
                "severity": "MEDIUM"
            }
        }
    }

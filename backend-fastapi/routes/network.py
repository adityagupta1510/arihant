"""
ARIHANT SOC - Network Detection Routes
======================================
API endpoints for network intrusion detection
"""

from fastapi import APIRouter, Request, HTTPException
from typing import List
from datetime import datetime

from schemas.request_schemas import NetworkDetectionRequest, NetworkBatchRequest
from schemas.response_schemas import NetworkDetectionResponse, NetworkBatchResponse
from services.prediction_service import PredictionService
from services.elastic_service import elastic_service

router = APIRouter()


@router.post(
    "/detect",
    response_model=NetworkDetectionResponse,
    summary="Detect Network Intrusion",
    description="Analyze network flow features to detect potential intrusions using LSTM model"
)
async def detect_network_intrusion(
    request: Request,
    data: NetworkDetectionRequest
):
    """
    Detect network intrusion from flow features
    
    - **features**: Array of network flow features (required)
    - **source_ip**: Source IP address (optional)
    - **dest_ip**: Destination IP address (optional)
    - **protocol**: Network protocol (optional)
    
    Returns attack type, confidence, severity, and recommendations
    """
    model_loader = request.app.state.model_loader
    ws_manager = request.app.state.ws_manager
    
    if not model_loader:
        raise HTTPException(status_code=503, detail="Model service not available")
    
    prediction_service = PredictionService(model_loader)
    
    # Run prediction
    result = await prediction_service.predict_network(data.features)
    
    # Log to Elasticsearch
    await elastic_service.log_prediction(
        index=elastic_service.INDEX_NETWORK,
        input_data={
            "features_count": len(data.features),
            "source_ip": data.source_ip,
            "dest_ip": data.dest_ip,
            "protocol": data.protocol
        },
        prediction=result,
        source_ip=data.source_ip
    )
    
    # Broadcast alert if attack detected with high confidence
    if result["is_attack"] and result["confidence"] > 0.7:
        import uuid
        await ws_manager.broadcast_alert(
            alert_id=str(uuid.uuid4()),
            attack_type=result["attack_type"],
            severity=result["severity"],
            confidence=result["confidence"],
            source="NETWORK",
            details={
                "source_ip": data.source_ip,
                "dest_ip": data.dest_ip,
                "protocol": data.protocol
            }
        )
    
    return NetworkDetectionResponse(
        success=True,
        timestamp=datetime.utcnow(),
        processing_time_ms=result["processing_time_ms"],
        attack_type=result["attack_type"],
        attack_label=result["attack_label"],
        confidence=result["confidence"],
        severity=result["severity"],
        is_attack=result["is_attack"],
        probabilities=result.get("probabilities"),
        recommendation=result["recommendation"]
    )


@router.post(
    "/detect/batch",
    response_model=NetworkBatchResponse,
    summary="Batch Network Detection",
    description="Analyze multiple network flows in a single request"
)
async def detect_network_batch(
    request: Request,
    data: NetworkBatchRequest
):
    """
    Batch detection for multiple network flows
    
    - **flows**: List of network flow objects to analyze
    
    Returns aggregated results with attack summary
    """
    model_loader = request.app.state.model_loader
    
    if not model_loader:
        raise HTTPException(status_code=503, detail="Model service not available")
    
    prediction_service = PredictionService(model_loader)
    
    results = []
    attack_summary = {}
    attacks_detected = 0
    
    start_time = datetime.now()
    
    for flow in data.flows:
        result = await prediction_service.predict_network(flow.features)
        
        results.append(NetworkDetectionResponse(
            success=True,
            timestamp=datetime.utcnow(),
            processing_time_ms=result["processing_time_ms"],
            attack_type=result["attack_type"],
            attack_label=result["attack_label"],
            confidence=result["confidence"],
            severity=result["severity"],
            is_attack=result["is_attack"],
            probabilities=result.get("probabilities"),
            recommendation=result["recommendation"]
        ))
        
        if result["is_attack"]:
            attacks_detected += 1
            attack_type = result["attack_type"]
            attack_summary[attack_type] = attack_summary.get(attack_type, 0) + 1
    
    total_time = (datetime.now() - start_time).total_seconds() * 1000
    
    return NetworkBatchResponse(
        success=True,
        timestamp=datetime.utcnow(),
        processing_time_ms=round(total_time, 2),
        results=results,
        total_analyzed=len(data.flows),
        attacks_detected=attacks_detected,
        attack_summary=attack_summary
    )


@router.get(
    "/stats",
    summary="Network Detection Statistics",
    description="Get statistics for network detections in the last 24 hours"
)
async def get_network_stats():
    """Get network detection statistics"""
    stats = await elastic_service.get_stats(
        elastic_service.INDEX_NETWORK,
        time_range="24h"
    )
    
    return {
        "success": True,
        "timestamp": datetime.utcnow().isoformat(),
        "stats": stats
    }


@router.get(
    "/history",
    summary="Network Detection History",
    description="Get recent network detection history"
)
async def get_network_history(
    limit: int = 50,
    time_range: str = "24h"
):
    """Get network detection history"""
    logs = await elastic_service.search_logs(
        elastic_service.INDEX_NETWORK,
        time_range=time_range,
        size=limit
    )
    
    return {
        "success": True,
        "timestamp": datetime.utcnow().isoformat(),
        "count": len(logs),
        "logs": logs
    }

"""
ARIHANT SOC - Alert Management Routes
=====================================
API endpoints for alert creation, management, and querying
"""

from fastapi import APIRouter, Request, HTTPException, Query
from typing import Optional, List
from datetime import datetime, timedelta
from enum import Enum
import uuid

from schemas.request_schemas import AlertCreateRequest, AlertUpdateRequest, SeverityLevel, AttackLayer
from schemas.response_schemas import (
    AlertResponse, AlertListResponse, AlertCreateResponse, AlertStatsResponse
)
from services.elastic_service import elastic_service

router = APIRouter()


class AlertStore:
    """In-memory alert store with Elasticsearch backup"""
    
    def __init__(self):
        self.alerts: dict = {}
        self._total_count = 0
    
    def create(self, alert_data: dict) -> AlertResponse:
        """Create a new alert"""
        alert_id = str(uuid.uuid4())
        
        alert = AlertResponse(
            id=alert_id,
            attack_type=alert_data["attack_type"],
            severity=alert_data["severity"],
            confidence=alert_data["confidence"],
            source=alert_data["source"],
            timestamp=datetime.utcnow(),
            resolved=False,
            details=alert_data.get("details"),
            source_ip=alert_data.get("source_ip")
        )
        
        self.alerts[alert_id] = alert
        self._total_count += 1
        
        return alert
    
    def get(self, alert_id: str) -> Optional[AlertResponse]:
        """Get alert by ID"""
        return self.alerts.get(alert_id)
    
    def update(self, alert_id: str, updates: dict) -> Optional[AlertResponse]:
        """Update an alert"""
        if alert_id not in self.alerts:
            return None
        
        alert = self.alerts[alert_id]
        
        for key, value in updates.items():
            if value is not None and hasattr(alert, key):
                setattr(alert, key, value)
        
        return alert
    
    def delete(self, alert_id: str) -> bool:
        """Delete an alert"""
        if alert_id in self.alerts:
            del self.alerts[alert_id]
            return True
        return False
    
    def list_alerts(
        self,
        severity: Optional[str] = None,
        source: Optional[str] = None,
        resolved: Optional[bool] = None,
        limit: int = 50,
        offset: int = 0
    ) -> tuple:
        """List alerts with filtering"""
        filtered = list(self.alerts.values())
        
        if severity:
            filtered = [a for a in filtered if a.severity == severity]
        if source:
            filtered = [a for a in filtered if a.source == source]
        if resolved is not None:
            filtered = [a for a in filtered if a.resolved == resolved]
        
        # Sort by timestamp descending
        filtered.sort(key=lambda x: x.timestamp, reverse=True)
        
        total = len(filtered)
        paginated = filtered[offset:offset + limit]
        
        return paginated, total
    
    def get_stats(self) -> dict:
        """Get alert statistics"""
        alerts = list(self.alerts.values())
        
        by_severity = {}
        by_source = {}
        resolved_count = 0
        last_24h = 0
        
        cutoff = datetime.utcnow() - timedelta(hours=24)
        
        for alert in alerts:
            # By severity
            sev = alert.severity
            by_severity[sev] = by_severity.get(sev, 0) + 1
            
            # By source
            src = alert.source
            by_source[src] = by_source.get(src, 0) + 1
            
            # Resolved
            if alert.resolved:
                resolved_count += 1
            
            # Last 24h
            if alert.timestamp >= cutoff:
                last_24h += 1
        
        return {
            "total_alerts": len(alerts),
            "by_severity": by_severity,
            "by_source": by_source,
            "resolved_count": resolved_count,
            "unresolved_count": len(alerts) - resolved_count,
            "last_24h_count": last_24h
        }
    
    def get_total_count(self) -> int:
        """Get total alerts ever created"""
        return self._total_count


# Global alert store
alert_store = AlertStore()


@router.post(
    "/create",
    response_model=AlertCreateResponse,
    summary="Create Alert",
    description="Create a new security alert"
)
async def create_alert(
    request: Request,
    data: AlertCreateRequest
):
    """
    Create a new security alert
    
    - **attack_type**: Type of attack detected
    - **severity**: Severity level (CRITICAL, HIGH, MEDIUM, LOW)
    - **confidence**: Detection confidence (0-1)
    - **source**: Detection source layer
    - **details**: Additional details (optional)
    - **source_ip**: Source IP if applicable (optional)
    """
    ws_manager = request.app.state.ws_manager
    
    alert = alert_store.create({
        "attack_type": data.attack_type,
        "severity": data.severity.value,
        "confidence": data.confidence,
        "source": data.source.value,
        "details": data.details,
        "source_ip": data.source_ip
    })
    
    # Log to Elasticsearch
    await elastic_service.log_alert({
        "id": alert.id,
        "attack_type": alert.attack_type,
        "severity": alert.severity,
        "confidence": alert.confidence,
        "source": alert.source,
        "source_ip": alert.source_ip,
        "details": alert.details
    })
    
    # Broadcast via WebSocket
    await ws_manager.broadcast_alert(
        alert_id=alert.id,
        attack_type=alert.attack_type,
        severity=alert.severity,
        confidence=alert.confidence,
        source=alert.source,
        details=alert.details
    )
    
    return AlertCreateResponse(
        success=True,
        timestamp=datetime.utcnow(),
        alert=alert,
        message="Alert created and broadcast successfully"
    )


@router.get(
    "",
    response_model=AlertListResponse,
    summary="List Alerts",
    description="Get list of alerts with optional filtering"
)
async def list_alerts(
    severity: Optional[SeverityLevel] = Query(None, description="Filter by severity"),
    source: Optional[AttackLayer] = Query(None, description="Filter by source"),
    resolved: Optional[bool] = Query(None, description="Filter by resolved status"),
    limit: int = Query(50, ge=1, le=500, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
):
    """
    List alerts with optional filtering
    
    - **severity**: Filter by severity level
    - **source**: Filter by detection source
    - **resolved**: Filter by resolved status
    - **limit**: Maximum number of results
    - **offset**: Pagination offset
    """
    alerts, total = alert_store.list_alerts(
        severity=severity.value if severity else None,
        source=source.value if source else None,
        resolved=resolved,
        limit=limit,
        offset=offset
    )
    
    return AlertListResponse(
        success=True,
        timestamp=datetime.utcnow(),
        alerts=alerts,
        total=total,
        page=offset // limit + 1,
        limit=limit,
        has_more=offset + limit < total
    )


@router.get(
    "/stats",
    response_model=AlertStatsResponse,
    summary="Alert Statistics",
    description="Get alert statistics and breakdown"
)
async def get_alert_stats():
    """Get alert statistics"""
    stats = alert_store.get_stats()
    
    return AlertStatsResponse(
        success=True,
        timestamp=datetime.utcnow(),
        **stats
    )


@router.get(
    "/{alert_id}",
    response_model=AlertResponse,
    summary="Get Alert",
    description="Get a specific alert by ID"
)
async def get_alert(alert_id: str):
    """Get alert by ID"""
    alert = alert_store.get(alert_id)
    
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    return alert


@router.patch(
    "/{alert_id}",
    response_model=AlertResponse,
    summary="Update Alert",
    description="Update an existing alert"
)
async def update_alert(
    alert_id: str,
    data: AlertUpdateRequest
):
    """
    Update an alert
    
    - **status**: New status
    - **resolved**: Mark as resolved
    - **notes**: Add analyst notes
    - **assigned_to**: Assign to analyst
    """
    updates = data.model_dump(exclude_unset=True)
    
    alert = alert_store.update(alert_id, updates)
    
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    return alert


@router.post(
    "/{alert_id}/resolve",
    response_model=AlertResponse,
    summary="Resolve Alert",
    description="Mark an alert as resolved"
)
async def resolve_alert(
    alert_id: str,
    notes: Optional[str] = None
):
    """Mark alert as resolved"""
    updates = {"resolved": True}
    if notes:
        updates["notes"] = notes
    
    alert = alert_store.update(alert_id, updates)
    
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    return alert


@router.delete(
    "/{alert_id}",
    summary="Delete Alert",
    description="Delete an alert"
)
async def delete_alert(alert_id: str):
    """Delete an alert"""
    success = alert_store.delete(alert_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    return {
        "success": True,
        "message": f"Alert {alert_id} deleted"
    }


@router.get(
    "/recent/critical",
    summary="Recent Critical Alerts",
    description="Get recent critical and high severity alerts"
)
async def get_critical_alerts(limit: int = 10):
    """Get recent critical alerts"""
    critical, _ = alert_store.list_alerts(severity="CRITICAL", limit=limit)
    high, _ = alert_store.list_alerts(severity="HIGH", limit=limit)
    
    combined = critical + high
    combined.sort(key=lambda x: x.timestamp, reverse=True)
    
    return {
        "success": True,
        "timestamp": datetime.utcnow().isoformat(),
        "alerts": combined[:limit],
        "critical_count": len(critical),
        "high_count": len(high)
    }


@router.post(
    "/bulk/resolve",
    summary="Bulk Resolve Alerts",
    description="Resolve multiple alerts at once"
)
async def bulk_resolve(alert_ids: List[str]):
    """Resolve multiple alerts"""
    resolved = []
    failed = []
    
    for alert_id in alert_ids:
        alert = alert_store.update(alert_id, {"resolved": True})
        if alert:
            resolved.append(alert_id)
        else:
            failed.append(alert_id)
    
    return {
        "success": True,
        "resolved_count": len(resolved),
        "failed_count": len(failed),
        "resolved": resolved,
        "failed": failed
    }

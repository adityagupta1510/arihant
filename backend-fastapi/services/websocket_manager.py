"""
ARIHANT SOC - WebSocket Manager
===============================
Manages WebSocket connections for real-time threat intelligence
"""

from typing import List, Dict, Any, Optional
from fastapi import WebSocket
import json
from datetime import datetime
import asyncio


class ConnectionManager:
    """
    Manages WebSocket connections for real-time threat intelligence broadcasting
    
    Supports:
    - Alert broadcasts
    - Full threat intelligence reports
    - System status updates
    - Client subscriptions
    """
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._stats = {
            "total_connections": 0,
            "total_messages_sent": 0,
            "total_alerts_broadcast": 0,
            "total_reports_broadcast": 0
        }
    
    @property
    def active_connections_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)
    
    async def connect(self, websocket: WebSocket):
        """Accept and store new WebSocket connection"""
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
            self._stats["total_connections"] += 1
        
        print(f"[WS] Client connected. Total: {self.active_connections_count}")
        
        # Send welcome message with capabilities
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "message": "Connected to ARIHANT SOC Threat Intelligence Stream",
            "capabilities": [
                "threat_intelligence",
                "alerts",
                "reports",
                "system_status"
            ],
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"[WS] Client disconnected. Total: {self.active_connections_count}")
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return
        
        # Add timestamp if not present
        if "timestamp" not in message:
            message["timestamp"] = datetime.utcnow().isoformat()
        
        disconnected = []
        
        async with self._lock:
            for connection in self.active_connections:
                try:
                    await connection.send_json(message)
                    self._stats["total_messages_sent"] += 1
                except Exception as e:
                    print(f"[WS] Failed to send to client: {e}")
                    disconnected.append(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)
    
    async def broadcast_alert(
        self,
        alert_id: str,
        attack_type: str,
        severity: str,
        confidence: float,
        source: str,
        details: Dict[str, Any] = None,
        report_summary: Optional[Dict[str, Any]] = None
    ):
        """
        Broadcast a threat alert to all connected clients
        
        Args:
            alert_id: Unique alert identifier
            attack_type: Type of attack detected
            severity: Severity level (CRITICAL, HIGH, MEDIUM, LOW)
            confidence: Detection confidence (0-1)
            source: Detection source (NETWORK, APPLICATION, AUDIO, HUMAN)
            details: Additional alert details
            report_summary: Optional report summary to include
        """
        alert_message = {
            "type": "alert",
            "data": {
                "id": alert_id,
                "attack_type": attack_type,
                "severity": severity,
                "confidence": confidence,
                "source": source,
                "timestamp": datetime.utcnow().isoformat(),
                "details": details or {}
            }
        }
        
        # Include report summary if provided
        if report_summary:
            alert_message["data"]["report"] = report_summary
        
        await self.broadcast(alert_message)
        self._stats["total_alerts_broadcast"] += 1
        print(f"[WS] Alert broadcast: {attack_type} ({severity})")
    
    async def broadcast_threat_intelligence(
        self,
        alert: Dict[str, Any],
        report: Dict[str, Any]
    ):
        """
        Broadcast complete threat intelligence package
        
        This is the primary method for real-time threat intelligence delivery.
        Includes both alert and full report data.
        """
        message = {
            "type": "threat_intelligence",
            "data": {
                "alert": {
                    "id": alert.get("id"),
                    "attack_type": alert.get("attack_type"),
                    "severity": alert.get("severity"),
                    "confidence": alert.get("confidence"),
                    "source": alert.get("source"),
                    "timestamp": alert.get("timestamp"),
                    "status": alert.get("status", "NEW"),
                    "source_ip": alert.get("source_ip")
                },
                "report": {
                    "report_id": report.get("report_id"),
                    "title": report.get("title"),
                    "summary": report.get("summary"),
                    "contextual_insight": report.get("contextual_insight"),
                    "risk_level": report.get("risk_level"),
                    "risk_info": report.get("risk_info"),
                    "confidence_score": report.get("confidence_score"),
                    "attack_details": report.get("attack_details"),
                    "recommended_actions": report.get("recommended_actions", [])[:5],
                    "precautions": report.get("precautions", [])[:3],
                    "non_technical_advice": report.get("non_technical_advice", [])[:3],
                    "ai_enhanced": report.get("ai_enhanced"),
                    "context_data": report.get("context_data", {})
                }
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.broadcast(message)
        self._stats["total_reports_broadcast"] += 1
        print(f"[WS] Threat intelligence broadcast: {alert.get('attack_type')} ({alert.get('severity')})")
    
    async def broadcast_report_update(self, report: Dict[str, Any]):
        """Broadcast a report update (e.g., status change, resolution)"""
        message = {
            "type": "report_update",
            "data": {
                "report_id": report.get("report_id"),
                "alert_id": report.get("alert_id"),
                "status": report.get("status"),
                "title": report.get("title"),
                "risk_level": report.get("risk_level"),
                "updated_at": datetime.utcnow().isoformat()
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.broadcast(message)
    
    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """Send message to specific client"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            print(f"[WS] Failed to send personal message: {e}")
            self.disconnect(websocket)
    
    async def broadcast_system_status(self, status: str, details: Dict[str, Any] = None):
        """Broadcast system status update"""
        status_message = {
            "type": "system_status",
            "status": status,
            "details": details or {},
            "connections": self.active_connections_count,
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.broadcast(status_message)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get WebSocket statistics"""
        return {
            **self._stats,
            "active_connections": self.active_connections_count
        }

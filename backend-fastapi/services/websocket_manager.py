"""
ARIHANT SOC - WebSocket Manager
===============================
Manages WebSocket connections for real-time alerts
"""

from typing import List, Dict, Any
from fastapi import WebSocket
import json
from datetime import datetime
import asyncio


class ConnectionManager:
    """
    Manages WebSocket connections for real-time alert broadcasting
    """
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()
    
    @property
    def active_connections_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)
    
    async def connect(self, websocket: WebSocket):
        """Accept and store new WebSocket connection"""
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
        print(f"[WS] Client connected. Total connections: {self.active_connections_count}")
        
        # Send welcome message
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "message": "Connected to ARIHANT SOC Alert Stream",
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"[WS] Client disconnected. Total connections: {self.active_connections_count}")
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return
        
        # Add timestamp if not present
        if "timestamp" not in message:
            message["timestamp"] = datetime.utcnow().isoformat()
        
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
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
        details: Dict[str, Any] = None
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
        
        await self.broadcast(alert_message)
        print(f"[WS] Alert broadcast: {attack_type} ({severity})")
    
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
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.broadcast(status_message)

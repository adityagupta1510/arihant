"""
ARIHANT SOC - Threat Intelligence Engine
========================================
Real-time, auto-triggered intelligence system
Orchestrates: Detection → Report → Alert → Notification → UI

This is the central brain that automatically processes all detections
and generates comprehensive threat intelligence.
"""

import os
import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field

from services.report_service import report_service
from services.elastic_service import elastic_service


@dataclass
class ThreatEvent:
    """Represents a detected threat event"""
    event_id: str
    attack_type: str
    confidence: float
    severity: str
    source: str  # network, application, audio, human
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    
    @classmethod
    def create(
        cls,
        attack_type: str,
        confidence: float,
        severity: str,
        source: str,
        **context
    ) -> "ThreatEvent":
        return cls(
            event_id=f"EVT-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{str(uuid.uuid4())[:8].upper()}",
            attack_type=attack_type,
            confidence=confidence,
            severity=severity,
            source=source,
            context=context
        )


class ThreatIntelligenceEngine:
    """
    Central intelligence engine for automated threat processing
    
    Flow: Detection → Report → Alert → WebSocket → Email (optional)
    """
    
    def __init__(self):
        self._ws_manager = None
        self._alert_store = None
        self._email_service = None
        self._event_handlers: List[Callable] = []
        self._processing_queue: asyncio.Queue = asyncio.Queue()
        self._is_running = False
        
    def initialize(self, ws_manager, alert_store, email_service=None):
        """Initialize with required services"""
        self._ws_manager = ws_manager
        self._alert_store = alert_store
        self._email_service = email_service
        print("[TI] Threat Intelligence Engine initialized")
    
    def register_event_handler(self, handler: Callable):
        """Register a custom event handler"""
        self._event_handlers.append(handler)
    
    async def process_detection(
        self,
        attack_type: str,
        confidence: float,
        severity: str,
        source: str,
        source_ip: Optional[str] = None,
        target_port: Optional[int] = None,
        traffic_pattern: Optional[str] = None,
        payload_snippet: Optional[str] = None,
        affected_users: Optional[int] = None,
        **additional_context
    ) -> Dict[str, Any]:
        """
        Process a detection event through the full intelligence pipeline
        
        This is the main entry point called after any ML model detection.
        
        Args:
            attack_type: Type of attack detected
            confidence: Detection confidence (0.0 to 1.0)
            severity: Severity level (CRITICAL, HIGH, MEDIUM, LOW)
            source: Detection source (network, application, audio, human)
            source_ip: Source IP address if available
            target_port: Target port if available
            traffic_pattern: Traffic pattern description
            payload_snippet: Snippet of malicious payload
            affected_users: Number of affected users
            **additional_context: Any additional context data
        
        Returns:
            Complete intelligence package with report, alert, and status
        """
        # Build context
        context = {
            k: v for k, v in {
                "source_ip": source_ip,
                "target_port": target_port,
                "traffic_pattern": traffic_pattern,
                "payload_snippet": payload_snippet,
                "affected_users": affected_users,
                **additional_context
            }.items() if v is not None
        }
        
        # Create threat event
        event = ThreatEvent.create(
            attack_type=attack_type,
            confidence=confidence,
            severity=severity,
            source=source,
            **context
        )
        
        print(f"[TI] Processing threat: {attack_type} ({severity}) from {source}")
        
        # Step 1: Generate Intelligence Report
        report = await report_service.generate_contextual_report(
            attack_type=attack_type,
            confidence=confidence,
            severity=severity,
            source=source,
            context=context
        )
        
        # Step 2: Create Alert with Report Link
        alert = await self._create_alert(event, report)
        
        # Step 3: Link report to alert
        report["alert_id"] = alert["id"]
        await report_service.store.save(report)
        
        # Step 4: Broadcast via WebSocket (full report)
        await self._broadcast_intelligence(alert, report)
        
        # Step 5: Log to Elasticsearch
        await self._log_event(event, report, alert)
        
        # Step 6: Send Email Notification (if configured and severity warrants)
        email_sent = False
        if self._email_service and severity in ["CRITICAL", "HIGH"]:
            email_sent = await self._send_email_notification(report)
        
        # Step 7: Call custom event handlers
        for handler in self._event_handlers:
            try:
                await handler(event, report, alert)
            except Exception as e:
                print(f"[TI] Event handler error: {e}")
        
        # Return complete intelligence package
        return {
            "success": True,
            "event_id": event.event_id,
            "alert": alert,
            "report": report,
            "email_sent": email_sent,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    
    async def _create_alert(self, event: ThreatEvent, report: Dict) -> Dict[str, Any]:
        """Create an alert linked to the report"""
        alert_id = str(uuid.uuid4())
        
        alert_data = {
            "id": alert_id,
            "attack_type": event.attack_type,
            "severity": event.severity,
            "confidence": event.confidence,
            "source": event.source.upper(),
            "timestamp": event.timestamp,
            "resolved": False,
            "status": "NEW",
            "report_id": report["report_id"],
            "source_ip": event.context.get("source_ip"),
            "contextual_insight": report.get("contextual_insight"),
            "details": event.context
        }
        
        # Store in alert store if available
        if self._alert_store:
            from schemas.response_schemas import AlertResponse
            alert_response = AlertResponse(
                id=alert_id,
                attack_type=event.attack_type,
                severity=event.severity,
                confidence=event.confidence,
                source=event.source.upper(),
                timestamp=datetime.fromisoformat(event.timestamp.replace("Z", "")),
                resolved=False,
                details=event.context,
                source_ip=event.context.get("source_ip")
            )
            self._alert_store.alerts[alert_id] = alert_response
            self._alert_store._total_count += 1
        
        return alert_data
    
    async def _broadcast_intelligence(self, alert: Dict, report: Dict):
        """Broadcast full intelligence package via WebSocket"""
        if not self._ws_manager:
            return
        
        # Broadcast alert with embedded report summary
        message = {
            "type": "threat_intelligence",
            "data": {
                "alert": {
                    "id": alert["id"],
                    "attack_type": alert["attack_type"],
                    "severity": alert["severity"],
                    "confidence": alert["confidence"],
                    "source": alert["source"],
                    "timestamp": alert["timestamp"],
                    "status": alert["status"]
                },
                "report": {
                    "report_id": report["report_id"],
                    "title": report["title"],
                    "summary": report["summary"],
                    "contextual_insight": report.get("contextual_insight"),
                    "risk_level": report["risk_level"],
                    "risk_info": report["risk_info"],
                    "confidence_score": report["confidence_score"],
                    "recommended_actions": report["recommended_actions"][:3],  # Top 3 actions
                    "ai_enhanced": report.get("ai_enhanced")
                },
                "context": alert.get("details", {})
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self._ws_manager.broadcast(message)
        print(f"[TI] Intelligence broadcast: {alert['attack_type']} ({alert['severity']})")
    
    async def _log_event(self, event: ThreatEvent, report: Dict, alert: Dict):
        """Log the complete event to Elasticsearch"""
        index_map = {
            "network": elastic_service.INDEX_NETWORK,
            "application": elastic_service.INDEX_APPLICATION,
            "audio": elastic_service.INDEX_AUDIO,
            "human": elastic_service.INDEX_HUMAN
        }
        
        index = index_map.get(event.source.lower(), elastic_service.INDEX_ALERTS)
        
        await elastic_service.log_prediction(
            index=index,
            input_data=event.context,
            prediction={
                "attack_type": event.attack_type,
                "confidence": event.confidence,
                "severity": event.severity,
                "is_attack": True,
                "report_id": report["report_id"],
                "alert_id": alert["id"]
            },
            source_ip=event.context.get("source_ip")
        )
        
        # Also log to alerts index
        await elastic_service.log_alert({
            "id": alert["id"],
            "event_id": event.event_id,
            "attack_type": event.attack_type,
            "severity": event.severity,
            "confidence": event.confidence,
            "source": event.source,
            "report_id": report["report_id"],
            "contextual_insight": report.get("contextual_insight"),
            "details": event.context
        })
    
    async def _send_email_notification(self, report: Dict) -> bool:
        """Send email notification for critical/high severity threats"""
        if not self._email_service:
            return False
        
        try:
            return await self._email_service.send_threat_alert(report)
        except Exception as e:
            print(f"[TI] Email notification failed: {e}")
            return False
    
    async def get_recent_intelligence(self, limit: int = 20) -> List[Dict]:
        """Get recent threat intelligence reports"""
        return await report_service.store.list_recent(limit)
    
    async def get_intelligence_by_alert(self, alert_id: str) -> Optional[Dict]:
        """Get intelligence report for a specific alert"""
        return await report_service.store.get_by_alert(alert_id)


# Singleton instance
threat_intelligence = ThreatIntelligenceEngine()

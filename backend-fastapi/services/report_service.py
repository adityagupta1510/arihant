"""
ARIHANT SOC - Enhanced Threat Report Generation Service
=======================================================
Real-time, auto-triggered, intelligence-driven reporting engine
Converts raw ML output into clear, actionable, context-aware insights
"""

import os
import json
import httpx
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum
import uuid

# Attack templates for known threats
ATTACK_TEMPLATES = {
    "DDoS": {
        "full_name": "Distributed Denial of Service (DDoS) Attack",
        "description": "A DDoS attack attempts to overwhelm a system, server, or network with a flood of internet traffic from multiple sources, making it unavailable to legitimate users.",
        "impact": "This can slow down or completely crash your system, leading to service downtime, financial loss, poor user experience, and potential reputation damage.",
        "precautions": [
            "Monitor unusual spikes in network traffic",
            "Enable firewall and rate-limiting rules",
            "Use DDoS protection services like Cloudflare or AWS Shield",
            "Restrict unknown IP access",
            "Implement traffic analysis and anomaly detection"
        ],
        "non_technical_advice": [
            "If your website becomes slow or unresponsive, contact your technical team immediately",
            "Avoid sharing sensitive system access publicly",
            "Ensure your service providers are informed about possible traffic overload",
            "Have a communication plan ready for customers during outages"
        ],
        "recommended_actions": [
            "Block suspicious IP addresses immediately",
            "Activate DDoS mitigation tools",
            "Analyze traffic logs for abnormal patterns",
            "Scale up server resources temporarily",
            "Contact your ISP for upstream filtering"
        ]
    },
    "DoS": {
        "full_name": "Denial of Service (DoS) Attack",
        "description": "A DoS attack floods a system with traffic from a single source to exhaust resources and make services unavailable.",
        "impact": "Your services may become slow or completely unavailable, affecting business operations.",
        "precautions": ["Implement rate limiting", "Configure firewall rules", "Monitor server resources"],
        "non_technical_advice": ["Report unresponsive services to IT", "Document when issues started"],
        "recommended_actions": ["Block attacking IP", "Enable rate limiting", "Review firewall rules"]
    },
    "SQL Injection": {
        "full_name": "SQL Injection Attack",
        "description": "SQL Injection inserts malicious code into database queries through user input fields.",
        "impact": "Attackers could steal, modify, or delete database information including sensitive data.",
        "precautions": ["Use parameterized queries", "Validate all inputs", "Implement WAF"],
        "non_technical_advice": ["Report unusual data changes", "Ensure secure coding practices"],
        "recommended_actions": ["Patch vulnerable code", "Audit database logs", "Implement input validation"]
    },
    "XSS": {
        "full_name": "Cross-Site Scripting (XSS) Attack",
        "description": "XSS attacks inject malicious scripts into web pages viewed by other users.",
        "impact": "User accounts could be compromised and sensitive information stolen.",
        "precautions": ["Encode user content", "Implement CSP headers", "Use HTTP-only cookies"],
        "non_technical_advice": ["Report strange website behavior", "Don't click suspicious links"],
        "recommended_actions": ["Fix vulnerable code", "Scan for similar vulnerabilities", "Strengthen CSP"]
    },
    "Phishing": {
        "full_name": "Phishing Attack",
        "description": "Phishing tricks users into revealing sensitive information by impersonating trusted entities.",
        "impact": "Credentials could be stolen, leading to account takeovers and data breaches.",
        "precautions": ["Implement email authentication", "Train employees", "Use MFA"],
        "non_technical_advice": ["Never click unexpected links", "Verify sender identity", "Report suspicious emails"],
        "recommended_actions": ["Block sender domain", "Alert employees", "Reset affected passwords"]
    },
    "BEC": {
        "full_name": "Business Email Compromise (BEC)",
        "description": "BEC impersonates executives to trick employees into transferring money or data.",
        "impact": "Financial losses can be substantial and difficult to recover.",
        "precautions": ["Verify financial requests", "Use multi-person approval", "Train employees"],
        "non_technical_advice": ["Verify unusual requests by phone", "Be suspicious of urgent requests"],
        "recommended_actions": ["Halt suspicious transactions", "Contact bank", "Report to law enforcement"]
    },
    "Spoof Audio": {
        "full_name": "AI-Generated Audio Spoofing (Deepfake Voice)",
        "description": "Audio spoofing uses AI to clone voices for fraud or impersonation.",
        "impact": "Attackers could impersonate executives to authorize fraudulent transactions.",
        "precautions": ["Implement voice verification", "Use code words", "Deploy deepfake detection"],
        "non_technical_advice": ["Verify unusual voice requests", "Establish callback procedures"],
        "recommended_actions": ["Verify caller identity", "Preserve audio for analysis", "Review procedures"]
    },
    "Insider Threat": {
        "full_name": "Insider Threat",
        "description": "Insider threats come from individuals within the organization misusing their access.",
        "impact": "Data breaches, IP theft, and sabotage can cause significant damage.",
        "precautions": ["Implement least privilege", "Monitor user behavior", "Conduct access reviews"],
        "non_technical_advice": ["Report unusual colleague behavior", "Follow data handling policies"],
        "recommended_actions": ["Investigate access scope", "Preserve evidence", "Revoke access"]
    },
    "Brute Force": {
        "full_name": "Brute Force Attack",
        "description": "Brute force systematically tries password combinations to gain unauthorized access.",
        "impact": "Successful attacks lead to account compromise and data theft.",
        "precautions": ["Implement account lockout", "Use strong passwords", "Enable MFA"],
        "non_technical_advice": ["Use unique passwords", "Enable two-factor authentication"],
        "recommended_actions": ["Block attacking IPs", "Force password reset", "Review login logs"]
    },
    "Malware": {
        "full_name": "Malware Infection",
        "description": "Malware is malicious software designed to damage or gain unauthorized access.",
        "impact": "Systems can be damaged, data stolen, and operations disrupted.",
        "precautions": ["Keep software updated", "Use antivirus", "Train users"],
        "non_technical_advice": ["Don't download from untrusted sources", "Report unusual behavior"],
        "recommended_actions": ["Isolate infected systems", "Run malware scans", "Restore from backups"]
    },
    "Reconnaissance": {
        "full_name": "Network Reconnaissance",
        "description": "Reconnaissance is the initial phase where attackers gather information about targets.",
        "impact": "Successful reconnaissance enables more targeted and effective attacks.",
        "precautions": ["Minimize exposed services", "Monitor for scanning", "Use honeypots"],
        "non_technical_advice": ["Report unusual network activity", "Limit public information"],
        "recommended_actions": ["Block scanning IPs", "Review exposed services", "Update IDS signatures"]
    }
}

SEVERITY_TONES = {
    "CRITICAL": {"tone": "critical", "urgency": "IMMEDIATE ACTION REQUIRED", "color": "#FF3B3B", "icon": "🚨"},
    "HIGH": {"tone": "warning", "urgency": "Urgent attention needed", "color": "#FF8C00", "icon": "⚠️"},
    "MEDIUM": {"tone": "caution", "urgency": "Review and address soon", "color": "#FFD700", "icon": "📋"},
    "LOW": {"tone": "informational", "urgency": "Monitor and review", "color": "#00E0FF", "icon": "ℹ️"}
}


class ReportStore:
    """In-memory report storage with persistence capability"""
    
    def __init__(self):
        self.reports: Dict[str, Dict] = {}
        self._lock = asyncio.Lock()
    
    async def save(self, report: Dict[str, Any]) -> str:
        """Save a report and return its ID"""
        async with self._lock:
            report_id = report.get("report_id") or self._generate_id()
            report["report_id"] = report_id
            self.reports[report_id] = report
            
            # Keep only last 1000 reports
            if len(self.reports) > 1000:
                oldest_keys = sorted(self.reports.keys())[:100]
                for key in oldest_keys:
                    del self.reports[key]
            
            return report_id
    
    async def get(self, report_id: str) -> Optional[Dict[str, Any]]:
        """Get a report by ID"""
        return self.reports.get(report_id)
    
    async def get_by_alert(self, alert_id: str) -> Optional[Dict[str, Any]]:
        """Get report linked to an alert"""
        for report in self.reports.values():
            if report.get("alert_id") == alert_id:
                return report
        return None
    
    async def list_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent reports"""
        sorted_reports = sorted(
            self.reports.values(),
            key=lambda x: x.get("timestamp", ""),
            reverse=True
        )
        return sorted_reports[:limit]
    
    def _generate_id(self) -> str:
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        unique_id = str(uuid.uuid4())[:8].upper()
        return f"RPT-{timestamp}-{unique_id}"


class ReportService:
    """Enhanced service for AI-powered, context-aware threat reports"""
    
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.use_llm = bool(self.openai_api_key)
        self.store = ReportStore()
    
    async def generate_contextual_report(
        self,
        attack_type: str,
        confidence: float,
        severity: str,
        source: str,
        context: Optional[Dict[str, Any]] = None,
        alert_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a context-aware threat intelligence report
        
        Args:
            attack_type: Type of attack detected
            confidence: Detection confidence (0.0 to 1.0)
            severity: Severity level
            source: Detection source layer
            context: Contextual data (source_ip, target_port, traffic_pattern, etc.)
            alert_id: Associated alert ID for linking
        
        Returns:
            Complete threat intelligence report
        """
        attack_normalized = self._normalize_attack_type(attack_type)
        severity_upper = severity.upper()
        confidence_pct = f"{int(confidence * 100)}%"
        
        template = ATTACK_TEMPLATES.get(attack_normalized, {})
        severity_info = SEVERITY_TONES.get(severity_upper, SEVERITY_TONES["MEDIUM"])
        
        # Generate contextual insight
        contextual_insight = self._generate_contextual_insight(
            attack_normalized, context, severity_upper
        )
        
        # Build the report
        report = {
            "report_id": self._generate_report_id(),
            "alert_id": alert_id,
            "title": template.get("full_name", f"Security Threat: {attack_type}") + " Detected",
            "summary": self._build_summary(template, attack_normalized, confidence_pct, severity_upper, context),
            "contextual_insight": contextual_insight,
            "attack_details": {
                "type": attack_normalized,
                "description": template.get("description", f"A {attack_type} security threat has been detected."),
                "impact": template.get("impact", "Potential security impact requires investigation."),
                "source_layer": source.upper()
            },
            "risk_level": severity_upper,
            "risk_info": severity_info,
            "confidence_score": confidence_pct,
            "confidence_value": confidence,
            "context_data": context or {},
            "precautions": template.get("precautions", ["Monitor system activity", "Review security logs"]),
            "non_technical_advice": template.get("non_technical_advice", ["Report unusual activity to IT"]),
            "recommended_actions": self._get_contextual_actions(attack_normalized, context, template),
            "detection_source": source,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "status": "NEW",
            "auto_generated": True
        }
        
        # Enhance with LLM if available
        if self.use_llm:
            report = await self._enhance_with_llm(report, context)
        
        # Save to store
        await self.store.save(report)
        
        return report
    
    def _generate_contextual_insight(
        self,
        attack_type: str,
        context: Optional[Dict],
        severity: str
    ) -> str:
        """Generate context-aware insight based on detection data"""
        if not context:
            return f"A {attack_type} threat has been detected by our AI security system."
        
        parts = []
        
        # Source IP insight
        if context.get("source_ip"):
            parts.append(f"Suspicious activity detected from IP {context['source_ip']}")
        
        # Target port insight
        if context.get("target_port"):
            port = context["target_port"]
            port_services = {
                80: "HTTP web traffic",
                443: "HTTPS secure traffic",
                22: "SSH remote access",
                3389: "RDP remote desktop",
                3306: "MySQL database",
                5432: "PostgreSQL database",
                21: "FTP file transfer",
                25: "SMTP email",
                53: "DNS services"
            }
            service = port_services.get(port, f"port {port}")
            parts.append(f"targeting {service}")
        
        # Traffic pattern insight
        if context.get("traffic_pattern"):
            pattern = context["traffic_pattern"]
            parts.append(f"with {pattern} traffic pattern")
        
        # Packets/requests insight
        if context.get("packets_per_second"):
            pps = context["packets_per_second"]
            if pps > 10000:
                parts.append(f"at extremely high volume ({pps:,} packets/sec)")
            elif pps > 1000:
                parts.append(f"at high volume ({pps:,} packets/sec)")
        
        # Request data insight
        if context.get("payload_snippet"):
            parts.append("containing suspicious payload patterns")
        
        # Build the insight
        if parts:
            insight = ", ".join(parts) + f", indicating a possible {attack_type} attempt."
        else:
            insight = f"A {attack_type} threat has been detected."
        
        # Add severity context
        if severity == "CRITICAL":
            insight = "🚨 CRITICAL: " + insight + " Immediate action required."
        elif severity == "HIGH":
            insight = "⚠️ HIGH PRIORITY: " + insight
        
        return insight
    
    def _build_summary(
        self,
        template: Dict,
        attack_type: str,
        confidence: str,
        severity: str,
        context: Optional[Dict]
    ) -> str:
        """Build executive summary with context"""
        full_name = template.get("full_name", attack_type)
        
        base_summary = {
            "CRITICAL": f"🚨 CRITICAL ALERT: {full_name} detected with {confidence} confidence.",
            "HIGH": f"⚠️ HIGH PRIORITY: Potential {full_name} identified requiring urgent attention.",
            "MEDIUM": f"📋 ATTENTION: Possible {full_name} detected. Review recommended.",
            "LOW": f"ℹ️ NOTICE: {full_name} activity detected. Monitor the situation."
        }.get(severity, f"Security threat detected: {full_name}")
        
        # Add context to summary
        if context:
            context_parts = []
            if context.get("source_ip"):
                context_parts.append(f"Source: {context['source_ip']}")
            if context.get("target_port"):
                context_parts.append(f"Target Port: {context['target_port']}")
            if context.get("traffic_pattern"):
                context_parts.append(f"Pattern: {context['traffic_pattern']}")
            
            if context_parts:
                base_summary += " [" + " | ".join(context_parts) + "]"
        
        return base_summary
    
    def _get_contextual_actions(
        self,
        attack_type: str,
        context: Optional[Dict],
        template: Dict
    ) -> List[str]:
        """Get recommended actions based on context"""
        actions = list(template.get("recommended_actions", []))
        
        if not context:
            return actions
        
        # Add context-specific actions
        if context.get("source_ip"):
            ip = context["source_ip"]
            actions.insert(0, f"Block IP address {ip} immediately")
            actions.append(f"Investigate all traffic from {ip} in the last 24 hours")
        
        if context.get("target_port"):
            port = context["target_port"]
            actions.append(f"Review firewall rules for port {port}")
            actions.append(f"Check for unauthorized services on port {port}")
        
        if context.get("traffic_pattern") == "sudden spike":
            actions.insert(1, "Enable emergency rate limiting")
            actions.append("Scale infrastructure if legitimate traffic expected")
        
        if context.get("affected_users"):
            actions.append(f"Notify affected users ({context['affected_users']} accounts)")
        
        return actions[:10]  # Limit to 10 actions
    
    async def _enhance_with_llm(
        self,
        report: Dict[str, Any],
        context: Optional[Dict]
    ) -> Dict[str, Any]:
        """Enhance report with LLM-generated insights"""
        try:
            context_str = json.dumps(context, indent=2) if context else "No additional context"
            
            prompt = f"""You are a cybersecurity expert explaining a threat to a non-technical business user.

Attack Type: {report['attack_details']['type']}
Severity: {report['risk_level']}
Context Data: {context_str}
Current Insight: {report['contextual_insight']}

Provide a JSON response with:
1. "simple_explanation": A 2-3 sentence explanation a CEO could understand
2. "business_impact": The business impact in plain language (2-3 sentences)
3. "executive_advice": One key actionable advice for leadership
4. "personalized_insight": A specific insight based on the context data provided

Keep language simple and actionable."""

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.openai_base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.openai_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "gpt-3.5-turbo",
                        "messages": [
                            {"role": "system", "content": "You are a helpful cybersecurity expert. Always respond with valid JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 600
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    
                    try:
                        llm_data = json.loads(content)
                        report["ai_enhanced"] = {
                            "simple_explanation": llm_data.get("simple_explanation", ""),
                            "business_impact": llm_data.get("business_impact", ""),
                            "executive_advice": llm_data.get("executive_advice", ""),
                            "personalized_insight": llm_data.get("personalized_insight", "")
                        }
                    except json.JSONDecodeError:
                        report["ai_enhanced"] = {"raw_explanation": content}
                        
        except Exception as e:
            print(f"[WARN] LLM enhancement failed: {e}")
            report["ai_enhanced"] = None
        
        return report
    
    def _normalize_attack_type(self, attack_type: str) -> str:
        """Normalize attack type to match templates"""
        mappings = {
            "ddos": "DDoS", "dos": "DoS", "sql injection": "SQL Injection",
            "sqli": "SQL Injection", "xss": "XSS", "phishing": "Phishing",
            "bec": "BEC", "spoof audio": "Spoof Audio", "deepfake": "Spoof Audio",
            "insider threat": "Insider Threat", "brute force": "Brute Force",
            "malware": "Malware", "reconnaissance": "Reconnaissance"
        }
        return mappings.get(attack_type.lower().strip(), attack_type)
    
    def _generate_report_id(self) -> str:
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        unique_id = str(uuid.uuid4())[:8].upper()
        return f"RPT-{timestamp}-{unique_id}"
    
    # Legacy method for backward compatibility
    async def generate_report(
        self,
        attack_type: str,
        confidence: float,
        severity: str,
        source: str,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Legacy method - redirects to contextual report generation"""
        return await self.generate_contextual_report(
            attack_type=attack_type,
            confidence=confidence,
            severity=severity,
            source=source,
            context=additional_context
        )
    
    async def get_attack_types(self) -> List[str]:
        return list(ATTACK_TEMPLATES.keys())
    
    async def get_severity_levels(self) -> Dict[str, Any]:
        return SEVERITY_TONES


# Singleton instance
report_service = ReportService()

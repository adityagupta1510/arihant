"""
ARIHANT SOC - Email Notification Service
========================================
Sends threat alert emails to configured recipients
"""

import os
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, List, Optional
from datetime import datetime


class EmailService:
    """
    Email notification service for threat alerts
    
    Supports SMTP with TLS/SSL
    """
    
    def __init__(self):
        self.smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.sender_email = os.getenv("ALERT_SENDER_EMAIL", self.smtp_user)
        self.recipient_emails = self._parse_recipients(os.getenv("ALERT_RECIPIENT_EMAILS", ""))
        self.enabled = bool(self.smtp_user and self.smtp_password and self.recipient_emails)
        
        if self.enabled:
            print(f"[EMAIL] Service enabled. Recipients: {len(self.recipient_emails)}")
        else:
            print("[EMAIL] Service disabled. Configure SMTP settings to enable.")
    
    def _parse_recipients(self, recipients_str: str) -> List[str]:
        """Parse comma-separated recipient emails"""
        if not recipients_str:
            return []
        return [email.strip() for email in recipients_str.split(",") if email.strip()]
    
    async def send_threat_alert(self, report: Dict[str, Any]) -> bool:
        """
        Send threat alert email based on report
        
        Args:
            report: Threat intelligence report
            
        Returns:
            Success status
        """
        if not self.enabled:
            return False
        
        try:
            subject = self._build_subject(report)
            html_body = self._build_html_body(report)
            text_body = self._build_text_body(report)
            
            return await self._send_email(subject, html_body, text_body)
            
        except Exception as e:
            print(f"[EMAIL] Failed to send alert: {e}")
            return False
    
    def _build_subject(self, report: Dict) -> str:
        """Build email subject line"""
        severity = report.get("risk_level", "UNKNOWN")
        attack_type = report.get("attack_details", {}).get("type", "Security Threat")
        
        severity_prefix = {
            "CRITICAL": "🚨 CRITICAL",
            "HIGH": "⚠️ HIGH",
            "MEDIUM": "📋 MEDIUM",
            "LOW": "ℹ️ LOW"
        }.get(severity, "⚠️")
        
        return f"[ARIHANT SOC] {severity_prefix}: {attack_type} Detected"
    
    def _build_html_body(self, report: Dict) -> str:
        """Build HTML email body"""
        severity = report.get("risk_level", "UNKNOWN")
        severity_color = {
            "CRITICAL": "#FF3B3B",
            "HIGH": "#FF8C00",
            "MEDIUM": "#FFD700",
            "LOW": "#00E0FF"
        }.get(severity, "#888888")
        
        context = report.get("context_data", {})
        actions = report.get("recommended_actions", [])[:5]
        
        context_html = ""
        if context:
            context_items = []
            if context.get("source_ip"):
                context_items.append(f"<li><strong>Source IP:</strong> {context['source_ip']}</li>")
            if context.get("target_port"):
                context_items.append(f"<li><strong>Target Port:</strong> {context['target_port']}</li>")
            if context.get("traffic_pattern"):
                context_items.append(f"<li><strong>Traffic Pattern:</strong> {context['traffic_pattern']}</li>")
            if context_items:
                context_html = f"<ul>{''.join(context_items)}</ul>"
        
        actions_html = "<ol>" + "".join(f"<li>{action}</li>" for action in actions) + "</ol>"
        
        ai_insight = ""
        if report.get("ai_enhanced"):
            ai_data = report["ai_enhanced"]
            if ai_data.get("simple_explanation"):
                ai_insight = f"""
                <div style="background: #f0f9ff; padding: 15px; border-radius: 8px; margin: 15px 0;">
                    <h4 style="margin: 0 0 10px 0; color: #0369a1;">AI Analysis</h4>
                    <p style="margin: 0;">{ai_data['simple_explanation']}</p>
                </div>
                """
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: white; padding: 30px; border-radius: 12px 12px 0 0; }}
                .severity-badge {{ display: inline-block; padding: 8px 16px; border-radius: 20px; font-weight: bold; color: white; background: {severity_color}; }}
                .content {{ background: #ffffff; padding: 25px; border: 1px solid #e2e8f0; }}
                .insight {{ background: #fef3c7; padding: 15px; border-radius: 8px; border-left: 4px solid #f59e0b; margin: 15px 0; }}
                .actions {{ background: #f8fafc; padding: 15px; border-radius: 8px; }}
                .footer {{ background: #f1f5f9; padding: 20px; border-radius: 0 0 12px 12px; text-align: center; font-size: 12px; color: #64748b; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1 style="margin: 0 0 10px 0;">🛡️ ARIHANT SOC Alert</h1>
                    <span class="severity-badge">{severity}</span>
                    <span style="margin-left: 10px; opacity: 0.8;">{report.get('confidence_score', 'N/A')} confidence</span>
                </div>
                
                <div class="content">
                    <h2 style="color: #1e293b; margin-top: 0;">{report.get('title', 'Security Threat Detected')}</h2>
                    
                    <div class="insight">
                        <strong>🎯 Contextual Insight:</strong><br>
                        {report.get('contextual_insight', report.get('summary', 'A security threat has been detected.'))}
                    </div>
                    
                    {ai_insight}
                    
                    <h3>📊 Detection Details</h3>
                    <ul>
                        <li><strong>Attack Type:</strong> {report.get('attack_details', {}).get('type', 'Unknown')}</li>
                        <li><strong>Source:</strong> {report.get('detection_source', 'Unknown').upper()}</li>
                        <li><strong>Report ID:</strong> {report.get('report_id', 'N/A')}</li>
                        <li><strong>Timestamp:</strong> {report.get('timestamp', 'N/A')}</li>
                    </ul>
                    
                    {f"<h3>🔍 Context Data</h3>{context_html}" if context_html else ""}
                    
                    <div class="actions">
                        <h3 style="margin-top: 0;">⚡ Recommended Actions</h3>
                        {actions_html}
                    </div>
                </div>
                
                <div class="footer">
                    <p>This is an automated alert from ARIHANT Security Operations Center.</p>
                    <p>Report ID: {report.get('report_id', 'N/A')} | Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _build_text_body(self, report: Dict) -> str:
        """Build plain text email body"""
        severity = report.get("risk_level", "UNKNOWN")
        actions = report.get("recommended_actions", [])[:5]
        
        actions_text = "\n".join(f"  {i+1}. {action}" for i, action in enumerate(actions))
        
        return f"""
ARIHANT SOC - SECURITY ALERT
============================

SEVERITY: {severity}
CONFIDENCE: {report.get('confidence_score', 'N/A')}

{report.get('title', 'Security Threat Detected')}

CONTEXTUAL INSIGHT:
{report.get('contextual_insight', report.get('summary', 'A security threat has been detected.'))}

DETECTION DETAILS:
- Attack Type: {report.get('attack_details', {}).get('type', 'Unknown')}
- Source: {report.get('detection_source', 'Unknown').upper()}
- Report ID: {report.get('report_id', 'N/A')}
- Timestamp: {report.get('timestamp', 'N/A')}

RECOMMENDED ACTIONS:
{actions_text}

---
This is an automated alert from ARIHANT Security Operations Center.
        """
    
    async def _send_email(self, subject: str, html_body: str, text_body: str) -> bool:
        """Send email via SMTP"""
        try:
            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            message["From"] = self.sender_email
            message["To"] = ", ".join(self.recipient_emails)
            
            # Attach both plain text and HTML versions
            message.attach(MIMEText(text_body, "plain"))
            message.attach(MIMEText(html_body, "html"))
            
            # Create secure connection
            context = ssl.create_default_context()
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.smtp_user, self.smtp_password)
                server.sendmail(
                    self.sender_email,
                    self.recipient_emails,
                    message.as_string()
                )
            
            print(f"[EMAIL] Alert sent to {len(self.recipient_emails)} recipients")
            return True
            
        except Exception as e:
            print(f"[EMAIL] Send failed: {e}")
            return False
    
    async def send_test_email(self) -> bool:
        """Send a test email to verify configuration"""
        if not self.enabled:
            return False
        
        test_report = {
            "report_id": "TEST-001",
            "title": "Test Alert - ARIHANT SOC",
            "risk_level": "LOW",
            "confidence_score": "100%",
            "contextual_insight": "This is a test alert to verify email notification configuration.",
            "summary": "Test alert from ARIHANT SOC email service.",
            "attack_details": {"type": "Test"},
            "detection_source": "system",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "recommended_actions": ["Verify email received", "Check formatting", "Confirm recipients"]
        }
        
        return await self.send_threat_alert(test_report)


# Singleton instance
email_service = EmailService()

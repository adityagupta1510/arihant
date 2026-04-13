"""
ARIHANT SOC - Report Generation Routes
======================================
API endpoints for AI-powered threat report generation

Supports both manual and auto-triggered report generation
"""

from fastapi import APIRouter, HTTPException, Query
from datetime import datetime
from typing import List, Optional

from schemas.report_schemas import (
    ReportGenerateRequest,
    ReportBatchRequest,
    ReportGenerateResponse,
    ReportBatchResponse,
    ThreatReportResponse,
    AttackTypesResponse,
    SeverityLevelsResponse
)
from services.report_service import report_service

router = APIRouter()


@router.post(
    "/generate",
    response_model=ReportGenerateResponse,
    summary="Generate Threat Report",
    description="""
    Generate a comprehensive, human-friendly threat report from ML detection output.
    
    The report includes:
    - Executive summary suitable for non-technical stakeholders
    - Detailed attack information and impact assessment
    - Risk level with color-coded severity
    - Precautionary measures and recommended actions
    - AI-enhanced explanations (when OpenAI API is configured)
    
    Supported attack types include: DDoS, DoS, SQL Injection, XSS, Phishing, BEC, 
    Spoof Audio, Insider Threat, Brute Force, and Malware.
    
    Unknown attack types will receive a generic but informative report.
    """
)
async def generate_report(data: ReportGenerateRequest):
    """
    Generate a threat report from detection output
    
    - **attack_type**: Type of attack detected (e.g., "DDoS", "SQL Injection")
    - **confidence**: Detection confidence score (0.0 to 1.0)
    - **severity**: Severity level (CRITICAL, HIGH, MEDIUM, LOW)
    - **source**: Detection source layer (network, application, audio, human)
    - **additional_context**: Optional additional context about the threat
    
    Returns a structured, human-friendly threat report
    """
    start_time = datetime.now()
    
    try:
        report = await report_service.generate_report(
            attack_type=data.attack_type,
            confidence=data.confidence,
            severity=data.severity.value,
            source=data.source.value if data.source else "network",
            additional_context=data.additional_context
        )
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return ReportGenerateResponse(
            success=True,
            timestamp=datetime.utcnow(),
            processing_time_ms=round(processing_time, 2),
            report=ThreatReportResponse(**report)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate report: {str(e)}"
        )


@router.post(
    "/generate/batch",
    response_model=ReportBatchResponse,
    summary="Generate Multiple Threat Reports",
    description="Generate reports for multiple threats in a single request"
)
async def generate_batch_reports(data: ReportBatchRequest):
    """
    Generate multiple threat reports in batch
    
    - **threats**: List of threat objects to generate reports for (max 20)
    
    Returns a list of structured threat reports
    """
    start_time = datetime.now()
    
    try:
        reports = []
        
        for threat in data.threats:
            report = await report_service.generate_report(
                attack_type=threat.attack_type,
                confidence=threat.confidence,
                severity=threat.severity.value,
                source=threat.source.value if threat.source else "network",
                additional_context=threat.additional_context
            )
            reports.append(ThreatReportResponse(**report))
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return ReportBatchResponse(
            success=True,
            timestamp=datetime.utcnow(),
            processing_time_ms=round(processing_time, 2),
            total_reports=len(reports),
            reports=reports
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate batch reports: {str(e)}"
        )


@router.get(
    "/attack-types",
    response_model=AttackTypesResponse,
    summary="Get Supported Attack Types",
    description="Get list of attack types with predefined report templates"
)
async def get_attack_types():
    """
    Get list of supported attack types
    
    Returns attack types that have predefined templates for detailed reports.
    Unknown attack types will still generate reports but with generic content.
    """
    attack_types = await report_service.get_attack_types()
    
    return AttackTypesResponse(
        success=True,
        attack_types=attack_types,
        total=len(attack_types)
    )


@router.get(
    "/severity-levels",
    response_model=SeverityLevelsResponse,
    summary="Get Severity Level Information",
    description="Get information about severity levels including colors and urgency"
)
async def get_severity_levels():
    """
    Get severity level metadata
    
    Returns information about each severity level including:
    - Tone (critical, warning, caution, informational)
    - Urgency message
    - Color code for UI
    - Icon
    """
    severity_levels = await report_service.get_severity_levels()
    
    return SeverityLevelsResponse(
        success=True,
        severity_levels=severity_levels
    )


@router.post(
    "/preview",
    summary="Preview Report Structure",
    description="Preview the report structure without full generation (for testing)"
)
async def preview_report(data: ReportGenerateRequest):
    """
    Preview report structure for testing
    
    Returns a simplified preview of what the report will contain
    """
    return {
        "success": True,
        "preview": {
            "attack_type": data.attack_type,
            "confidence": f"{int(data.confidence * 100)}%",
            "severity": data.severity.value,
            "source": data.source.value if data.source else "network",
            "will_use_template": data.attack_type.lower() in [
                "ddos", "dos", "sql injection", "xss", "phishing", 
                "bec", "spoof audio", "insider threat", "brute force", "malware"
            ],
            "has_additional_context": data.additional_context is not None
        },
        "timestamp": datetime.utcnow().isoformat()
    }


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT RETRIEVAL ENDPOINTS (For auto-generated reports)
# ═══════════════════════════════════════════════════════════════════════════════

@router.get(
    "/recent",
    summary="Get Recent Reports",
    description="Get recently generated threat intelligence reports"
)
async def get_recent_reports(
    limit: int = Query(20, ge=1, le=100, description="Maximum number of reports")
):
    """
    Get recent threat intelligence reports
    
    Returns reports sorted by timestamp (newest first)
    """
    reports = await report_service.store.list_recent(limit)
    
    return {
        "success": True,
        "timestamp": datetime.utcnow().isoformat(),
        "count": len(reports),
        "reports": reports
    }


@router.get(
    "/{report_id}",
    summary="Get Report by ID",
    description="Get a specific threat intelligence report by its ID"
)
async def get_report(report_id: str):
    """
    Get a specific report by ID
    
    - **report_id**: The unique report identifier (e.g., RPT-20240115103000-ABC12345)
    """
    report = await report_service.store.get(report_id)
    
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    return {
        "success": True,
        "timestamp": datetime.utcnow().isoformat(),
        "report": report
    }


@router.get(
    "/by-alert/{alert_id}",
    summary="Get Report by Alert ID",
    description="Get the threat intelligence report linked to a specific alert"
)
async def get_report_by_alert(alert_id: str):
    """
    Get report linked to an alert
    
    - **alert_id**: The alert ID to find the linked report
    """
    report = await report_service.store.get_by_alert(alert_id)
    
    if not report:
        raise HTTPException(status_code=404, detail="No report found for this alert")
    
    return {
        "success": True,
        "timestamp": datetime.utcnow().isoformat(),
        "report": report
    }


@router.get(
    "/stats/summary",
    summary="Report Statistics",
    description="Get statistics about generated reports"
)
async def get_report_stats():
    """Get report generation statistics"""
    reports = await report_service.store.list_recent(1000)
    
    # Calculate stats
    by_severity = {}
    by_attack_type = {}
    by_source = {}
    ai_enhanced_count = 0
    
    for report in reports:
        # By severity
        sev = report.get("risk_level", "UNKNOWN")
        by_severity[sev] = by_severity.get(sev, 0) + 1
        
        # By attack type
        attack = report.get("attack_details", {}).get("type", "Unknown")
        by_attack_type[attack] = by_attack_type.get(attack, 0) + 1
        
        # By source
        source = report.get("detection_source", "unknown")
        by_source[source] = by_source.get(source, 0) + 1
        
        # AI enhanced
        if report.get("ai_enhanced"):
            ai_enhanced_count += 1
    
    return {
        "success": True,
        "timestamp": datetime.utcnow().isoformat(),
        "total_reports": len(reports),
        "by_severity": by_severity,
        "by_attack_type": by_attack_type,
        "by_source": by_source,
        "ai_enhanced_count": ai_enhanced_count,
        "ai_enhancement_rate": f"{(ai_enhanced_count / len(reports) * 100):.1f}%" if reports else "0%"
    }

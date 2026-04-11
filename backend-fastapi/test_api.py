#!/usr/bin/env python3
"""
ARIHANT SOC - API Test Script
=============================
Quick tests for all API endpoints

Usage:
    python test_api.py
    python test_api.py --base-url http://localhost:8080
"""

import argparse
import requests
import json
from datetime import datetime


def test_health(base_url: str) -> bool:
    """Test health endpoint"""
    print("\n[TEST] Health Check...")
    try:
        resp = requests.get(f"{base_url}/api/health", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            print(f"  ✓ Status: {data.get('status')}")
            print(f"  ✓ Models: {data.get('models', {}).get('models_loaded', [])}")
            return True
        else:
            print(f"  ✗ Failed: {resp.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_network_detection(base_url: str) -> bool:
    """Test network detection endpoint"""
    print("\n[TEST] Network Detection...")
    try:
        payload = {
            "features": [0.5, 0.3, 0.8, 0.1, 0.9, 0.2, 0.7, 0.4, 0.6, 0.5] * 5,
            "source_ip": "192.168.1.100",
            "dest_ip": "10.0.0.1",
            "protocol": "TCP"
        }
        resp = requests.post(
            f"{base_url}/api/network/detect",
            json=payload,
            timeout=30
        )
        if resp.status_code == 200:
            data = resp.json()
            print(f"  ✓ Attack Type: {data.get('attack_type')}")
            print(f"  ✓ Confidence: {data.get('confidence')}")
            print(f"  ✓ Severity: {data.get('severity')}")
            print(f"  ✓ Processing Time: {data.get('processing_time_ms')}ms")
            return True
        else:
            print(f"  ✗ Failed: {resp.status_code} - {resp.text}")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_application_detection(base_url: str) -> bool:
    """Test application detection endpoint"""
    print("\n[TEST] Application Detection...")
    try:
        payload = {
            "request_data": "SELECT * FROM users WHERE id=1 OR 1=1--",
            "url": "/api/users",
            "method": "GET"
        }
        resp = requests.post(
            f"{base_url}/api/application/detect",
            json=payload,
            timeout=30
        )
        if resp.status_code == 200:
            data = resp.json()
            print(f"  ✓ Attack Type: {data.get('attack_type')}")
            print(f"  ✓ Confidence: {data.get('confidence')}")
            print(f"  ✓ Is Attack: {data.get('is_attack')}")
            print(f"  ✓ Indicators: {data.get('indicators', [])[:3]}")
            return True
        else:
            print(f"  ✗ Failed: {resp.status_code} - {resp.text}")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_human_detection(base_url: str) -> bool:
    """Test human threat detection endpoint"""
    print("\n[TEST] Human Threat Detection...")
    try:
        payload = {
            "email_text": "URGENT: Your account has been compromised! Click here immediately to verify your identity and restore access. Act now or your account will be suspended!",
            "subject": "URGENT: Account Security Alert",
            "sender": "security@bank-verify.com"
        }
        resp = requests.post(
            f"{base_url}/api/human/detect",
            json=payload,
            timeout=30
        )
        if resp.status_code == 200:
            data = resp.json()
            print(f"  ✓ Phishing: {data.get('phishing')}")
            print(f"  ✓ Confidence: {data.get('confidence')}")
            print(f"  ✓ Threat Type: {data.get('threat_type')}")
            print(f"  ✓ Highlighted: {data.get('highlighted_phrases', [])[:3]}")
            return True
        else:
            print(f"  ✗ Failed: {resp.status_code} - {resp.text}")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_alerts(base_url: str) -> bool:
    """Test alert management endpoints"""
    print("\n[TEST] Alert Management...")
    try:
        # Create alert
        payload = {
            "attack_type": "Test Attack",
            "severity": "MEDIUM",
            "confidence": 0.85,
            "source": "NETWORK",
            "details": {"test": True}
        }
        resp = requests.post(
            f"{base_url}/api/alerts/create",
            json=payload,
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            alert_id = data.get('alert', {}).get('id')
            print(f"  ✓ Created Alert: {alert_id}")
            
            # Get stats
            stats_resp = requests.get(f"{base_url}/api/alerts/stats", timeout=10)
            if stats_resp.status_code == 200:
                stats = stats_resp.json()
                print(f"  ✓ Total Alerts: {stats.get('total_alerts')}")
            
            return True
        else:
            print(f"  ✗ Failed: {resp.status_code} - {resp.text}")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="ARIHANT SOC API Tests")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the API"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("  ARIHANT SOC - API Test Suite")
    print(f"  Base URL: {args.base_url}")
    print(f"  Time: {datetime.now().isoformat()}")
    print("=" * 60)
    
    results = {
        "Health": test_health(args.base_url),
        "Network Detection": test_network_detection(args.base_url),
        "Application Detection": test_application_detection(args.base_url),
        "Human Threat Detection": test_human_detection(args.base_url),
        "Alert Management": test_alerts(args.base_url),
    }
    
    print("\n" + "=" * 60)
    print("  TEST RESULTS")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    print("=" * 60)
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    exit(main())

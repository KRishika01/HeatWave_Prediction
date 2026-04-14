"""
Verification script for SMS Alert System.
Run this to test if your configuration is working (Mock or Real).
Usage:
  python test_alerts.py
"""

from alert_system import SMSAlertSystem
import os

def test_alert_flow():
    print("--- SMS ALERT SYSTEM TEST ---")
    
    # 1. Create a mock result
    test_result = {
        "risk_level": 3,
        "risk_label": "Severe",
        "emoji": "🔴",
        "composite_score": 92.4,
        "advisory": "⚠ SEVERE HEAT WAVE predicted for tomorrow. Stay indoors!",
        "date": "2026-05-15"
    }

    # 2. Check environment variables
    sid = os.environ.get("TWILIO_ACCOUNT_SID")
    token = os.environ.get("TWILIO_AUTH_TOKEN")
    to = os.environ.get("ALERT_RECIPIENTS")

    if not sid or not token or not to:
        print("[!] Note: Missing TWILIO env vars. System will run in MOCK mode.")
    else:
        print(f"[✓] Found Twilio credentials for account: {sid[:5]}...")

    # 3. Initialize and send
    # Threshold at 2 means High/Severe triggers alert
    manager = SMSAlertSystem(threshold_risk=2)
    
    print("\nSending Test Alert for Delhi...")
    success = manager.send_alert("Delhi", test_result)
    
    if success:
        print("\n[✓] Alert processing complete.")
    else:
        print("\n[✗] Alert processing failed.")

    # 4. Test Low Risk (should not trigger alert)
    print("\nSending Test Alert for Low Risk (should be skipped)...")
    low_result = {"risk_level": 0, "risk_label": "Low"}
    manager.send_alert("Hyderabad", low_result)

if __name__ == "__main__":
    test_alert_flow()

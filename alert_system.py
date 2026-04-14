"""
========================================================================
HEATWAVE RISK PREDICTION SYSTEM
Alert System Module — SMS Notifications
========================================================================
This module handles sending alerts via SMS when heatwave risk levels 
exceed a defined threshold. It supports Twilio and a mock mode.
========================================================================
"""

import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SMSAlertSystem:
    def __init__(self, threshold_risk=2):
        """
        threshold_risk: Risk level at which to trigger an alert.
                        0: Low, 1: Moderate, 2: High, 3: Severe
                        Default is 2 (High).
        """
        self.threshold_risk = threshold_risk
        
        # Twilio Configuration (from environment or config)
        self.account_sid = os.environ.get("TWILIO_ACCOUNT_SID", "YOUR_SID_HERE")
        self.auth_token  = os.environ.get("TWILIO_AUTH_TOKEN",  "YOUR_TOKEN_HERE")
        self.from_number = os.environ.get("TWILIO_FROM_NUMBER", "+1234567890")
        self.to_numbers  = os.environ.get("ALERT_RECIPIENTS", "").split(",")
        
        self.is_configured = (
            self.account_sid and self.account_sid != "YOUR_SID_HERE" and 
            self.auth_token and self.auth_token != "YOUR_TOKEN_HERE" and
            self.to_numbers != [""] and self.to_numbers != []
        )

        logger.info(f"Alert System Initialized. Threshold: {self.threshold_risk}")

        if not self.is_configured:
            logger.warning("SMS Alert System: Twilio credentials missing or incomplete. Running in MOCK mode.")
            logger.info(f"  SID: {'set' if self.account_sid and self.account_sid != 'YOUR_SID_HERE' else 'MISSING'}")
            logger.info(f"  Token: {'set' if self.auth_token and self.auth_token != 'YOUR_TOKEN_HERE' else 'MISSING'}")
            logger.info(f"  Recipients: {self.to_numbers if self.to_numbers != [''] else 'MISSING'}")
        else:
            try:
                from twilio.rest import Client
                self.client = Client(self.account_sid, self.auth_token)
                logger.info("Twilio Client successfully initialized.")
            except ImportError:
                logger.error("twilio library not found. Please install it: pip install twilio")
                self.is_configured = False

    def send_alert(self, city, result):
        """
        Evaluates the prediction result and sends an SMS if risk exceeds threshold.
        """
        risk_level = result.get("risk_level", 0)
        risk_label = result.get("risk_label", "Unknown")
        date_str   = result.get("date", "Today")
        
        if risk_level < self.threshold_risk:
            logger.info(f"Risk level {risk_level} ({risk_label}) is below threshold {self.threshold_risk}. No alert sent.")
            return False

        # Compact message for Trial Accounts (Keep under 160 chars)
        message_body = (
            f"HEATWAVE ALERT! {city}\n"
            f"Date: {date_str}\n"
            f"Risk: {risk_label} {result.get('emoji', '')}\n"
            f"{result.get('advisory', 'Stay safe!')[:60]}"
        )

        if not self.is_configured:
            print("\n" + "!" * 40)
            print("  [MOCK SMS ALERT] — Twilio not configured")
            print(f"  Target: {self.to_numbers}")
            print(f"  Message Preview:\n{message_body}")
            print("!" * 40 + "\n")
            return True

        # Send real SMS
        print(f"\n  [🚀] Sending real SMS to {len(self.to_numbers)} numbers...")
        success_count = 0
        for to_number in self.to_numbers:
            to_number = to_number.strip()
            if not to_number: continue
            try:
                message = self.client.messages.create(
                    body=message_body,
                    from_=self.from_number,
                    to=to_number
                )
                logger.info(f"  [✓] Alert sent to {to_number}. SID: {message.sid}")
                success_count += 1
            except Exception as e:
                logger.error(f"  [✗] Failed to send alert to {to_number}: {e}")
        
        return success_count > 0

if __name__ == "__main__":
    print("SMS Alert System Module Loaded.")

import smtplib
from email.message import EmailMessage

from .config import config_provider

def send_email_otp(recipient_email: str, otp: str):
    smtp_config = config_provider.get_smtp_config()

    subject = "Your OTP Code"
    body = f"Your OTP code is: {otp}\nThis will expire in 10 minutes."

    msg = EmailMessage()
    msg["From"] = smtp_config["email_from"]
    msg["To"] = recipient_email
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        with smtplib.SMTP(smtp_config["host"], smtp_config["port"]) as server:
            server.starttls()
            server.login(smtp_config["user"], smtp_config["password"])
            server.send_message(msg)
        print(f"✅ OTP sent to {recipient_email} - {otp}")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        raise

import smtplib
from email.message import EmailMessage
from core.config import SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, EMAIL_FROM

def send_email_otp(recipient_email: str, otp: str):
    subject = "Your OTP Code"
    body = f"Your OTP code is: {otp}\nThis will expire in 10 minutes."

    msg = EmailMessage()
    msg["From"] = EMAIL_FROM
    msg["To"] = recipient_email
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        print(f"✅ OTP sent to {recipient_email}")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        raise

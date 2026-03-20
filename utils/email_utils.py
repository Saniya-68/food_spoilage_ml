from flask_mail import Message


class NotificationManager:
    def __init__(self, mail_instance, sender_email):
        self.mail = mail_instance
        self.sender_email = sender_email

    def send_expiry_alert(self, recipient_email, item_name, days_left):
        body = (
            f"Reminder: Your item '{item_name}' is expiring in {days_left} day(s). "
            "Use the recipe engine to save it and avoid waste."
        )
        msg = Message(
            subject="Food Spoilage Alert: Act Now",
            sender=self.sender_email,
            recipients=[recipient_email],
            body=body,
        )
        self.mail.send(msg)

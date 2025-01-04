import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
from typing import Dict


async def send_email(patient_info: Dict[str, str]):
    sender_email = "zhangsu1305@gmail.com"
    sender_password = "hmbz wzzr tyjp nqsh"
    receiver_email = "zhangsu1305@gmail.com"
    subject = f"Psychiatrist Appointment Request for {patient_info['name']}"
    body = (
        f"Dear Psychiatrist,\n\n"
        f"We have a new patient requiring your attention. Here are their details:\n\n"
        f"Name: {patient_info['name']}\n"
        f"Email: {patient_info['email']}\n"
        f"Gender: {'Male' if patient_info['gender'] == 1 else 'Female'}\n"
        f"Age: {patient_info['age']}\n\n"
        f"Please reach out to the patient at your earliest convenience.\n\n"
        f"Best regards,\nYour Team"
    )

    # Create email message object
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = Header(subject, "utf-8")

    # Add email body
    message.attach(MIMEText(body, "plain", "utf-8"))

    try:
        # Connect to Gmail SMTP server
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()  # Enable TLS encryption

        # Login to Gmail account
        server.login(sender_email, sender_password)

        # Send email
        server.send_message(message)
        print("Email sent successfully!")

    except Exception as e:
        print(f"Error sending email: {str(e)}")

    finally:
        # Close server connection
        server.quit()


if __name__ == "__main__":
    import asyncio

    # Test patient information
    test_patient = {
        "name": "John Doe",
        "email": "john.doe@example.com",
        "gender": 1,  # 1 for Male, 0 for Female
        "age": "35",
    }

    # Run the async function
    asyncio.run(send_email(test_patient))

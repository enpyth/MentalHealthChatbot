# from langchain_google_community import GmailToolkit
# from langchain_google_community.gmail.utils import (
#     build_resource_service,
#     get_gmail_credentials,
# )
# from langchain_openai import ChatOpenAI
# from langgraph.prebuilt import create_react_agent


# # Can review scopes here https://developers.google.com/gmail/api/auth/scopes
# # For instance, readonly scope is 'https://www.googleapis.com/auth/gmail.readonly'
# def send_email_via_gmail():
#     credentials = get_gmail_credentials(
#         token_file="config/token.json",
#         scopes=["https://mail.google.com/"],
#         client_secrets_file="config/credentials.json",
#     )
#     api_resource = build_resource_service(credentials=credentials)
#     toolkit = GmailToolkit(api_resource=api_resource)
#     tools = toolkit.get_tools()
#     llm = ChatOpenAI(model="gpt-4o-mini")
#     agent_executor = create_react_agent(llm, tools)

#     example_query = (
#         "Send an email to zhangsu1305@gmail.com book a psychiatrist for the patient."
#     )
    
#     events = agent_executor.stream(
#         {"messages": [("user", example_query)]},
#         stream_mode="values",
#     )
#     for event in events:
#         event["messages"][-1].pretty_print()


# if __name__ == "__main__":
#     send_email_via_gmail()

from langchain_google_community import GmailToolkit
from langchain_google_community.gmail.utils import (
    build_resource_service,
    get_gmail_credentials,
)
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from typing import Dict

# Can review scopes here https://developers.google.com/gmail/api/auth/scopes
# For instance, readonly scope is 'https://www.googleapis.com/auth/gmail.readonly'
def send_email_via_gmail(patient_info: Dict[str, str]):
    """
    Sends an email using Gmail API to book a psychiatrist for the patient.

    Args:
        patient_info (dict): A dictionary containing patient details such as name, email, gender, and age.
    """
    # Validate the input data
    if not all(key in patient_info for key in ["name", "email", "gender", "age"]):
        raise ValueError("Patient information must include name, email, gender, and age.")

    # Prepare the email content dynamically
    email_subject = f"Psychiatrist Appointment Request for {patient_info['name']}"
    email_body = (
        f"Dear Psychiatrist,\n\n"
        f"We have a new patient requiring your attention. Here are their details:\n\n"
        f"Name: {patient_info['name']}\n"
        f"Email: {patient_info['email']}\n"
        f"Gender: {'Male' if patient_info['gender'] == 1 else 'Female'}\n"
        f"Age: {patient_info['age']}\n\n"
        f"Please reach out to the patient at your earliest convenience.\n\n"
        f"Best regards,\nYour Team"
    )

    # Set up Gmail API credentials and services
    credentials = get_gmail_credentials(
        token_file="config/token.json",
        scopes=["https://mail.google.com/"],
        client_secrets_file="config/credentials.json",
    )
    api_resource = build_resource_service(credentials=credentials)
    toolkit = GmailToolkit(api_resource=api_resource)
    tools = toolkit.get_tools()
    llm = ChatOpenAI(model="gpt-4o-mini")
    agent_executor = create_react_agent(llm, tools)

    # Format the query for sending the email
    example_query = (
        f"Send an email to zhangsu1305@gmail.com with subject '{email_subject}' and body '{email_body}'."
    )

    # Execute the email sending process
    try:
        events = agent_executor.stream(
            {"messages": [("user", example_query)]},
            stream_mode="values",
        )
        for event in events:
            event["messages"][-1].pretty_print()
        print("Email sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")

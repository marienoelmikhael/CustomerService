import os
from dotenv import load_dotenv
from pathlib import Path

# Add references
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    PromptAgentDefinition,
    CodeInterpreterTool,
    CodeInterpreterToolAuto,
)


def main():

    # Clear the console
    os.system("cls" if os.name == "nt" else "clear")

    # Load environment variables from .env file
    load_dotenv()
    project_endpoint = os.getenv("PROJECT_ENDPOINT")
    model_deployment = os.getenv("MODEL_DEPLOYMENT_NAME")

    # Ask the user for their budget
    while True:
        try:
            budget = float(input("Enter your budget (in USD): "))
            break
        except ValueError:
            print("Please enter a valid number.")

    # Connect to the AI Project and OpenAI clients
    script_dir = Path(__file__).parent
    file_path = script_dir / "phones.csv"

    with (
        DefaultAzureCredential(
            exclude_environment_credential=True,
            exclude_managed_identity_credential=True,
        ) as credential,
        AIProjectClient(endpoint=project_endpoint, credential=credential) as project_client,
        project_client.get_openai_client() as openai_client,
    ):

        # Upload the phones.csv file and create a CodeInterpreterTool
        uploaded_file = openai_client.files.create(
            file=open(file_path, "rb"),
            purpose="assistants",
        )
        print(f"Uploaded {uploaded_file.filename}")

        code_interpreter = CodeInterpreterTool(
            container=CodeInterpreterToolAuto(file_ids=[uploaded_file.id])
        )

        # Define an agent that uses the CodeInterpreterTool
        agent = project_client.agents.create_version(
            agent_name="phone-budget-agent",
            definition=PromptAgentDefinition(
                model=model_deployment,
                instructions=(
                    "You are a helpful shopping assistant. "
                    "You have access to a CSV file containing phone data including names and prices. "
                    "When given a budget, use Python to read the CSV file and filter phones "
                    "that are priced below or equal to that budget. "
                    "Present the results in a clear, readable format showing the phone name and price."
                ),
                tools=[code_interpreter],
            ),
        )

        print(f"Using agent: {agent.name}")

        # Create a conversation for the chat session
        conversation = openai_client.conversations.create()

        # Send the initial budget-based query automatically
        initial_prompt = (
            f"My budget is ${budget:.2f}. "
            f"Please read the phones.csv file and list all phones that cost less than or equal to ${budget:.2f}. "
            f"Show the phone name and price for each result, sorted by price."
        )

        print(f"\nYou: {initial_prompt}\n")

        # Send user message
        openai_client.conversations.items.create(
            conversation_id=conversation.id,
            items=[
                {
                    "type": "message",
                    "role": "user",
                    "content": initial_prompt,
                }
            ],
        )

        # Get response
        response = openai_client.responses.create(
            conversation=conversation.id,
            extra_body={
                "agent": {"name": agent.name, "type": "agent_reference"}
            },
            input="",
        )

        # Check for failure
        if response.status == "failed":
            print(f"Response failed: {response.error}")
        else:
            print(f"Agent: {response.output_text}")

        # Continue with follow-up questions in a loop
        while True:

            user_prompt = input("\nAny follow-up questions? (or type 'quit' to exit): ")

            if user_prompt.lower() == "quit":
                break

            if len(user_prompt) == 0:
                print("Please enter a prompt.")
                continue

            # Send user message
            openai_client.conversations.items.create(
                conversation_id=conversation.id,
                items=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": user_prompt,
                    }
                ],
            )

            # Get response
            response = openai_client.responses.create(
                conversation=conversation.id,
                extra_body={
                    "agent": {"name": agent.name, "type": "agent_reference"}
                },
                input="",
            )

            # Check for failure
            if response.status == "failed":
                print(f"Response failed: {response.error}")
                continue

            # Show agent response
            print(f"Agent: {response.output_text}")

        # Conversation log
        print("\nConversation Log:\n")

        items = openai_client.conversations.items.list(
            conversation_id=conversation.id
        )

        for item in items:
            if item.type == "message":
                role = item.role.upper()
                content = item.content[0].text
                print(f"{role}: {content}\n")

        # Clean up
        openai_client.conversations.delete(
            conversation_id=conversation.id
        )
        print("Conversation deleted")

        project_client.agents.delete_version(
            agent_name=agent.name,
            agent_version=agent.version,
        )
        print("Agent deleted")


if __name__ == "__main__":
    main()

import os
import json
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from typing import Any

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    PromptAgentDefinition,
    FunctionTool,
)
from openai.types.responses.response_input_param import (
    FunctionCallOutput,
    ResponseInputParam,
)


# ========================
# Custom function that the agent will call
# ========================
def recommend_phone(
    budget: float,
    brand: str = "",
    min_storage: int = 0,
    min_screen_size: float = 0.0,
) -> str:

    # Load dataset
    script_dir = Path(__file__).parent
    file_path = script_dir / "phones.csv"
    df = pd.read_csv(file_path)

    # Apply filters
    filtered = df[df["Price"] <= budget]

    if brand:
        filtered = filtered[
            filtered["Brand"].str.lower().str.contains(brand.lower())
        ]

    if min_storage > 0:
        filtered = filtered[filtered["Storage"] >= min_storage]

    if min_screen_size > 0:
        filtered = filtered[filtered["ScreenSize"] >= min_screen_size]

    if filtered.empty:
        return json.dumps(
            {"message": "No phones found matching your criteria. Try adjusting filters."}
        )

    # Sort by price (best value first)
    filtered = filtered.sort_values(by="Price")

    # Take top 3
    results = filtered.head(3).to_dict(orient="records")

    return json.dumps({"recommendations": results})


# ========================
# Main driver
# ========================
def main():

    os.system("cls" if os.name == "nt" else "clear")

    load_dotenv()
    project_endpoint = os.getenv("PROJECT_ENDPOINT")
    model_deployment = os.getenv("MODEL_DEPLOYMENT_NAME")

    # Connect to AI Project + OpenAI
    with (
        DefaultAzureCredential(
            exclude_environment_credential=True,
            exclude_managed_identity_credential=True,
        ) as credential,
        AIProjectClient(endpoint=project_endpoint, credential=credential) as project_client,
        project_client.get_openai_client() as openai_client,
    ):

        # Create the function tool we will expose to the agent
        tool = FunctionTool(
            name="recommend_phone",
            description="Recommends phones given user budget and preferences.",
            parameters={
                "type": "object",
                "properties": {
                    "budget": {
                        "type": "number",
                        "description": "Maximum budget in $",
                    },
                    "brand": {
                        "type": "string",
                        "description": "Preferred brand (optional)",
                    },
                    "min_storage": {
                        "type": "integer",
                        "description": "Minimum storage in GB",
                    },
                    "min_screen_size": {
                        "type": "number",
                        "description": "Minimum screen size in inches",
                    },
                },
                "required": ["budget"],
                "additionalProperties": False,
            },
        )

        # Define the agent
        agent = project_client.agents.create_version(
            agent_name="phone-advisor-agent",
            definition=PromptAgentDefinition(
                model=model_deployment,
                instructions="""
You are a friendly phone sales assistant.

Ask user about:
- Their budget
- Preferred brand (optional)
- Minimum storage required
- Minimum screen size

Use the function recommend_phone when you have collected enough preferences.
""",
                tools=[tool],
            ),
        )

        print(f"Using agent: {agent.name}")

        # Start conversation
        conversation = openai_client.conversations.create()

        while True:
            user_prompt = input("You: ")

            if user_prompt.lower() == "quit":
                break

            # Send user message into Azure AI
            openai_client.conversations.items.create(
                conversation_id=conversation.id,
                items=[{"type": "message", "role": "user", "content": user_prompt}],
            )

            response = openai_client.responses.create(
                conversation=conversation.id,
                extra_body={
                    "agent": {"name": agent.name, "type": "agent_reference"}
                },
                input="",
            )

            # If failed, show error
            if response.status == "failed":
                print(f"Error: {response.error}")
                continue

            # Check if the agent decided to call our function
            input_list: ResponseInputParam = []
            for item in response.output:
                if item.type == "function_call":
                    if item.name == "recommend_phone":

                        result = recommend_phone(**json.loads(item.arguments))

                        input_list.append(
                            FunctionCallOutput(
                                type="function_call_output",
                                call_id=item.call_id,
                                output=result,
                            )
                        )

            # If we got any function outputs, send back to the model
            if input_list:
                response = openai_client.responses.create(
                    input=input_list,
                    previous_response_id=response.id,
                    extra_body={
                        "agent": {"name": agent.name, "type": "agent_reference"}
                    },
                )

            # Print agent output text
            print(f"Agent: {response.output_text}")

        # Clean up
        openai_client.conversations.delete(conversation_id=conversation.id)
        project_client.agents.delete_version(
            agent_name=agent.name,
            agent_version=agent.version,
        )


if __name__ == "__main__":
    main()

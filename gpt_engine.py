import openai
import json
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

# Fetch API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_epics_and_stories(user_goal):
    prompt = f"""
    Based on the product goal: "{user_goal}", generate 1-2 epics and 2-3 user stories per epic.
    
    Respond strictly in this JSON format:

    {{
        "epics": [
            {{
                "summary": "Epic summary",
                "description": "Epic description",
                "stories": [
                    {{
                        "summary": "Story summary",
                        "description": "Story description",
                        "acceptance_criteria": "Given... When... Then..."
                    }}
                ]
            }}
        ]
    }}
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use "gpt-4" if available
            messages=[
                {"role": "system", "content": "You are a product manager assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=1000
        )
        content = response.choices[0].message.content
        return content
    except Exception as e:
        print("OpenAI API Error:", e)
        return None


def parse_response_to_json(response_text):
    try:
        cleaned = response_text.replace("'", '"')
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print("Failed to parse GPT response:", e)
        return None

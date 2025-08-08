# gpt_engine.py
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set! Please add it to your .env file.")

# Initialize OpenAI client with the API key
client = OpenAI(api_key=api_key)

def generate_epics_and_stories(product_goal: str) -> str:
    """
    Given a product goal string, generate epics and user stories
    in a structured text format.
    """
    prompt = f"""
You are Silo, a product management assistant.

Given this product goal:

\"\"\"{product_goal}\"\"\"

Break it down into epics and user stories.

Output ONLY in the following structured text format:

Epic 1: [Epic summary]
Description: [Epic description]

Stories:
  1. Summary: [User story summary]
     Description: [User story detailed description]
     Acceptance Criteria: [Acceptance criteria for the story]

  2. Summary: ...
     Description: ...
     Acceptance Criteria: ...

(Use numbering for epics and stories as shown, indent stories under epics.)
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # or "gpt-4" if you have access
            messages=[
                {"role": "system", "content": "You are a helpful product management assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1500,
            temperature=0.4,
        )
        text_response = response.choices[0].message.content.strip()
        return text_response

    except Exception as e:
        return f"Silo was unable to process the request: {str(e)}"

def parse_response_to_json(response_text: str):
    """
    Placeholder parser to convert structured text into JSON-like dict.
    You can extend this to parse the actual structured text into a dictionary
    for easier processing or Jira integration.

    For now, returns raw text.
    """
    return response_text

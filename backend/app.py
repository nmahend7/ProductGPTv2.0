from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os
import json
from dotenv import load_dotenv

# Loading environment variables from .env
load_dotenv()

app = Flask(__name__)
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")
CORS(app, resources={r"/*": {"origins": FRONTEND_ORIGIN}})

@app.route("/health")
def health():
    return {"status": "ok"}, 200


# Initializing OpenAI client with new SDK. PLEASE KEEP THE SECRET PRIVATE! This is my key lol
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Fallback mechanism if JSON parsing fails. Revisit.
def legacy_parse_response(text):
    epics = []
    current_epic = None

    lines = text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("Epic "):
            if current_epic:
                epics.append(current_epic)

            epic_title = line.split(":", 1)[1].strip()
            i += 1
            epic_description = ""
            if i < len(lines):
                epic_description = lines[i].replace("Description:", "").strip()

            current_epic = {
                "epic": epic_title,
                "description": epic_description,
                "stories": []
            }

        elif line and line[0].isdigit() and "Summary:" in line:
            story_summary = line.split("Summary:", 1)[1].strip()
            i += 1
            story_description = ""
            acceptance_criteria = []

            while i < len(lines) and "Acceptance Criteria:" not in lines[i]:
                if "Description:" in lines[i]:
                    story_description = lines[i].split("Description:", 1)[1].strip()
                i += 1

            if i < len(lines) and "Acceptance Criteria:" in lines[i]:
                i += 1
                while i < len(lines) and not lines[i].strip().startswith(tuple("0123456789")) and not lines[i].strip().startswith("Epic "):
                    crit_line = lines[i].strip()
                    if crit_line.startswith("- "):
                        acceptance_criteria.append(crit_line[2:].strip())
                    i += 1
                i -= 1

            current_epic["stories"].append({
                "summary": story_summary,
                "description": story_description,
                "acceptanceCriteria": acceptance_criteria
            })

        i += 1

    if current_epic:
        epics.append(current_epic)

    return epics

# Main flow or route
@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        goal = data.get("goal", "")

        if not goal:
            return jsonify({"epics": [], "error": "No goal provided."}), 400

        prompt = f"Generate product epics and stories for the following product goal:\n{goal}"

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": """
You are an expert product manager helping engineering teams deliver quality software.

Return ONLY valid JSON using this format:

[
  {
    "epic": "Epic Title",
    "description": "What this epic covers",
    "stories": [
      {
        "summary": "Short summary of the user story",
        "description": "As a [user], I want [feature], so that [value]",
        "acceptanceCriteria": [
          "Clearly testable requirement 1",
          "Measurable acceptance criterion 2"
        ],
        "estimate": "1–2 days",
        "testCases": [
          "Verify that user can do X",
          "Verify error handling when Y happens"
        ]
      }
    ]
  }
]

Guidelines:
- Write high-quality user stories.
- Make acceptance criteria specific, measurable, and testable.
- Use realistic time estimates (e.g., in hours or days).
- Include test cases a QA would write to validate the feature.
- Include engineering dependencies such as APIs, cybersecurity, certification, compliance etc.
- Include UI UX design considerations where applicable 

DO NOT include any commentary or markdown. Only output pure JSON as shown above.
"""
                },
                {"role": "user", "content": prompt}
            ]
        )

        raw_output = response.choices[0].message.content.strip()

        try:
            #  Try parsing directly as JSON
            parsed_epics = json.loads(raw_output)
            return jsonify({
                "epics": parsed_epics,
                "status": "success"
            })

        except json.JSONDecodeError:
            # Fallback to legacy text parser if it does not work
            parsed_epics = legacy_parse_response(raw_output)
            return jsonify({
                "epics": parsed_epics,
                "status": "partial",
                "note": "Returned using fallback parser due to invalid JSON"
            })

    except Exception as e:
        return jsonify({
            "epics": [],
            "status": "error",
            "error": "Unexpected server error.",
            "details": str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=True)

from gpt_engine import generate_epics_and_stories, parse_response_to_json

def main():
    print("Welcome to Silo — your product goal-to-stories assistant!\n")
    user_goal = input("What is your product goal? (e.g., 'Build a personal finance tracker'):\n> ")

    print("\nThinking...")
    raw_response = generate_epics_and_stories(user_goal)

    # Check if GPT returned an error message
    if raw_response.startswith("Silo was unable"):
        print(raw_response)
        return

    # Currently parsing just returns raw text; adjust if you add real parsing later
    try:
        epics = parse_response_to_json(raw_response)
    except Exception as e:
        print(f"Failed to parse GPT response: {e}")
        print("Silo couldn’t understand the response. Try rephrasing your goal.")
        return

    print("\n--- Generated Epics and Stories ---\n")
    # If epics is still raw text, just print it as-is
    if isinstance(epics, str):
        print(epics)
    else:
        # If you later parse into structured data, print nicely
        for epic in epics:
            print(f"Epic: {epic['summary']}")
            print(f"Description: {epic['description']}\n")
            for i, story in enumerate(epic["stories"], 1):
                print(f"  Story {i}: {story['summary']}")
                print(f"    Description: {story['description']}")
                print(f"    Acceptance Criteria: {story['acceptance_criteria']}\n")

if __name__ == "__main__":
    main()

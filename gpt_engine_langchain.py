# gpt_engine.py
import torch
from dotenv import load_dotenv
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from utils import BaselineMemory

load_dotenv()

system_prompt = """
You are an expert product manager helping engineering teams deliver quality software.

Return ONLY valid JSON using this schema:

[
  [
    "epic": "Epic Title",
    "description": "What this epic covers",
    "stories": [
      [
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
      ]
    ]
  ]
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

combined_system_prompt = system_prompt + "\nCurrent baseline:\n{baseline}"

model_name = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True,
)

llm = HuggingFacePipeline(pipeline=pipe)

chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt),
    HumanMessagePromptTemplate.from_template(
        "Generate product epics and stories for the following product goal:\n{user_input}" if "{user_input}" is None else "Follow Up:"
    )
])

# 4. Create memory & chain
memory = BaselineMemory()

chain = LLMChain(
    llm=llm,
    prompt=chat_prompt,
    memory=memory,
    verbose=True
)

# 5. Chat loop with custom commands
print("Type 'save' to save current response as baseline. Type 'exit' to quit.\n")
print("Generate product epics and stories for the following product goal: \n")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    elif user_input.lower() == "save":
        if memory.chat_history:
            latest_ai = memory.chat_history[-1]["ai"]
            memory.update_baseline(latest_ai)
            print("[Baseline updated]")
        else:
            print("[No AI response yet to save]")
    else:
        output = chain.invoke({'user_input':user_input})
        if not memory.chat_history:
            memory.update_baseline(output)
        print("AI:", output)

print("\nFinal Baseline:\n", memory.baseline_response)








































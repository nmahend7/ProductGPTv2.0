from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from langchain_core.memory import BaseMemory

class BaselineMemory(BaseMemory, BaseModel):
    baseline_response: Optional[str] = Field(default=None)
    chat_history: List[Dict[str, str]] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True  # To allow BaseMemory as base class

    @property
    def memory_variables(self) -> List[str]:
        # Variables exposed to prompt templates
        return ["baseline"]

    def load_memory_variables(self, inputs: Dict[str, str]) -> Dict[str, str]:
        # Return the baseline to be used in prompt inputs
        return {"baseline": self.baseline_response or ""}

    def save_context(self, inputs: Dict[str, str], outputs: Dict[str, str]) -> None:
        # Save the new conversation context
        user_message = inputs.get("input", "")
        ai_message = outputs.get("output", "")

        self.chat_history.append({"user": user_message, "ai": ai_message})

        # If no baseline yet, set first response as baseline
        if self.baseline_response is None:
            self.baseline_response = ai_message

    def update_baseline(self, new_baseline: str):
        # Update the baseline response explicitly on demand
        self.baseline_response = new_baseline

    def clear(self):
        # Clear baseline and history
        self.baseline_response = None
        self.chat_history.clear()
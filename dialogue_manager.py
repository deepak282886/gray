from dialogue_state import DialogueState


class DialogueManager:
    def __init__(self):
        self.state = DialogueState()  # Initialize the dialogue state

    def plan_action(self, user_input):
        """Plan the next action based on the user input."""
        intent_entities_prompt = f"Extract intent and entities from: {user_input}"
        response = get_haiku_response(intent_entities_prompt)
        response_data = json.loads(response)
        intent = response_data["intent"]
        entities = response_data["entities"]

        self.state.update_state(intent, entities)  # Update the dialogue state
        action = self.decide_next_action()  # Decide the next action
        return action

    def decide_next_action(self):
        """Decide the next action based on the current state."""
        context = self.state.get_context()  # Get the current context

        if context["current_intent"] == "question":
            answer = fetch_answer_from_graph(context["entities"])  # Fetch answer based on entities
            response_prompt = f"Generate response for: The answer is {answer}"
            response = get_haiku_response(response_prompt)
            response_data = json.loads(response)
            return response_data["response"]
        else:
            return "I didn't understand that. Can you please clarify?"
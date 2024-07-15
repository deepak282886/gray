class DialogueState:
    def __init__(self):
        self.history = []  # List to store the history of interactions
        self.entities = {}  # Dictionary to store recognized entities
        self.current_intent = None  # Current intent of the user

    def update_state(self, intent, entities):
        """Update the dialogue state with new intent and entities."""
        self.history.append((intent, entities))  # Add to history
        self.entities.update(entities)  # Update entities
        self.current_intent = intent  # Set the current intent

    def get_context(self):
        """Retrieve the current context of the conversation."""
        return {
            "history": self.history,
            "entities": self.entities,
            "current_intent": self.current_intent
        }
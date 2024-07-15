class FeedbackHandler:
    def __init__(self, dialogue_manager, graph_manager):
        self.dialogue_manager = dialogue_manager
        self.graph_manager = graph_manager

    def handle_feedback(self, user_feedback):
        """Handle feedback from the user to correct or improve the conversation."""
        if "incorrect" in user_feedback:
            self.dialogue_manager.state.history.pop()  # Remove last incorrect state
            correction = "Can you please rephrase?"
            return correction
        return "Thank you for your feedback."
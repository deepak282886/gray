import networkx as nx


class GraphManager:
    def __init__(self):
        self.G = nx.DiGraph()  # Create an empty directed graph with weighted edges

    def add_to_graph(self, question, answer):
        """Add question and answer to the graph with weighted edges."""
        words = question.split() + answer.split()
        
        for i in range(len(words) - 1):
            if self.G.has_edge(words[i], words[i + 1]):
                self.G[words[i]][words[i + 1]]['weight'] += 1
            else:
                self.G.add_edge(words[i], words[i + 1], weight=1)

    def predict_next_word(self, current_node):
        """Predict the next word based on the highest weight edge."""
        neighbors = list(self.G.successors(current_node))
        if not neighbors:
            return None
        next_node = max(neighbors, key=lambda neighbor: self.G[current_node][neighbor]['weight'])
        return next_node

    def generate_answer(self, question):
        """Generate an answer by traversing the graph from the end of the question."""
        q_words = question.split()
        current_node = q_words[-1]
        answer_words = []

        while True:
            next_word = self.predict_next_word(current_node)
            if next_word and next_word not in q_words:
                answer_words.append(next_word)
                current_node = next_word
            else:
                break
        
        return ' '.join(answer_words)


def fetch_answer_from_graph(entities):
    """Fetch answer from the graph based on entities."""
    # Simulate a graph lookup (replace with actual logic)
    if "France" in entities:
        return "Paris"
    return "Unknown"
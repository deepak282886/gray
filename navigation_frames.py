from llm_call import get_response

# Function to navigate reference frames using LLM
def navigate_reference_frames(network, matrix1, matrix2):
    # Step 1: Select the subject
    subjects = ["Math", "Science"]
    subject_descriptions = {
        "Math": "Math is the study of numbers, shapes, and patterns.",
        "Science": "Science is the systematic study of the structure and behavior of the physical and natural world through observation and experiment."
    }
    prompt = f"""Given the following subjects and their descriptions:
                {"; ".join([f"{subject}: {description}" for subject, description in subject_descriptions.items()])}
                Which subject is most relevant to the transformation from matrix1: {matrix1} to matrix2: {matrix2}? 
                Give only the subject, nothing else. You can only choose from the given list."""
    subject = get_response(prompt)
    print(subject)
    if subject not in subjects:
        return "Relevant subject not found."

    subject_node = network.get_frame(subject)

    # Step 2: Select the concept within the subject
    concepts = list(subject_node.children.keys())
    concept_descriptions = {concept: network.get_frame(concept).get_information().strip() for concept in concepts}
    prompt = f"""Given the following concepts in {subject} and their descriptions:
                 {"; ".join([f"{concept}: {description}" for concept, description in concept_descriptions.items()])}
                 Which concept is most relevant to the transformation from matrix1: {matrix1} to matrix2: {matrix2}? 
                 Give only the concept, nothing else. You can only choose from the given list."""
    concept = get_response(prompt)
    print(concept)
    if concept not in concepts:
        return "Relevant concept not found."

    concept_node = network.get_frame(concept)

    # Step 3: Select the sub-category within the concept
    sub_categories = list(concept_node.children.keys())
    sub_category_descriptions = {sub_category: network.get_frame(sub_category).get_information().strip() for sub_category in sub_categories}
    prompt = f"""Given the following sub-categories in {concept} and their descriptions:
                 {"; ".join([f"{sub_category}: {description}" for sub_category, description in sub_category_descriptions.items()])}
                 Which sub-category is most relevant to the change from matrix1: {matrix1} to matrix2: {matrix2}? 
                 Choose only the sub-category or series of sub-categories from the given list. You can only choose from the given list.
                 Just give names of sub-categories in a list, nothing else. No explanation required.
                 Give the output in a list like ['sub-category1', 'sub-category2'.....]"""
    sub_category = get_response(prompt)
    print(sub_category)
    if sub_category in sub_categories:
        return f"Exact sub-category found: {sub_category}"
    else:
        # If the exact sub-category is not found, check if it's a series of sub-categories
        prompt = f"""Given the following sub-categories in {concept} and their descriptions:
                     {"; ".join([f"{sub_category}: {description}" for sub_category, description in sub_category_descriptions.items()])}
                     What series of sub-categories could explain the change from matrix1: {matrix1} to matrix2: {matrix2}?
                     Choose only the sub-category or series of sub-categories from the given list. You can only choose from the given list.
                     Just give names of sub-categories in a list, nothing else. No explanation required.
                     Give the output in a list like ['sub-category1', 'sub-category2'.....]"""
        series_of_sub_categories = get_response(prompt)
        print(series_of_sub_categories)
        # Add the new series of sub-categories to the reference frames
        network.add_information_to_frame(series_of_sub_categories, "This is a series of sub-categories identified by the LLM.")
        network.link_frames(concept, series_of_sub_categories)

        return f"Series of sub-categories identified and added: {series_of_sub_categories}"
    
    
import networkx as nx
from llm_call import get_response


def navigate_reference_frames_llm(G, matrix1_description, matrix2_description):
    """
    Use an LLM to hypothesize transformations based on descriptions of matrix states and navigate a graph.

    Args:
    G (nx.DiGraph): A directed graph representing transformations and categories.
    matrix1_description (str): Description of the initial matrix state.
    matrix2_description (str): Description of the transformed matrix state.

    Returns:
    str: Description of the most relevant transformation paths in the graph.
    """
    # Step 1: Query LLM to hypothesize transformations based on descriptions
    prompt = f"Given the initial state described as '{matrix1_description}' and a transformed state described as '{matrix2_description}', what are the most likely transformations that could have occurred? Please list potential transformations."
    transformations = get_llm_response(prompt)  # Placeholder for LLM call

    # Step 2: Find matching nodes in the graph
    relevant_nodes = [node for node in G.nodes if node in transformations]

    # Step 3: Traverse from general observations to specific transformations
    paths = []
    for node in relevant_nodes:
        for observation in general_observations:  # Defined globally or passed as an argument
            if nx.has_path(G, observation, node):
                path = nx.shortest_path(G, source=observation, target=node)
                paths.append(path)

    if not paths:
        return "No relevant transformation paths found in the graph."

    # Format the output to show relevant paths
    formatted_paths = [" -> ".join(path) for path in paths]
    return "Identified Paths:\n" + "\n".join(formatted_paths)


# # Example usage
# G = create_example_graph()  # Assume this function sets up your graph as described before
# matrix1_desc = "a simple two-dimensional numeric matrix with low values"
# matrix2_desc = "the same matrix but each element is doubled, suggesting a uniform scale transformation"
result = navigate_reference_frames_llm(G, matrix1_desc, matrix2_desc)
print(result)
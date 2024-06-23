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
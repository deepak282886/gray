import networkx as nx
import matplotlib.pyplot as plt

# Create a directed graph
G = nx.DiGraph()

# General Observations (Expanded)
general_observations = [
    "Color", "Location", "Size", "Pattern", "Shape", "Symmetry",
    "Counting", "Object Interactions", "Spatial Relationships", "Logical Operations",
    "Arithmetic Operations", "Temporal Relationships"  # Added Arithmetic and Temporal
]

# Transformations (Expanded and Refined)
transformations = {
    "Color": ["Color Change", "Color Inversion", "Color Blending", "Saturation Change", "Hue Change", "Brightness Change"],
    "Location": ["Translation", "Rotation", "Reflection", "Position Change", "Relative Positioning", "Grid Alignment", "Mirroring"],
    "Size": ["Scaling", "Expansion", "Reduction", "Proportional Scaling"],
    "Pattern": ["Replication", "Sorting", "Grouping", "Pattern Matching", "Pattern Completion", "Pattern Inversion", "Alternating Patterns"],
    "Shape": ["Morphing", "Deformation", "Shape Change", "Combining Shapes", "Fragmenting Shapes", "Extrusion", "Intrusion"],
    "Symmetry": ["Reflective Symmetry", "Rotational Symmetry", "Symmetrical Arrangement", "Asymmetrical Placement"],
    "Counting": ["Counting Objects", "Counting Differences", "Parity (Even/Odd)"],
    "Object Interactions": ["Object Overlap", "Object Containment", "Object Connection/Disconnection", "Object Replacement"],
    "Spatial Relationships": ["Adjacent Placement", "Above/Below Placement", "Inside/Outside Placement", "Diagonal Placement"],
    "Logical Operations": ["AND", "OR", "NOT", "XOR", "Implication"],
    "Arithmetic Operations": ["Addition", "Subtraction", "Multiplication", "Division"],
    "Temporal Relationships": ["Before/After", "Simultaneous"]
}

# Build the graph
for observation in general_observations:
    G.add_node(observation)
    for transformation in transformations[observation]:
        G.add_node(transformation)
        G.add_edge(observation, transformation)

# Plot the graph (further refined layout)
plt.figure(figsize=(40, 30))  # Even larger figure size for clarity
pos = nx.spring_layout(G, k=0.3, iterations=100, seed=42)  # Fine-tuned layout parameters
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=3500, edge_color='gray', font_size=14, font_weight='bold')
plt.title("Exhaustive Knowledge Graph for ARC Challenge", fontsize=24)
plt.show()
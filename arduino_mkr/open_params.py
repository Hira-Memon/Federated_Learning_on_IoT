import numpy as np

# Load the saved tree parameters
data = np.load('tree_params.npz')

# Print all arrays in a format that's easy to copy to Arduino
print("// Tree structure parameters")
print("const int feature_indices[] = {", ", ".join(map(str, data['feature_indices'])), "};")
print("const float thresholds[] = {", ", ".join(map(str, data['thresholds'])), "};")

# Process the values array (2D)
values_str = []
for node_values in data['values']:
    # Each node has values for [no_precipitation, precipitation]
    values_str.append(f"{{{node_values[0][0]}, {node_values[0][1]}}}")
print("const float values[][2] = [", ", ".join(values_str), "];")

print("const int children_left[] = {", ", ".join(map(str, data['children_left'])), "};")
print("const int children_right[] = {", ", ".join(map(str, data['children_right'])), "};")
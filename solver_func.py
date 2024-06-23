import numpy as np

class Solver:
    def __init__(self, network):
        self.network = network

    def expand_zeroes(self, matrix, block_size):
        expanded_matrix = np.zeros((matrix.shape[0] * block_size, matrix.shape[1] * block_size), dtype=int)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] == 0:
                    expanded_matrix[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = 0
                else:
                    expanded_matrix[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = matrix[i, j]
        return expanded_matrix

    def duplicate_values(self, matrix, value):
        n = matrix.shape[0]
        expanded_matrix = np.zeros((n * 3, n * 3), dtype=int)
        for i in range(n):
            for j in range(n):
                if matrix[i, j] == value:
                    expanded_matrix[i*3:(i+1)*3, j*3:(j+1)*3] = matrix
                else:
                    expanded_matrix[i*3:(i+1)*3, j*3:(j+1)*3] = matrix[i, j]
        return expanded_matrix

    def solve(self, matrix1, matrix2, categories):
        transformations = {
            "Size Transformations": {
                "Expansion": [
                    lambda matrix: self.expand_zeroes(matrix, 3)  # Example block size
                ],
                "Reduction": [
                    # Define other size reduction transformations if necessary
                ]
            },
            "Replication Transformations": {
                "Replication": [
                    lambda matrix: self.duplicate_values(matrix, 7)  # Example value
                ]
            }
        }
        for category in categories:
            for sub_category, funcs in transformations[category].items():
                for func in funcs:
                    try:
                        result = func(matrix1)
                        if np.array_equal(result, matrix2):
                            return f"Transformation {func.__name__} from {sub_category} in {category} works."
                    except Exception as e:
                        print(f"Error applying {func.__name__}: {e}")
                        continue
        return "No matching transformation found."
# neuralscript-core/stdlib/vector_operations.py
import numpy as np

def ns_vector_dot_product(vec1, vec2):
    if not isinstance(vec1, np.ndarray) or not isinstance(vec2, np.ndarray):
        raise TypeError("Arguments must be numpy arrays for dot product.")
    if vec1.shape != vec2.shape:
        raise ValueError("Vectors must have the same shape for dot product.")
    return np.dot(vec1, vec2)

def ns_vector_magnitude(vec):
    if not isinstance(vec, np.ndarray):
        raise TypeError("Argument must be a numpy array for magnitude.")
    return np.linalg.norm(vec)

def ns_vector_normalize(vec):
    if not isinstance(vec, np.ndarray):
        raise TypeError("Argument must be a numpy array for normalize.")
    mag = np.linalg.norm(vec)
    if mag == 0:
        return vec # Or raise error, or return zero vector of same shape
    return vec / mag

def ns_vector_sum_elements(vec):
    if not isinstance(vec, np.ndarray):
        raise TypeError("Argument must be a numpy array for sum_elements.")
    return np.sum(vec)

# Add more vector operations as needed...

# A dictionary to easily register these functions in the runtime
STD_LIB_VECTOR_FUNCTIONS = {
    "vector_dot": ns_vector_dot_product,
    "vector_mag": ns_vector_magnitude,
    "vector_norm": ns_vector_normalize,
    "vector_sum": ns_vector_sum_elements,
}

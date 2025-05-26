# neuralscript-core/stdlib/probability_distributions.py
import numpy as np

class NormalDistribution:
    def __init__(self, mean: np.ndarray, stddev: np.ndarray):
        self.mean = np.array(mean, dtype=float)
        self.stddev = np.array(stddev, dtype=float)
        if self.mean.shape != self.stddev.shape:
            # Basic check, could be more sophisticated for broadcasting
            raise ValueError("Mean and stddev must have the same shape or be broadcastable.")
        print(f"StdLib: NormalDistribution created with mean shape {self.mean.shape}, stddev shape {self.stddev.shape}")

    def sample(self):
        s = np.random.normal(self.mean, self.stddev)
        print(f"StdLib: NormalDistribution sampled, shape {s.shape}")
        return s

    def log_prob(self, value: np.ndarray):
        value = np.array(value, dtype=float)
        # Simplified log_prob, not handling multivariate correctly without scipy or more math
        # This is a placeholder for element-wise log N(value | mean, stddev^2)
        var = self.stddev**2
        log_likelihood = -0.5 * np.log(2 * np.pi * var) - (value - self.mean)**2 / (2 * var)
        lp = np.sum(log_likelihood) # Summing log probs if inputs are vectors/matrices
        print(f"StdLib: NormalDistribution log_prob for value shape {value.shape}, result {lp}")
        return lp

    def __repr__(self):
        return f"NormalDistribution(mean={self.mean}, stddev={self.stddev})"

class CategoricalDistribution:
    def __init__(self, logits: np.ndarray):
        self.logits = np.array(logits, dtype=float) # Expects 1D array of logits for now
        if self.logits.ndim != 1:
            raise ValueError("Logits must be a 1D array for CategoricalDistribution.")
        self.probabilities = self._softmax(self.logits)
        print(f"StdLib: CategoricalDistribution created with {len(self.logits)} categories.")

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x)) # Subtract max for numerical stability
        return e_x / e_x.sum(axis=0)

    def sample(self):
        s = np.random.choice(len(self.logits), p=self.probabilities)
        print(f"StdLib: CategoricalDistribution sampled, result category {s}")
        return s # Returns the index of the sampled category

    def log_prob(self, value: int): # Value is the category index
        if not (0 <= value < len(self.logits)):
            raise ValueError("Value out of range for CategorialDistribution log_prob.")
        # Ensure probabilities are not zero to avoid log(0)
        if self.probabilities[value] == 0:
            return -np.inf # Or raise an error
        lp = np.log(self.probabilities[value])
        print(f"StdLib: CategoricalDistribution log_prob for category {value}, result {lp}")
        return lp
        
    def __repr__(self):
        return f"CategoricalDistribution(num_categories={len(self.logits)})"

# --- Factory functions for NeuralScript ---
def create_normal_distribution(mean, stddev):
    return NormalDistribution(mean, stddev)

def create_categorical_distribution(logits):
    return CategoricalDistribution(logits)

# --- Operations on distribution objects ---
def sample_from_distribution(dist_object):
    if hasattr(dist_object, 'sample'):
        return dist_object.sample()
    raise TypeError(f"Object of type {type(dist_object).__name__} does not have a sample method.")

def log_probability_of_value(dist_object, value):
    if hasattr(dist_object, 'log_prob'):
        return dist_object.log_prob(value)
    raise TypeError(f"Object of type {type(dist_object).__name__} does not have a log_prob method.")

STD_LIB_DISTRIBUTION_FACTORIES_AND_OPS = {
    "create_normal": create_normal_distribution,
    "create_categorical": create_categorical_distribution,
    "sample": sample_from_distribution, # Generic sample function
    "log_prob": log_probability_of_value, # Generic log_prob
}

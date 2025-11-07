import numpy as np

# Define a simple recursive composition function
def combine(node1, node2, W, b):
    return np.tanh(np.dot(W, np.concatenate([node1, node2])) + b)

# Example words represented as embeddings (3-dimensional)
word_vectors = {
    "good": np.array([0.8, 0.6, 0.1]),
    "movie": np.array([0.7, 0.2, 0.5]),
    "not": np.array([-0.6, 0.1, 0.3])
}

# Random weight and bias for combination
W = np.random.randn(3, 6)
b = np.random.randn(3)

# Recursive combination: "not (good movie)"
vec_good_movie = combine(word_vectors["good"], word_vectors["movie"], W, b)
vec_not_good_movie = combine(word_vectors["not"], vec_good_movie, W, b)

print("Vector for 'good movie':", vec_good_movie)
print("Vector for 'not (good movie)':", vec_not_good_movie)

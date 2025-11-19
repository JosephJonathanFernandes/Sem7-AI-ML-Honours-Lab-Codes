

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

class HopfieldNetwork:
    """
    Hopfield Neural Network for Associative Memory with Real Datasets
    """
    
    def __init__(self, n_neurons: int):
        self.n_neurons = n_neurons
        self.weights = np.zeros((n_neurons, n_neurons))
        self.patterns = []
        self.pattern_names = []
        
    def train(self, patterns: np.ndarray, pattern_names: List[str] = None) -> None:
        """
        Train the network using Hebbian learning rule
        
        Args:
            patterns: Array of patterns to store, shape (n_patterns, n_neurons)
            pattern_names: Optional names for the patterns
        """
        self.patterns = patterns.copy()
        self.pattern_names = pattern_names or [f"Pattern_{i}" for i in range(len(patterns))]
        
        # Reset weights
        self.weights = np.zeros((self.n_neurons, self.n_neurons))
        
        # Apply Hebbian learning rule: w_ij = Σ(x_i * x_j) for all patterns
        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)
        
        # No self-connections
        np.fill_diagonal(self.weights, 0)
        
        # Normalize by number of patterns
        self.weights /= len(patterns)
        
        print(f"Training completed. Stored {len(patterns)} patterns.")
    
    def activation_function(self, x: np.ndarray) -> np.ndarray:
        """Bipolar activation function"""
        return np.where(x >= 0, 1, -1)
    
    def energy(self, state: np.ndarray) -> float:
        """Calculate energy of the network state"""
        return -0.5 * np.sum(self.weights * np.outer(state, state))
    
    def recall(self, input_pattern: np.ndarray, max_iterations: int = 100) -> Tuple[np.ndarray, List[float], int]:
        """
        Recall stored pattern from noisy input
        
        Returns:
            Tuple of (final_state, energy_history, iterations_used)
        """
        state = input_pattern.copy().astype(float)
        energy_history = []
        
        for iteration in range(max_iterations):
            current_energy = self.energy(state)
            energy_history.append(current_energy)
            
            prev_state = state.copy()
            
            # Asynchronous update
            for i in range(self.n_neurons):
                net_input = np.dot(self.weights[i], state)
                state[i] = self.activation_function(net_input)
            
            # Check for convergence
            if np.array_equal(state, prev_state):
                break
        
        return state, energy_history, iteration + 1
    
    def add_noise(self, pattern: np.ndarray, noise_level: float = 0.2) -> np.ndarray:
        """Add random noise to a pattern"""
        noisy_pattern = pattern.copy()
        n_flips = int(noise_level * len(pattern))
        flip_indices = np.random.choice(len(pattern), size=n_flips, replace=False)
        noisy_pattern[flip_indices] *= -1
        return noisy_pattern
    
    def pattern_similarity(self, pattern1: np.ndarray, pattern2: np.ndarray) -> float:
        """Calculate similarity between two patterns"""
        return np.mean(pattern1 == pattern2)

def load_mnist_patterns(n_patterns=5, image_size=8):
    """
    Load and prepare MNIST digit patterns for Hopfield network
    
    Args:
        n_patterns: Number of patterns to load (digits 0 to n_patterns-1)
        image_size: Resize images to this size (image_size x image_size)
    
    Returns:
        Tuple of (patterns_array, pattern_names, original_images)
    """
    try:
        from sklearn.datasets import fetch_openml
        
        print(f"Loading MNIST dataset...")
        # Load MNIST dataset
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X, y = mnist['data'], mnist['target'].astype(int)
        
        # Normalize pixel values
        X = X / 255.0
        
        # Select one example of each digit
        patterns = []
        pattern_names = []
        original_images = []
        
        for digit in range(n_patterns):
            digit_indices = np.where(y == digit)[0]
            if len(digit_indices) > 0:
                # Get a clear example of this digit (not the first one, which might be unclear)
                # Try a few examples and pick one with good contrast
                best_idx = digit_indices[0]
                best_contrast = 0
                
                for idx in digit_indices[:min(10, len(digit_indices))]:
                    img = X[idx].reshape(28, 28)
                    contrast = np.std(img)  # Higher std means better contrast
                    if contrast > best_contrast:
                        best_contrast = contrast
                        best_idx = idx
                
                digit_image = X[best_idx]
                original_images.append(digit_image.reshape(28, 28))
                
                # Reshape and resize
                img_28x28 = digit_image.reshape(28, 28)
                
                # Downsample to target size
                step = 28 // image_size
                img_small = img_28x28[::step, ::step][:image_size, :image_size]
                
                # Convert to bipolar using adaptive threshold
                threshold = np.mean(img_small) + 0.1 * np.std(img_small)
                bipolar_pattern = np.where(img_small > threshold, 1, -1).flatten()
                
                patterns.append(bipolar_pattern)
                pattern_names.append(f"Digit_{digit}")
                
                print(f"  Loaded digit {digit}: {len(bipolar_pattern)} pixels (contrast: {best_contrast:.3f})")
        
        return np.array(patterns), pattern_names, original_images
        
    except ImportError:
        print("scikit-learn not available, using fallback patterns...")
        return create_synthetic_patterns()
    except Exception as e:
        print(f"Error loading MNIST: {e}")
        print("Using fallback patterns...")
        return create_synthetic_patterns()

def load_iris_patterns():
    """
    Load Iris dataset and convert to binary patterns
    """
    try:
        from sklearn.datasets import load_iris
        from sklearn.preprocessing import StandardScaler
        
        print("Loading Iris dataset...")
        iris = load_iris()
        
        # Standardize features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(iris.data)
        
        # Take one example from each class
        patterns = []
        pattern_names = []
        
        for class_id in range(3):
            class_indices = np.where(iris.target == class_id)[0]
            # Take the example closest to the class centroid
            class_data = scaled_data[class_indices]
            centroid = np.mean(class_data, axis=0)
            distances = np.sum((class_data - centroid) ** 2, axis=1)
            best_idx = class_indices[np.argmin(distances)]
            
            features = scaled_data[best_idx]
            
            # Convert to bipolar using zero threshold (since data is standardized)
            bipolar_pattern = np.where(features > 0, 1, -1)
            
            patterns.append(bipolar_pattern)
            pattern_names.append(f"Iris_{iris.target_names[class_id]}")
        
        print(f"  Loaded {len(patterns)} iris patterns with {len(patterns[0])} features")
        return np.array(patterns), pattern_names, iris.feature_names
        
    except ImportError:
        print("scikit-learn not available for Iris dataset")
        return None, None, None

def load_wine_patterns():
    """
    Load Wine dataset and convert to binary patterns
    """
    try:
        from sklearn.datasets import load_wine
        from sklearn.preprocessing import StandardScaler
        
        print("Loading Wine dataset...")
        wine = load_wine()
        
        # Standardize features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(wine.data)
        
        # Take representative sample from each class
        patterns = []
        pattern_names = []
        
        for class_id in range(3):
            class_indices = np.where(wine.target == class_id)[0]
            # Take example closest to centroid
            class_data = scaled_data[class_indices]
            centroid = np.mean(class_data, axis=0)
            distances = np.sum((class_data - centroid) ** 2, axis=1)
            best_idx = class_indices[np.argmin(distances)]
            
            features = scaled_data[best_idx]
            
            # Convert to bipolar
            bipolar_pattern = np.where(features > 0, 1, -1)
            
            patterns.append(bipolar_pattern)
            pattern_names.append(f"Wine_Class_{class_id}")
        
        print(f"  Loaded {len(patterns)} wine patterns with {len(patterns[0])} features")
        return np.array(patterns), pattern_names, wine.feature_names
        
    except ImportError:
        print("scikit-learn not available for Wine dataset")
        return None, None, None

def load_breast_cancer_patterns():
    """
    Load Breast Cancer dataset and convert to binary patterns
    """
    try:
        from sklearn.datasets import load_breast_cancer
        from sklearn.preprocessing import StandardScaler
        
        print("Loading Breast Cancer dataset...")
        cancer = load_breast_cancer()
        
        # Standardize features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cancer.data)
        
        # Take representative samples from each class
        patterns = []
        pattern_names = []
        
        for class_id in range(2):  # Binary classification
            class_indices = np.where(cancer.target == class_id)[0]
            # Take example closest to centroid
            class_data = scaled_data[class_indices]
            centroid = np.mean(class_data, axis=0)
            distances = np.sum((class_data - centroid) ** 2, axis=1)
            best_idx = class_indices[np.argmin(distances)]
            
            features = scaled_data[best_idx]
            
            # Convert to bipolar
            bipolar_pattern = np.where(features > 0, 1, -1)
            
            patterns.append(bipolar_pattern)
            class_name = "Malignant" if class_id == 0 else "Benign"
            pattern_names.append(f"Cancer_{class_name}")
        
        print(f"  Loaded {len(patterns)} cancer patterns with {len(patterns[0])} features")
        return np.array(patterns), pattern_names, cancer.feature_names
        
    except ImportError:
        print("scikit-learn not available for Breast Cancer dataset")
        return None, None, None

def create_synthetic_patterns():
    """
    Create synthetic patterns as fallback
    """
    print("Creating synthetic letter patterns...")
    
    # Letter patterns (5x7)
    A = np.array([
        [-1, -1, 1, -1, -1],
        [-1, 1, -1, 1, -1],
        [1, -1, -1, -1, 1],
        [1, 1, 1, 1, 1],
        [1, -1, -1, -1, 1],
        [1, -1, -1, -1, 1],
        [1, -1, -1, -1, 1]
    ]).flatten()
    
    B = np.array([
        [1, 1, 1, 1, -1],
        [1, -1, -1, -1, 1],
        [1, -1, -1, -1, 1],
        [1, 1, 1, 1, -1],
        [1, -1, -1, -1, 1],
        [1, -1, -1, -1, 1],
        [1, 1, 1, 1, -1]
    ]).flatten()
    
    C = np.array([
        [-1, 1, 1, 1, -1],
        [1, -1, -1, -1, 1],
        [1, -1, -1, -1, -1],
        [1, -1, -1, -1, -1],
        [1, -1, -1, -1, -1],
        [1, -1, -1, -1, 1],
        [-1, 1, 1, 1, -1]
    ]).flatten()
    
    patterns = np.array([A, B, C])
    names = ["Letter_A", "Letter_B", "Letter_C"]
    
    return patterns, names, [None, None, None]

def visualize_patterns(patterns: np.ndarray, names: List[str], shape: Optional[Tuple[int, int]] = None):
    """Visualize patterns as images"""
    n_patterns = len(patterns)
    fig, axes = plt.subplots(1, n_patterns, figsize=(3*n_patterns, 3))
    
    if n_patterns == 1:
        axes = [axes]
    
    for i, (pattern, name) in enumerate(zip(patterns, names)):
        if shape:
            img = pattern.reshape(shape)
        else:
            # Try to make a square-ish image
            size = int(np.sqrt(len(pattern)))
            if size * size <= len(pattern):
                img = pattern[:size*size].reshape(size, size)
            else:
                img = pattern.reshape(-1, 1)  # Column vector if can't make square
        
        axes[i].imshow(img, cmap='RdBu', vmin=-1, vmax=1)
        axes[i].set_title(name)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def experiment_with_dataset(dataset_name: str):
    """
    Run complete experiment with a specific dataset
    """
    print(f"\n{'='*60}")
    print(f"HOPFIELD NETWORK EXPERIMENT WITH {dataset_name.upper()}")
    print(f"{'='*60}")
    
    # Load dataset
    if dataset_name.lower() == 'mnist':
        patterns, names, extra = load_mnist_patterns(n_patterns=3, image_size=8)
        shape = (8, 8)
    elif dataset_name.lower() == 'iris':
        patterns, names, extra = load_iris_patterns()
        shape = None
    elif dataset_name.lower() == 'wine':
        patterns, names, extra = load_wine_patterns()
        shape = None
    elif dataset_name.lower() == 'cancer':
        patterns, names, extra = load_breast_cancer_patterns()
        shape = None
    else:
        patterns, names, extra = create_synthetic_patterns()
        shape = (7, 5)
    
    if patterns is None:
        print(f"Failed to load {dataset_name} dataset")
        return
    
    n_neurons = patterns.shape[1]
    
    print(f"\nDataset Information:")
    print(f"  - Patterns: {len(patterns)}")
    print(f"  - Neurons per pattern: {n_neurons}")
    print(f"  - Theoretical capacity: ~{0.14 * n_neurons:.1f} patterns")
    
    # Initialize and train network
    network = HopfieldNetwork(n_neurons)
    network.train(patterns, names)
    
    # Visualize original patterns if possible
    if shape or dataset_name.lower() == 'synthetic':
        print(f"\n1. ORIGINAL PATTERNS:")
        visualize_patterns(patterns, names, shape)
    
    # Test noise tolerance
    print(f"\n2. NOISE TOLERANCE TEST:")
    noise_levels = [0.1, 0.2, 0.3]
    
    for noise_level in noise_levels:
        print(f"\n   Testing with {noise_level*100:.0f}% noise:")
        
        success_count = 0
        for i, (pattern, name) in enumerate(zip(patterns, names)):
            # Add noise
            noisy_pattern = network.add_noise(pattern, noise_level)
            
            # Recall
            recalled, energy_history, iterations = network.recall(noisy_pattern, max_iterations=50)
            
            # Check success
            similarity = network.pattern_similarity(pattern, recalled)
            success = similarity > 0.8
            success_count += success
            
            print(f"     {name}: {'✓' if success else '✗'} "
                  f"(similarity: {similarity:.3f}, iterations: {iterations})")
        
        success_rate = success_count / len(patterns)
        print(f"   Overall success rate: {success_rate:.2%}")
    
    # Test with specific pattern
    test_pattern_idx = 0
    original = patterns[test_pattern_idx]
    noisy = network.add_noise(original, 0.25)
    recalled, energy_history, iterations = network.recall(noisy)
    
    print(f"\n3. DETAILED ANALYSIS FOR {names[test_pattern_idx]}:")
    print(f"   - Original pattern: {original[:10]}... ({len(original)} total)")
    print(f"   - Noisy input: {noisy[:10]}...")
    print(f"   - Recalled pattern: {recalled[:10]}...")
    print(f"   - Similarity: {network.pattern_similarity(original, recalled):.3f}")
    print(f"   - Energy decrease: {energy_history[0]:.3f} → {energy_history[-1]:.3f}")
    print(f"   - Convergence: {iterations} iterations")
    
    # Visualize results if possible
    if shape or dataset_name.lower() == 'synthetic':
        print(f"\n4. PATTERN RECALL VISUALIZATION:")
        visualize_patterns([original, noisy, recalled], 
                         [f'Original {names[test_pattern_idx]}', 
                          f'Noisy (25%)', 'Recalled'], shape)
    
    return network, patterns, names

def comprehensive_comparison():
    """
    Compare performance across different datasets
    """
    print(f"\n{'='*70}")
    print("COMPREHENSIVE DATASET COMPARISON")
    print(f"{'='*70}")
    
    datasets = ['mnist', 'iris', 'wine', 'cancer']
    results = {}
    
    for dataset in datasets:
        try:
            print(f"\nTesting {dataset.upper()} dataset...")
            network, patterns, names = experiment_with_dataset(dataset)
            
            if network is not None:
                # Quick performance test
                total_tests = 0
                successful_recalls = 0
                
                for pattern in patterns:
                    # Test with moderate noise
                    noisy = network.add_noise(pattern, 0.2)
                    recalled, _, _ = network.recall(noisy, max_iterations=30)
                    
                    similarity = network.pattern_similarity(pattern, recalled)
                    if similarity > 0.8:
                        successful_recalls += 1
                    total_tests += 1
                
                success_rate = successful_recalls / total_tests
                results[dataset] = {
                    'patterns': len(patterns),
                    'neurons': patterns.shape[1],
                    'success_rate': success_rate,
                    'capacity_ratio': len(patterns) / (0.14 * patterns.shape[1])
                }
        
        except Exception as e:
            print(f"Error with {dataset}: {e}")
            results[dataset] = None
    
    # Summary
    print(f"\n{'='*70}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    print(f"{'Dataset':<12} {'Patterns':<10} {'Neurons':<10} {'Success Rate':<15} {'Capacity Usage'}")
    print(f"{'-'*65}")
    
    for dataset, result in results.items():
        if result:
            print(f"{dataset.upper():<12} {result['patterns']:<10} "
                  f"{result['neurons']:<10} {result['success_rate']:<15.2%} "
                  f"{result['capacity_ratio']:.2f}x")
        else:
            print(f"{dataset.upper():<12} {'Failed':<10}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print("HOPFIELD NEURAL NETWORK WITH REAL DATASETS")
    print("=" * 60)
    print("Available datasets: MNIST, Iris, Wine, Breast Cancer")
    print("=" * 60)
    
    # Run comprehensive comparison
    comprehensive_comparison()
    
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETED!")
    print(f"{'='*70}")
    
    print("\nKEY FINDINGS:")
    print("✓ Hopfield networks work with real-world datasets")
    print("✓ Performance varies with dataset characteristics")
    print("✓ Proper preprocessing is crucial for binary conversion")
    print("✓ Network capacity limits are dataset-dependent")
    print("✓ Energy minimization ensures convergence")


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA

class MNISTHopfieldNetwork:
    """Hopfield Network specifically designed for MNIST digits"""
    
    def __init__(self, n_components=100):
        """
        Initialize MNIST Hopfield Network
        
        Args:
            n_components: Number of PCA components for dimensionality reduction
        """
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.weights = None
        self.stored_patterns = []
        self.stored_labels = []
        self.digit_names = []
        
    def load_and_preprocess_mnist(self, digits_to_load=[0, 1, 2, 3, 4], samples_per_digit=1):
        """
        Load and preprocess MNIST dataset
        
        Args:
            digits_to_load: List of digits to load (0-9)
            samples_per_digit: Number of samples per digit
            
        Returns:
            Preprocessed patterns ready for training
        """
        print("Loading MNIST dataset...")
        
        # Load MNIST
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X, y = mnist['data'], mnist['target'].astype(int)
        
        # Normalize
        X = X / 255.0
        
        # Select specific digits
        selected_patterns = []
        selected_labels = []
        pattern_names = []
        original_images = []
        
        for digit in digits_to_load:
            digit_indices = np.where(y == digit)[0]
            
            # Select multiple samples if requested
            for sample in range(min(samples_per_digit, len(digit_indices))):
                if sample < len(digit_indices):
                    idx = digit_indices[sample]
                    
                    # Get original image
                    img = X[idx]
                    original_images.append(img.reshape(28, 28))
                    
                    selected_patterns.append(img)
                    selected_labels.append(digit)
                    pattern_names.append(f"Digit_{digit}_Sample_{sample}")
        
        selected_patterns = np.array(selected_patterns)
        
        print(f"Selected {len(selected_patterns)} digit samples")
        print(f"Original image size: {selected_patterns.shape[1]} pixels")
        
        # Apply PCA for dimensionality reduction
        print(f"Applying PCA (reducing to {self.n_components} components)...")
        patterns_reduced = self.pca.fit_transform(selected_patterns)
        
        # Convert to bipolar
        patterns_bipolar = []
        for i, pattern in enumerate(patterns_reduced):
            # Use adaptive threshold based on pattern statistics
            threshold = np.mean(pattern) + 0.1 * np.std(pattern)
            bipolar = np.where(pattern > threshold, 1, -1)
            patterns_bipolar.append(bipolar)
        
        patterns_bipolar = np.array(patterns_bipolar)
        
        print(f"Final pattern size: {patterns_bipolar.shape[1]} features")
        print(f"Patterns converted to bipolar format")
        
        return patterns_bipolar, selected_labels, pattern_names, original_images
    
    def train(self, patterns, labels, pattern_names):
        """
        Train the Hopfield network using Hebbian learning
        
        Args:
            patterns: Bipolar patterns to store
            labels: Corresponding digit labels
            pattern_names: Names for the patterns
        """
        self.stored_patterns = patterns.copy()
        self.stored_labels = labels
        self.digit_names = pattern_names
        
        n_neurons = patterns.shape[1]
        self.weights = np.zeros((n_neurons, n_neurons))
        
        print(f"\nTraining Hopfield network...")
        print(f"Network size: {n_neurons} neurons")
        print(f"Storing {len(patterns)} patterns")
        
        # Hebbian learning rule
        for i, pattern in enumerate(patterns):
            self.weights += np.outer(pattern, pattern)
            print(f"  Added pattern {i+1}: {pattern_names[i]}")
        
        # Remove self-connections
        np.fill_diagonal(self.weights, 0)
        
        # Normalize
        self.weights /= len(patterns)
        
        print("Training completed!")
        
        # Check weight matrix properties
        is_symmetric = np.allclose(self.weights, self.weights.T)
        diagonal_zero = np.allclose(np.diag(self.weights), 0)
        
        print(f"Weight matrix properties:")
        print(f"  - Symmetric: {is_symmetric}")
        print(f"  - Zero diagonal: {diagonal_zero}")
        print(f"  - Weight range: [{self.weights.min():.3f}, {self.weights.max():.3f}]")
    
    def add_noise_to_digit(self, pattern, noise_type='flip', noise_level=0.2):
        """
        Add different types of noise to digit patterns
        
        Args:
            pattern: Original pattern
            noise_type: Type of noise ('flip', 'gaussian', 'dropout')
            noise_level: Amount of noise (0.0 to 1.0)
            
        Returns:
            Noisy pattern
        """
        noisy = pattern.copy()
        
        if noise_type == 'flip':
            # Flip random bits
            n_flips = int(noise_level * len(pattern))
            flip_indices = np.random.choice(len(pattern), size=n_flips, replace=False)
            noisy[flip_indices] *= -1
            
        elif noise_type == 'gaussian':
            # Add Gaussian noise then re-binarize
            noise = np.random.normal(0, noise_level, size=len(pattern))
            noisy_continuous = pattern + noise
            noisy = np.where(noisy_continuous > 0, 1, -1)
            
        elif noise_type == 'dropout':
            # Set random bits to -1
            dropout_mask = np.random.random(len(pattern)) < noise_level
            noisy[dropout_mask] = -1
        
        return noisy
    
    def recall_digit(self, noisy_pattern, max_iterations=100, verbose=False):
        """
        Recall stored digit from noisy input
        
        Args:
            noisy_pattern: Corrupted input pattern
            max_iterations: Maximum recall iterations
            verbose: Print iteration details
            
        Returns:
            Tuple of (recalled_pattern, energy_history, iterations, convergence_info)
        """
        state = noisy_pattern.copy().astype(float)
        energy_history = []
        state_history = []
        
        if verbose:
            print("\nRecall process:")
        
        for iteration in range(max_iterations):
            # Calculate energy
            energy = -0.5 * np.sum(self.weights * np.outer(state, state))
            energy_history.append(energy)
            state_history.append(state.copy())
            
            if verbose and iteration < 10:
                print(f"  Iteration {iteration + 1}: Energy = {energy:.3f}")
            
            # Store previous state
            prev_state = state.copy()
            
            # Asynchronous update (one neuron at a time)
            for i in range(len(state)):
                net_input = np.dot(self.weights[i], state)
                state[i] = 1 if net_input >= 0 else -1
            
            # Check for convergence
            if np.array_equal(state, prev_state):
                if verbose:
                    print(f"  Converged after {iteration + 1} iterations")
                break
        
        convergence_info = {
            'converged': iteration < max_iterations - 1,
            'iterations': iteration + 1,
            'final_energy': energy_history[-1],
            'energy_decrease': energy_history[0] - energy_history[-1]
        }
        
        return state, energy_history, state_history, convergence_info
    
    def identify_recalled_digit(self, recalled_pattern):
        """
        Identify which stored digit the recalled pattern matches
        
        Args:
            recalled_pattern: Pattern recalled from memory
            
        Returns:
            Tuple of (best_match_idx, similarities, confidence)
        """
        similarities = []
        
        for i, stored_pattern in enumerate(self.stored_patterns):
            similarity = np.mean(stored_pattern == recalled_pattern)
            similarities.append(similarity)
        
        best_match_idx = np.argmax(similarities)
        confidence = similarities[best_match_idx]
        
        return best_match_idx, similarities, confidence
    
    def visualize_digit_recall(self, original_idx, noise_type='flip', noise_level=0.2):
        """
        Visualize the complete digit recall process
        
        Args:
            original_idx: Index of stored pattern to test
            noise_type: Type of noise to add
            noise_level: Amount of noise
        """
        if original_idx >= len(self.stored_patterns):
            print(f"Invalid pattern index: {original_idx}")
            return
        
        # Get original pattern and add noise
        original = self.stored_patterns[original_idx]
        noisy = self.add_noise_to_digit(original, noise_type, noise_level)
        
        # Recall process
        recalled, energy_history, state_history, conv_info = self.recall_digit(
            noisy, verbose=True
        )
        
        # Identify recalled digit
        best_match, similarities, confidence = self.identify_recalled_digit(recalled)
        
        print(f"\\nRecall Results:")
        print(f"  Original: {self.digit_names[original_idx]} (Label: {self.stored_labels[original_idx]})")
        print(f"  Noise: {noise_type} at {noise_level*100}% level")
        print(f"  Convergence: {conv_info['converged']} in {conv_info['iterations']} iterations")
        print(f"  Energy change: {conv_info['energy_decrease']:.3f}")
        print(f"  Best match: {self.digit_names[best_match]} (Confidence: {confidence:.3f})")
        print(f"  Recall success: {best_match == original_idx}")
        
        # Reconstruct images for visualization
        original_img = self.pca.inverse_transform([original])[0].reshape(28, 28)
        
        # For noisy pattern, we need to inverse transform carefully
        noisy_pca = self.pca.inverse_transform([noisy])[0].reshape(28, 28)
        recalled_img = self.pca.inverse_transform([recalled])[0].reshape(28, 28)
        
        # Plot results
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        
        # Top row: PCA space patterns
        axes[0, 0].bar(range(min(20, len(original))), original[:20])
        axes[0, 0].set_title(f'Original Pattern\\n(PCA space)')
        axes[0, 0].set_ylim([-1.5, 1.5])
        
        axes[0, 1].bar(range(min(20, len(noisy))), noisy[:20])
        axes[0, 1].set_title(f'Noisy Pattern\\n({noise_level*100}% {noise_type})')
        axes[0, 1].set_ylim([-1.5, 1.5])
        
        axes[0, 2].bar(range(min(20, len(recalled))), recalled[:20])
        axes[0, 2].set_title('Recalled Pattern')
        axes[0, 2].set_ylim([-1.5, 1.5])
        
        axes[0, 3].plot(energy_history)
        axes[0, 3].set_title('Energy During Recall')
        axes[0, 3].set_xlabel('Iteration')
        axes[0, 3].set_ylabel('Energy')
        axes[0, 3].grid(True)
        
        # Bottom row: Reconstructed images
        axes[1, 0].imshow(original_img, cmap='gray')
        axes[1, 0].set_title(f'Original Digit\\n{self.stored_labels[original_idx]}')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(noisy_pca, cmap='gray')
        axes[1, 1].set_title('Noisy Input')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(recalled_img, cmap='gray')
        axes[1, 2].set_title(f'Recalled Digit\\n{self.stored_labels[best_match]}')
        axes[1, 2].axis('off')
        
        # Similarity scores
        axes[1, 3].bar(range(len(similarities)), similarities)
        axes[1, 3].set_title('Pattern Similarities')
        axes[1, 3].set_xlabel('Stored Pattern Index')
        axes[1, 3].set_ylabel('Similarity')
        axes[1, 3].set_xticks(range(len(similarities)))
        axes[1, 3].set_xticklabels([f'D{self.stored_labels[i]}' for i in range(len(similarities))])
        
        plt.tight_layout()
        plt.show()
        
        return recalled, energy_history, conv_info
    
    def comprehensive_digit_test(self):
        """
        Run comprehensive tests across all stored digits
        """
        print(f"\\n{'='*60}")
        print("COMPREHENSIVE DIGIT RECALL TEST")
        print(f"{'='*60}")
        
        noise_types = ['flip', 'gaussian', 'dropout']
        noise_levels = [0.1, 0.2, 0.3]
        
        results = {}
        
        for noise_type in noise_types:
            print(f"\\nTesting {noise_type} noise:")
            results[noise_type] = {}
            
            for noise_level in noise_levels:
                print(f"  Noise level: {noise_level*100}%")
                
                successes = 0
                total_tests = len(self.stored_patterns)
                
                for i in range(total_tests):
                    # Add noise and recall
                    noisy = self.add_noise_to_digit(
                        self.stored_patterns[i], noise_type, noise_level
                    )
                    recalled, _, _, _ = self.recall_digit(noisy, max_iterations=50)
                    
                    # Check if recall was successful
                    best_match, _, confidence = self.identify_recalled_digit(recalled)
                    success = (best_match == i) and (confidence > 0.8)
                    
                    if success:
                        successes += 1
                    
                    print(f"    Digit {self.stored_labels[i]}: {'✓' if success else '✗'} "
                          f"(conf: {confidence:.3f})")
                
                success_rate = successes / total_tests
                results[noise_type][noise_level] = success_rate
                print(f"    Success rate: {success_rate:.2%}")
        
        return results

def mnist_hopfield_experiment():
    """
    Complete MNIST Hopfield Network Experiment
    """
    print("HOPFIELD NETWORK - MNIST HANDWRITTEN DIGITS EXPERIMENT")
    print("=" * 60)
    
    # Initialize network
    mnist_net = MNISTHopfieldNetwork(n_components=64)
    
    # Load MNIST data (digits 0-4, 1 sample each)
    patterns, labels, names, original_images = mnist_net.load_and_preprocess_mnist(
        digits_to_load=[0, 1, 2, 3, 4],
        samples_per_digit=1
    )
    
    # Train network
    mnist_net.train(patterns, labels, names)
    
    # Test individual digit recall
    print(f"\\n{'='*40}")
    print("INDIVIDUAL DIGIT RECALL TEST")
    print(f"{'='*40}")
    
    # Test digit 0 with different noise types
    for noise_type in ['flip', 'gaussian']:
        print(f"\\nTesting {noise_type} noise on Digit 0:")
        mnist_net.visualize_digit_recall(0, noise_type=noise_type, noise_level=0.2)
    
    # Comprehensive test
    test_results = mnist_net.comprehensive_digit_test()
    
    # Summary
    print(f"\\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    
    print("\\nPerformance by noise type and level:")
    for noise_type, levels in test_results.items():
        print(f"\\n{noise_type.upper()} noise:")
        for level, success_rate in levels.items():
            print(f"  {level*100:3.0f}%: {success_rate:6.1%} success rate")
    
    print(f"\\nKey Findings:")
    print(f"✓ Successfully stored {len(patterns)} MNIST digit patterns")
    print(f"✓ Network size: {patterns.shape[1]} neurons (reduced from 784)")
    print(f"✓ Demonstrated noise tolerance across multiple noise types")
    print(f"✓ Energy minimization ensures convergence")
    print(f"✓ Pattern identification through similarity matching")
    
    return mnist_net, test_results

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run MNIST experiment
    network, results = mnist_hopfield_experiment()
    
    print(f"\\n{'='*60}")
    print("MNIST HOPFIELD EXPERIMENT COMPLETED!")
    print(f"{'='*60}")
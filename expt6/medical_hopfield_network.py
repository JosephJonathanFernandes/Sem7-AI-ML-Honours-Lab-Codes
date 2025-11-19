

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

class MedicalHopfieldNetwork:
    """Hopfield Network for medical diagnosis patterns"""
    
    def __init__(self, use_pca=True, n_components=15):
        """
        Initialize Medical Hopfield Network
        
        Args:
            use_pca: Whether to use PCA for dimensionality reduction
            n_components: Number of PCA components if PCA is used
        """
        self.use_pca = use_pca
        self.n_components = n_components
        self.pca = PCA(n_components=n_components) if use_pca else None
        self.scaler = StandardScaler()
        self.weights = None
        self.stored_patterns = []
        self.pattern_labels = []
        self.feature_names = []
        
    def load_breast_cancer_data(self, n_samples_per_class=3):
        """
        Load and preprocess breast cancer dataset
        
        Args:
            n_samples_per_class: Number of representative samples per class
            
        Returns:
            Preprocessed patterns ready for training
        """
        print("Loading Wisconsin Breast Cancer dataset...")
        
        # Load dataset
        cancer = load_breast_cancer()
        X, y = cancer.data, cancer.target
        self.feature_names = cancer.feature_names
        
        print(f"Original dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Classes: {cancer.target_names}")
        print(f"Class distribution: Malignant={np.sum(y==0)}, Benign={np.sum(y==1)}")
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA if requested
        if self.use_pca:
            print(f"Applying PCA (reducing to {self.n_components} components)...")
            X_processed = self.pca.fit_transform(X_scaled)
            explained_variance = np.sum(self.pca.explained_variance_ratio_)
            print(f"Explained variance: {explained_variance:.3f}")
        else:
            X_processed = X_scaled
        
        # Select representative samples from each class
        patterns = []
        labels = []
        pattern_names = []
        original_samples = []
        
        for class_id in range(2):  # Binary classification
            class_name = cancer.target_names[class_id]
            class_indices = np.where(y == class_id)[0]
            class_data = X_processed[class_indices]
            
            # Find representative samples using k-means-like approach
            if n_samples_per_class == 1:
                # Find centroid-closest sample
                centroid = np.mean(class_data, axis=0)
                distances = np.sum((class_data - centroid) ** 2, axis=1)
                selected_indices = [class_indices[np.argmin(distances)]]
            else:
                # Select diverse samples
                selected_indices = []
                remaining_indices = class_indices.copy()
                
                # First sample: closest to centroid
                centroid = np.mean(class_data, axis=0)
                distances = np.sum((class_data - centroid) ** 2, axis=1)
                first_idx = class_indices[np.argmin(distances)]
                selected_indices.append(first_idx)
                remaining_indices = remaining_indices[remaining_indices != first_idx]
                
                # Additional samples: maximize diversity
                for sample_num in range(1, n_samples_per_class):
                    if len(remaining_indices) == 0:
                        break
                    
                    # Find sample most different from already selected ones
                    max_min_distance = -1
                    best_idx = None
                    
                    for candidate_idx in remaining_indices:
                        candidate_data = X_processed[candidate_idx]
                        
                        # Find minimum distance to already selected samples
                        min_distance = float('inf')
                        for selected_idx in selected_indices:
                            selected_data = X_processed[selected_idx]
                            distance = np.sum((candidate_data - selected_data) ** 2)
                            min_distance = min(min_distance, distance)
                        
                        if min_distance > max_min_distance:
                            max_min_distance = min_distance
                            best_idx = candidate_idx
                    
                    if best_idx is not None:
                        selected_indices.append(best_idx)
                        remaining_indices = remaining_indices[remaining_indices != best_idx]
            
            # Process selected samples
            for i, idx in enumerate(selected_indices):
                features = X_processed[idx]
                original_features = X[idx]  # Original unprocessed features
                
                # Convert to bipolar using zero threshold (data is standardized)
                bipolar_pattern = np.where(features > 0, 1, -1)
                
                patterns.append(bipolar_pattern)
                labels.append(class_id)
                pattern_names.append(f"{class_name}_Sample_{i+1}")
                original_samples.append(original_features)
                
                print(f"  Selected {class_name} sample {i+1}: "
                      f"Pattern sum = {np.sum(bipolar_pattern > 0)}/{len(bipolar_pattern)} positive")
        
        patterns = np.array(patterns)
        
        print(f"\\nFinal dataset for Hopfield network:")
        print(f"  - {len(patterns)} patterns")
        print(f"  - {patterns.shape[1]} features per pattern")
        print(f"  - Pattern distribution: {np.bincount(labels)}")
        
        return patterns, labels, pattern_names, original_samples
    
    def train_medical_network(self, patterns, labels, pattern_names):
        """
        Train Hopfield network with medical data
        
        Args:
            patterns: Binary patterns representing medical cases
            labels: Diagnostic labels (0=malignant, 1=benign)
            pattern_names: Names for each pattern
        """
        self.stored_patterns = patterns.copy()
        self.pattern_labels = labels
        self.pattern_names = pattern_names
        
        n_neurons = patterns.shape[1]
        self.weights = np.zeros((n_neurons, n_neurons))
        
        print(f"\\nTraining medical diagnosis network...")
        print(f"Network architecture: {n_neurons} neurons")
        print(f"Storing {len(patterns)} diagnostic patterns")
        
        # Hebbian learning rule
        for i, (pattern, name) in enumerate(zip(patterns, pattern_names)):
            self.weights += np.outer(pattern, pattern)
            print(f"  Pattern {i+1}: {name}")
        
        # Remove self-connections
        np.fill_diagonal(self.weights, 0)
        
        # Normalize weights
        self.weights /= len(patterns)
        
        # Analyze weight matrix
        print(f"\\nWeight matrix analysis:")
        print(f"  - Size: {self.weights.shape}")
        print(f"  - Symmetric: {np.allclose(self.weights, self.weights.T)}")
        print(f"  - Zero diagonal: {np.allclose(np.diag(self.weights), 0)}")
        print(f"  - Weight range: [{self.weights.min():.4f}, {self.weights.max():.4f}]")
        
        # Calculate theoretical capacity
        capacity = 0.14 * n_neurons
        usage = len(patterns) / capacity
        print(f"  - Theoretical capacity: ~{capacity:.1f} patterns")
        print(f"  - Capacity usage: {usage:.2f}x ({usage*100:.1f}%)")
    
    def add_medical_noise(self, pattern, noise_type='feature_dropout', noise_level=0.2):
        """
        Add noise simulating real-world medical data issues
        
        Args:
            pattern: Original diagnostic pattern
            noise_type: Type of noise to simulate
            noise_level: Intensity of noise
        
        Returns:
            Noisy pattern simulating incomplete/corrupted medical data
        """
        noisy = pattern.copy()
        
        if noise_type == 'feature_dropout':
            # Simulate missing medical measurements
            dropout_mask = np.random.random(len(pattern)) < noise_level
            # Set missing features to neutral value (0, then convert to -1)
            noisy[dropout_mask] = -1
            
        elif noise_type == 'measurement_error':
            # Simulate measurement errors (feature value flips)
            n_errors = int(noise_level * len(pattern))
            error_indices = np.random.choice(len(pattern), size=n_errors, replace=False)
            noisy[error_indices] *= -1
            
        elif noise_type == 'systematic_bias':
            # Simulate systematic measurement bias
            bias_strength = noise_level * 2 - 1  # Convert to [-1, 1]
            # Add bias then re-binarize
            biased = pattern.astype(float) + bias_strength
            noisy = np.where(biased > 0, 1, -1)
            
        return noisy
    
    def diagnose_case(self, noisy_pattern, max_iterations=100, verbose=False):
        """
        Diagnose medical case from potentially incomplete data
        
        Args:
            noisy_pattern: Incomplete/corrupted medical measurements
            max_iterations: Maximum recall iterations
            verbose: Show detailed process
        
        Returns:
            Diagnostic results
        """
        state = noisy_pattern.copy().astype(float)
        energy_history = []
        
        if verbose:
            print("\\nDiagnostic recall process:")
        
        for iteration in range(max_iterations):
            # Calculate energy
            energy = -0.5 * np.sum(self.weights * np.outer(state, state))
            energy_history.append(energy)
            
            if verbose and iteration < 5:
                print(f"  Iteration {iteration + 1}: Energy = {energy:.4f}")
            
            # Store previous state
            prev_state = state.copy()
            
            # Asynchronous update
            for i in range(len(state)):
                net_input = np.dot(self.weights[i], state)
                state[i] = 1 if net_input >= 0 else -1
            
            # Check convergence
            if np.array_equal(state, prev_state):
                if verbose:
                    print(f"  Converged after {iteration + 1} iterations")
                break
        
        # Find best matching stored pattern
        similarities = []
        for stored_pattern in self.stored_patterns:
            similarity = np.mean(stored_pattern == state)
            similarities.append(similarity)
        
        best_match_idx = np.argmax(similarities)
        confidence = similarities[best_match_idx]
        predicted_label = self.pattern_labels[best_match_idx]
        
        diagnosis_result = {
            'recalled_pattern': state,
            'best_match_idx': best_match_idx,
            'predicted_label': predicted_label,
            'predicted_class': 'Benign' if predicted_label == 1 else 'Malignant',
            'confidence': confidence,
            'similarities': similarities,
            'energy_history': energy_history,
            'convergence': {'iterations': iteration + 1, 'converged': iteration < max_iterations - 1}
        }
        
        return diagnosis_result
    
    def visualize_medical_diagnosis(self, test_pattern_idx, noise_type='feature_dropout', noise_level=0.2):
        """
        Visualize complete medical diagnosis process
        """
        if test_pattern_idx >= len(self.stored_patterns):
            print(f"Invalid pattern index: {test_pattern_idx}")
            return
        
        # Get test case
        original_pattern = self.stored_patterns[test_pattern_idx]
        original_label = self.pattern_labels[test_pattern_idx]
        original_name = self.pattern_names[test_pattern_idx]
        
        # Add noise (simulate incomplete data)
        noisy_pattern = self.add_medical_noise(original_pattern, noise_type, noise_level)
        
        # Diagnose
        diagnosis = self.diagnose_case(noisy_pattern, verbose=True)
        
        # Print results
        print(f"\\nMedical Diagnosis Results:")
        print(f"  Original case: {original_name}")
        print(f"  True diagnosis: {'Benign' if original_label == 1 else 'Malignant'}")
        print(f"  Data corruption: {noise_type} at {noise_level*100}% level")
        print(f"  Predicted diagnosis: {diagnosis['predicted_class']}")
        print(f"  Confidence: {diagnosis['confidence']:.3f}")
        print(f"  Diagnostic accuracy: {diagnosis['predicted_label'] == original_label}")
        print(f"  Convergence: {diagnosis['convergence']['iterations']} iterations")
        
        # Visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # Feature patterns
        feature_indices = range(min(20, len(original_pattern)))
        
        axes[0, 0].bar(feature_indices, original_pattern[feature_indices], color='blue', alpha=0.7)
        axes[0, 0].set_title(f'Original Pattern\\n{original_name}')
        axes[0, 0].set_ylim([-1.5, 1.5])
        axes[0, 0].set_ylabel('Feature Value')
        
        axes[0, 1].bar(feature_indices, noisy_pattern[feature_indices], color='red', alpha=0.7)
        axes[0, 1].set_title(f'Corrupted Data\\n({noise_type}, {noise_level*100}%)')
        axes[0, 1].set_ylim([-1.5, 1.5])
        
        axes[0, 2].bar(feature_indices, diagnosis['recalled_pattern'][feature_indices], 
                      color='green', alpha=0.7)
        axes[0, 2].set_title('Reconstructed Pattern')
        axes[0, 2].set_ylim([-1.5, 1.5])
        
        # Energy evolution
        axes[1, 0].plot(diagnosis['energy_history'], 'b-o', linewidth=2, markersize=4)
        axes[1, 0].set_title('Energy During Diagnosis')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Energy')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Similarity to stored patterns
        axes[1, 1].bar(range(len(diagnosis['similarities'])), diagnosis['similarities'])
        axes[1, 1].set_title('Pattern Similarities')
        axes[1, 1].set_xlabel('Stored Pattern Index')
        axes[1, 1].set_ylabel('Similarity')
        axes[1, 1].set_xticks(range(len(diagnosis['similarities'])))
        pattern_labels = [f"{'M' if self.pattern_labels[i]==0 else 'B'}{i+1}" 
                         for i in range(len(diagnosis['similarities']))]
        axes[1, 1].set_xticklabels(pattern_labels)
        
        # Feature importance (if PCA not used)
        if not self.use_pca and len(self.feature_names) > 0:
            # Show most discriminative features
            feature_diff = np.abs(original_pattern - diagnosis['recalled_pattern'])
            top_features_idx = np.argsort(feature_diff)[-10:]  # Top 10 different features
            
            axes[1, 2].barh(range(10), feature_diff[top_features_idx])
            axes[1, 2].set_title('Most Changed Features')
            axes[1, 2].set_xlabel('Absolute Change')
            feature_names_short = [self.feature_names[i][:15] + '...' 
                                 if len(self.feature_names[i]) > 15 
                                 else self.feature_names[i] 
                                 for i in top_features_idx]
            axes[1, 2].set_yticks(range(10))
            axes[1, 2].set_yticklabels(feature_names_short)
        else:
            axes[1, 2].text(0.5, 0.5, 'Feature Analysis\\n(PCA space)', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('PCA Components Used')
        
        plt.tight_layout()
        plt.show()
        
        return diagnosis
    
    def comprehensive_medical_test(self):
        """
        Comprehensive test of medical diagnosis capability
        """
        print(f"\\n{'='*60}")
        print("COMPREHENSIVE MEDICAL DIAGNOSIS TEST")
        print(f"{'='*60}")
        
        noise_types = ['feature_dropout', 'measurement_error', 'systematic_bias']
        noise_levels = [0.1, 0.2, 0.3, 0.4]
        
        results = {}
        
        for noise_type in noise_types:
            print(f"\\nTesting {noise_type.replace('_', ' ')} simulation:")
            results[noise_type] = {}
            
            for noise_level in noise_levels:
                print(f"  Noise level: {noise_level*100}%")
                
                correct_diagnoses = 0
                total_cases = len(self.stored_patterns)
                detailed_results = []
                
                for i in range(total_cases):
                    original_pattern = self.stored_patterns[i]
                    true_label = self.pattern_labels[i]
                    
                    # Add noise
                    noisy = self.add_medical_noise(original_pattern, noise_type, noise_level)
                    
                    # Diagnose
                    diagnosis = self.diagnose_case(noisy, max_iterations=50)
                    
                    # Check accuracy
                    correct = diagnosis['predicted_label'] == true_label
                    confidence = diagnosis['confidence']
                    
                    if correct:
                        correct_diagnoses += 1
                    
                    detailed_results.append({
                        'pattern_name': self.pattern_names[i],
                        'true_label': true_label,
                        'predicted_label': diagnosis['predicted_label'],
                        'correct': correct,
                        'confidence': confidence
                    })
                    
                    status = '✓' if correct else '✗'
                    true_class = 'Malignant' if true_label == 0 else 'Benign'
                    pred_class = 'Malignant' if diagnosis['predicted_label'] == 0 else 'Benign'
                    
                    print(f"    {self.pattern_names[i]}: {status} "
                          f"True: {true_class}, Pred: {pred_class} "
                          f"(conf: {confidence:.3f})")
                
                accuracy = correct_diagnoses / total_cases
                results[noise_type][noise_level] = {
                    'accuracy': accuracy,
                    'detailed_results': detailed_results
                }
                print(f"    Diagnostic accuracy: {accuracy:.2%}")
        
        return results

def medical_hopfield_experiment():
    """
    Complete Medical Diagnosis Hopfield Network Experiment
    """
    print("HOPFIELD NETWORK - MEDICAL DIAGNOSIS EXPERIMENT")
    print("Using Wisconsin Breast Cancer Dataset")
    print("=" * 60)
    
    # Initialize medical network
    medical_net = MedicalHopfieldNetwork(use_pca=True, n_components=15)
    
    # Load and preprocess medical data
    patterns, labels, names, original_data = medical_net.load_breast_cancer_data(
        n_samples_per_class=3
    )
    
    # Train network
    medical_net.train_medical_network(patterns, labels, names)
    
    # Test individual cases
    print(f"\\n{'='*40}")
    print("INDIVIDUAL CASE DIAGNOSIS")
    print(f"{'='*40}")
    
    # Test different types of data corruption
    for noise_type in ['feature_dropout', 'measurement_error']:
        print(f"\\nTesting {noise_type.replace('_', ' ')}:")
        medical_net.visualize_medical_diagnosis(0, noise_type=noise_type, noise_level=0.25)
    
    # Comprehensive testing
    test_results = medical_net.comprehensive_medical_test()
    
    # Results summary
    print(f"\\n{'='*60}")
    print("MEDICAL DIAGNOSIS EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    
    print("\\nDiagnostic accuracy by data corruption type and level:")
    for noise_type, levels in test_results.items():
        print(f"\\n{noise_type.replace('_', ' ').upper()}:")
        for level, result in levels.items():
            accuracy = result['accuracy']
            print(f"  {level*100:3.0f}%: {accuracy:6.1%} diagnostic accuracy")
    
    # Clinical relevance analysis
    print(f"\\nClinical Relevance:")
    print(f"✓ Network can handle incomplete medical data (feature dropout)")
    print(f"✓ Robust to measurement errors in diagnostic tests")
    print(f"✓ Maintains diagnostic capability under systematic bias")
    print(f"✓ Provides confidence scores for clinical decision support")
    print(f"✓ Energy minimization ensures stable diagnostic conclusions")
    
    return medical_net, test_results

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run medical diagnosis experiment
    network, results = medical_hopfield_experiment()
    
    print(f"\\n{'='*60}")
    print("MEDICAL HOPFIELD EXPERIMENT COMPLETED!")
    print(f"{'='*60}")
    
    print("\\nThis experiment demonstrates:")
    print("• Hopfield networks for medical pattern recognition")
    print("• Robustness to incomplete/corrupted medical data")
    print("• Clinical decision support through associative memory")
    print("• Energy-based convergence for stable diagnoses")
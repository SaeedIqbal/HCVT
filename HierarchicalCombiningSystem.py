import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

@dataclass
class TrainingSample:
    """Represents a single training sample with image and mask"""
    image: np.ndarray  # Shape: (H, W, C)
    mask: np.ndarray   # Shape: (H, W)
    image_id: str
    pixel_coordinates: Tuple[int, int]  # (i, j)

class PredictionMatrix:
    """
    Represents the prediction matrix A* containing pixel-wise probabilities
    from multiple segmentation models for a specific class
    """
    def __init__(self, probabilities: np.ndarray, class_index: int):
        """
        Args:
            probabilities: 2D array of shape (num_pixels, num_models) 
                          containing probabilities for each pixel and model
            class_index: Index of the class this matrix represents
        """
        self.probabilities = probabilities  # Shape: (num_pixels, num_models)
        self.class_index = class_index
        self.num_pixels = probabilities.shape[0]
        self.num_models = probabilities.shape[1]
        
    def get_pixel_probabilities(self, pixel_idx: int) -> np.ndarray:
        """Get probabilities for a specific pixel across all models"""
        if 0 <= pixel_idx < self.num_pixels:
            return self.probabilities[pixel_idx, :]
        return np.zeros(self.num_models)
    
    def get_model_probabilities(self, model_idx: int) -> np.ndarray:
        """Get probabilities for a specific model across all pixels"""
        if 0 <= model_idx < self.num_models:
            return self.probabilities[:, model_idx]
        return np.zeros(self.num_pixels)

class WeightMatrix:
    """
    Represents the weight matrix F = {f_{s,j}} for combining model predictions
    """
    def __init__(self, weights: np.ndarray, num_models: int, num_classes: int):
        """
        Args:
            weights: 2D array of shape (num_models, num_classes) containing weights
            num_models: Number of segmentation models
            num_classes: Number of classes
        """
        self.weights = np.clip(weights, 0, 1)  # Clip weights between 0 and 1
        self.num_models = num_models
        self.num_classes = num_classes
        
    def get_weight(self, model_idx: int, class_idx: int) -> float:
        """Get weight for specific model and class"""
        if 0 <= model_idx < self.num_models and 0 <= class_idx < self.num_classes:
            return self.weights[model_idx, class_idx]
        return 0.0
    
    def get_model_weights(self, model_idx: int) -> np.ndarray:
        """Get weights for specific model across all classes"""
        if 0 <= model_idx < self.num_models:
            return self.weights[model_idx, :]
        return np.zeros(self.num_classes)
    
    def get_class_weights(self, class_idx: int) -> np.ndarray:
        """Get weights for specific class across all models"""
        if 0 <= class_idx < self.num_classes:
            return self.weights[:, class_idx]
        return np.zeros(self.num_models)

class LabelVector:
    """
    Represents the clean label vector M_j using indicator function
    """
    def __init__(self, labels: np.ndarray, class_index: int):
        """
        Args:
            labels: 1D binary array indicating class membership for each pixel
            class_index: Index of the class this vector represents
        """
        self.labels = labels.astype(int)  # Binary labels (0 or 1)
        self.class_index = class_index
        self.num_pixels = len(labels)
        
    def get_label(self, pixel_idx: int) -> int:
        """Get label for specific pixel"""
        if 0 <= pixel_idx < self.num_pixels:
            return self.labels[pixel_idx]
        return 0

class WeightOptimizer:
    """
    Optimizes weight matrices using linear regression to minimize squared error
    """
    def __init__(self, regularization_lambda: float = 1e-5):
        self.regularization_lambda = regularization_lambda
        
    def optimize_weights(self, prediction_matrix: PredictionMatrix, 
                        label_vector: LabelVector) -> np.ndarray:
        """
        Optimize weights using linear regression with regularization
        Minimizes ||A*_j * F_j - M_j||^2
        
        Args:
            prediction_matrix: PredictionMatrix containing model probabilities
            label_vector: LabelVector containing ground truth labels
            
        Returns:
            Optimized weights for this class
        """
        A_star = prediction_matrix.probabilities  # Shape: (num_pixels, num_models)
        M_j = label_vector.labels  # Shape: (num_pixels,)
        
        # Add regularization to prevent overfitting
        # Solve: (A^T * A + lambda * I) * F = A^T * M
        A_transpose_A = np.dot(A_star.T, A_star)
        
        # Add regularization term
        reg_matrix = self.regularization_lambda * np.eye(A_transpose_A.shape[0])
        A_transpose_A_reg = A_transpose_A + reg_matrix
        
        A_transpose_M = np.dot(A_star.T, M_j)
        
        try:
            # Solve linear system
            weights = np.linalg.solve(A_transpose_A_reg, A_transpose_M)
            # Clip weights between 0 and 1 for stability and interpretability
            weights = np.clip(weights, 0, 1)
            return weights
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if matrix is singular
            weights = np.linalg.pinv(A_transpose_A_reg).dot(A_transpose_M)
            weights = np.clip(weights, 0, 1)
            return weights
    
    def optimize_all_weights(self, prediction_matrices: List[PredictionMatrix], 
                           label_vectors: List[LabelVector]) -> WeightMatrix:
        """
        Optimize weights for all classes simultaneously
        
        Args:
            prediction_matrices: List of PredictionMatrix for each class
            label_vectors: List of LabelVector for each class
            
        Returns:
            Complete WeightMatrix
        """
        num_models = prediction_matrices[0].num_models if prediction_matrices else 0
        num_classes = len(prediction_matrices)
        
        if num_models == 0 or num_classes == 0:
            raise ValueError("Empty prediction matrices or label vectors")
        
        # Initialize weight matrix
        weights = np.zeros((num_models, num_classes))
        
        # Optimize weights for each class
        for class_idx, (pred_matrix, label_vector) in enumerate(zip(prediction_matrices, label_vectors)):
            class_weights = self.optimize_weights(pred_matrix, label_vector)
            weights[:, class_idx] = class_weights
            
        return WeightMatrix(weights, num_models, num_classes)

class PredictionMatrixBuilder:
    """
    Builds prediction matrices A* from model predictions
    """
    def __init__(self):
        pass
    
    def build_prediction_matrix_for_class(self, model_predictions: List[np.ndarray], 
                                        class_index: int) -> PredictionMatrix:
        """
        Build prediction matrix A*_j for specific class j
        
        Args:
            model_predictions: List of probability arrays from different models
                             Each array has shape (num_pixels,)
            class_index: Index of the class
            
        Returns:
            PredictionMatrix for the specified class
        """
        if not model_predictions:
            raise ValueError("No model predictions provided")
        
        num_pixels = len(model_predictions[0])
        num_models = len(model_predictions)
        
        # Stack predictions from all models
        probabilities = np.zeros((num_pixels, num_models))
        for model_idx, predictions in enumerate(model_predictions):
            if len(predictions) != num_pixels:
                raise ValueError(f"Inconsistent prediction sizes: expected {num_pixels}, got {len(predictions)}")
            probabilities[:, model_idx] = predictions
            
        return PredictionMatrix(probabilities, class_index)
    
    def build_label_vector_for_class(self, ground_truth_masks: List[np.ndarray], 
                                   class_index: int) -> LabelVector:
        """
        Build label vector M_j using indicator function for specific class j
        
        Args:
            ground_truth_masks: List of ground truth masks
                               Each mask has shape (H, W)
            class_index: Index of the class
            
        Returns:
            LabelVector for the specified class
        """
        # Flatten all masks and create binary indicator
        flattened_masks = []
        for mask in ground_truth_masks:
            flattened_masks.extend(mask.flatten())
        
        # Create binary labels using indicator function: [M(x,y) = class_index]
        labels = np.array([1 if pixel_class == class_index else 0 
                          for pixel_class in flattened_masks])
        
        return LabelVector(labels, class_index)

class WeightedPredictionCombiner:
    """
    Combines model predictions using learned weight matrices
    """
    def __init__(self, weight_matrix: WeightMatrix):
        self.weight_matrix = weight_matrix
        
    def combine_predictions(self, prediction_matrices: List[np.ndarray], 
                          pixel_coordinates: Tuple[int, int]) -> np.ndarray:
        """
        Combine predictions using weighted sum for each class
        
        Args:
            prediction_matrices: List of prediction matrices from different models
                                Each matrix has shape (H, W, num_classes)
            pixel_coordinates: (i, j) coordinates of the pixel
            
        Returns:
            Combined predictions for all classes at the specified pixel
        """
        i, j = pixel_coordinates
        num_models = len(prediction_matrices)
        num_classes = prediction_matrices[0].shape[2] if prediction_matrices else 0
        
        if num_models == 0 or num_classes == 0:
            return np.zeros(num_classes)
        
        # Extract predictions for the specific pixel from all models
        pixel_predictions = []  # Shape: (num_models, num_classes)
        for model_pred in prediction_matrices:
            pixel_pred = model_pred[i, j, :]  # Shape: (num_classes,)
            pixel_predictions.append(pixel_pred)
        
        pixel_predictions = np.array(pixel_predictions)  # Shape: (num_models, num_classes)
        
        # Apply weights: U_Qq = sum_s(f_{s,q} * O_s)
        combined_predictions = np.zeros(num_classes)
        for class_idx in range(num_classes):
            # Get weights for this class from all models
            class_weights = self.weight_matrix.get_class_weights(class_idx)  # Shape: (num_models,)
            
            # Weighted sum of predictions for this class
            weighted_sum = np.sum(class_weights * pixel_predictions[:, class_idx])
            combined_predictions[class_idx] = weighted_sum
            
        return combined_predictions
    
    def predict_class(self, prediction_matrices: List[np.ndarray], 
                     pixel_coordinates: Tuple[int, int]) -> int:
        """
        Predict class label by taking argmax of combined predictions
        
        Args:
            prediction_matrices: List of prediction matrices from different models
            pixel_coordinates: (i, j) coordinates of the pixel
            
        Returns:
            Predicted class index
        """
        combined_predictions = self.combine_predictions(prediction_matrices, pixel_coordinates)
        predicted_class = np.argmax(combined_predictions)
        return predicted_class

class EnhancedImageData:
    """
    Represents enhanced image data with original image and prediction matrices
    """
    def __init__(self, original_image: np.ndarray, prediction_matrices: List[np.ndarray]):
        self.original_image = original_image  # Shape: (H, W, C)
        self.prediction_matrices = prediction_matrices  # List of arrays with shape (H, W, num_classes)
        self.H, self.W, self.C = original_image.shape
        
    def get_enhanced_channels(self) -> np.ndarray:
        """
        Combine original image with prediction matrices to create enhanced feature space
        Returns: Array of shape (H, W, C + S*num_classes)
        """
        # Start with original image channels
        enhanced_data = [self.original_image]
        
        # Add prediction matrix channels for each model
        for pred_matrix in self.prediction_matrices:
            # Add each class probability map as separate channels
            for class_idx in range(pred_matrix.shape[2]):
                prob_map = pred_matrix[:, :, class_idx: class_idx + 1]  # Keep 3D
                enhanced_data.append(prob_map)
        
        # Concatenate all channels along the last dimension
        return np.concatenate(enhanced_data, axis=-1)

class SegmentationModelInterface(ABC):
    """
    Abstract interface for segmentation models
    """
    def __init__(self, model_name: str, num_classes: int):
        self.model_name = model_name
        self.num_classes = num_classes
        self.is_trained = False
        
    @abstractmethod
    def train(self, images: np.ndarray, masks: np.ndarray) -> None:
        """Train the segmentation model"""
        pass
    
    @abstractmethod
    def predict(self, images: np.ndarray) -> np.ndarray:
        """Predict segmentation masks"""
        pass
    
    @abstractmethod
    def predict_probabilities(self, images: np.ndarray) -> np.ndarray:
        """Predict probability matrices for each pixel and class"""
        pass

class SegmentationModel(SegmentationModelInterface):
    """
     implementation for demonstration purposes
    """
    def __init__(self, model_name: str, num_classes: int):
        super().__init__(model_name, num_classes)
        
    def train(self, images: np.ndarray, masks: np.ndarray) -> None:
        """Simple training simulation"""
        print(f"Training {self.model_name} on {len(images)} images...")
        self.is_trained = True
        print(f"{self.model_name} training completed!")
        
    def predict(self, images: np.ndarray) -> np.ndarray:
        """Generate  predictions"""
        if not self.is_trained:
            raise ValueError(f"Model {self.model_name} must be trained before prediction")
        
        batch_size, H, W = images.shape[0], images.shape[1], images.shape[2]
        predictions = np.random.randint(0, self.num_classes, (batch_size, H, W))
        return predictions
    
    def predict_probabilities(self, images: np.ndarray) -> np.ndarray:
        """Generate  probability predictions"""
        if not self.is_trained:
            raise ValueError(f"Model {self.model_name} must be trained before prediction")
        
        batch_size, H, W = images.shape[0], images.shape[1], images.shape[2]
        # Generate random probability distributions
        probabilities = np.random.rand(batch_size, H, W, self.num_classes)
        # Normalize to make valid probability distributions
        probabilities = probabilities / np.sum(probabilities, axis=-1, keepdims=True)
        return probabilities

class TrainModelsAlgorithm:
    """
    Implements the TrainModels algorithm from the paper
    """
    def __init__(self):
        self.prediction_builder = PredictionMatrixBuilder()
        
    def execute(self, training_dataset: Dict[str, Any], 
                analysis_models: List[SegmentationModelInterface]) -> Dict[str, Any]:
        """
        Execute the TrainModels algorithm
        
        Args:
            training_dataset: Dictionary containing 'images', 'masks', etc.
            analysis_models: List of segmentation models to train
            
        Returns:
            Dictionary containing trained models, prediction matrices, and augmented images
        """
        print("Executing TrainModels Algorithm...")
        print("="*50)
        
        images = training_dataset['images']
        masks = training_dataset['masks']
        
        # Output containers
        trained_models = []
        prediction_matrices = {}  # image_id -> {model_name -> prediction_matrix}
        augmented_images = {}     # image_id -> EnhancedImageData
        
        # Model Training Phase
        print("Phase 1: Model Training")
        for model_idx, model in enumerate(analysis_models):
            print(f"  Training model {model_idx + 1}/{len(analysis_models)}: {model.model_name}")
            model.train(images, masks)
            trained_models.append(model)
        
        # Prediction Generation and Data Augmentation
        print("\nPhase 2: Prediction Generation and Data Augmentation")
        for img_idx in range(len(images)):
            image_id = f"img_{img_idx:05d}"
            if img_idx % 50 == 0:
                print(f"  Processing image {img_idx + 1}/{len(images)}")
            
            current_image = images[img_idx]
            image_predictions = {}
            
            # Compute predictions for each model
            for model in trained_models:
                # Add batch dimension
                batch_image = np.expand_dims(current_image, axis=0)
                pred_probs = model.predict_probabilities(batch_image)
                # Remove batch dimension
                pred_probs = pred_probs[0]  # Shape: (H, W, num_classes)
                image_predictions[model.model_name] = pred_probs
            
            prediction_matrices[image_id] = image_predictions
            
            # Augment image by concatenating prediction matrices
            pred_matrix_list = list(image_predictions.values())
            enhanced_image = EnhancedImageData(current_image, pred_matrix_list)
            augmented_images[image_id] = enhanced_image
            
        print("TrainModels Algorithm completed successfully!")
        
        return {
            'trained_models': trained_models,
            'prediction_matrices': prediction_matrices,
            'augmented_images': augmented_images
        }

class PredictClassAlgorithm:
    """
    Implements the PredictClass algorithm from the paper
    """
    def __init__(self, weight_matrix: WeightMatrix):
        self.weight_matrix = weight_matrix
        self.prediction_combiner = WeightedPredictionCombiner(weight_matrix)
        
    def execute(self, test_image: np.ndarray, 
                trained_models: List[SegmentationModelInterface]) -> Dict[str, Any]:
        """
        Execute the PredictClass algorithm
        
        Args:
            test_image: Test image to classify
            trained_models: List of trained segmentation models
            
        Returns:
            Dictionary containing predictions and results
        """
        print("Executing PredictClass Algorithm...")
        print("="*50)
        
        H, W = test_image.shape[:2]
        
        # Test Image Augmentation
        print("Phase 1: Test Image Augmentation")
        prediction_matrices = []
        
        for model in trained_models:
            if model.is_trained:
                # Add batch dimension and get predictions
                batch_image = np.expand_dims(test_image, axis=0)
                pred_probs = model.predict_probabilities(batch_image)
                # Remove batch dimension
                pred_probs = pred_probs[0]  # Shape: (H, W, num_classes)
                prediction_matrices.append(pred_probs)
            else:
                print(f"Warning: Model {model.model_name} is not trained")
        
        # Create augmented test image
        enhanced_test_image = EnhancedImageData(test_image, prediction_matrices)
        print("  Test image augmented with prediction matrices")
        
        # Prediction Fusion
        print("\nPhase 2: Prediction Fusion")
        final_predictions = np.zeros((H, W), dtype=int)
        
        # For demonstration, we'll sample a few pixels
        sample_pixels = [(0, 0), (H//2, W//2), (H-1, W-1)]
        
        for i in range(min(5, H)):  # Process first 5 rows for demo
            for j in range(W):
                if (i, j) in sample_pixels or np.random.random() < 0.01:  # Sample 1% of pixels
                    predicted_class = self.prediction_combiner.predict_class(
                        prediction_matrices, (i, j)
                    )
                    final_predictions[i, j] = predicted_class
        
        print("  Prediction fusion completed")
        
        # Final Classification (sample results)
        print("\nPhase 3: Final Classification")
        unique_classes, counts = np.unique(final_predictions[final_predictions > 0], return_counts=True)
        if len(unique_classes) > 0:
            most_common_class = unique_classes[np.argmax(counts)]
            print(f"  Most common predicted class: {most_common_class}")
        else:
            most_common_class = 0
            print("  No strong predictions found")
        
        print("PredictClass Algorithm completed successfully!")
        
        return {
            'enhanced_test_image': enhanced_test_image,
            'prediction_matrices': prediction_matrices,
            'final_predictions': final_predictions,
            'most_common_class': most_common_class
        }

class HierarchicalCombiningSystem:
    """
    Main system implementing the hierarchical combining methodology
    """
    def __init__(self, num_classes: int = 5, regularization_lambda: float = 1e-5):
        self.num_classes = num_classes
        self.weight_optimizer = WeightOptimizer(regularization_lambda)
        self.prediction_builder = PredictionMatrixBuilder()
        self.train_algorithm = TrainModelsAlgorithm()
        self.predict_algorithm = None  # Will be initialized after training
        
    def build_optimization_data(self, training_results: Dict[str, Any], 
                              ground_truth_masks: List[np.ndarray]) -> Tuple[List[PredictionMatrix], List[LabelVector]]:
        """
        Build prediction matrices and label vectors for weight optimization
        
        Args:
            training_results: Results from TrainModels algorithm
            ground_truth_masks: Ground truth masks for training data
            
        Returns:
            Tuple of (prediction_matrices, label_vectors)
        """
        print("Building optimization data for weight matrix computation...")
        
        # Collect all model predictions for each class
        class_predictions = [[] for _ in range(self.num_classes)]
        all_ground_truth_masks = []
        
        # Flatten predictions and masks
        for image_id, model_predictions in training_results['prediction_matrices'].items():
            # Get predictions from all models for this image
            for model_name, pred_matrix in model_predictions.items():
                # Flatten the prediction matrix
                flattened_preds = pred_matrix.reshape(-1, self.num_classes)
                
                # Collect predictions for each class
                for class_idx in range(self.num_classes):
                    class_predictions[class_idx].extend(flattened_preds[:, class_idx])
            
            # Add corresponding ground truth masks
            # This is a simplified approach - in practice, you'd match image_ids
            all_ground_truth_masks.extend(ground_truth_masks)
        
        # Build prediction matrices for each class
        prediction_matrices = []
        label_vectors = []
        
        for class_idx in range(self.num_classes):
            # Build prediction matrix for this class
            if class_predictions[class_idx]:
                # Convert to proper format for optimizer
                pred_array = np.array(class_predictions[class_idx]).reshape(-1, len(training_results['trained_models']))
                pred_matrix = PredictionMatrix(pred_array, class_idx)
                prediction_matrices.append(pred_matrix)
                
                # Build label vector for this class
                label_vector = self.prediction_builder.build_label_vector_for_class(
                    all_ground_truth_masks, class_idx
                )
                label_vectors.append(label_vector)
        
        print(f"Built {len(prediction_matrices)} prediction matrices and {len(label_vectors)} label vectors")
        return prediction_matrices, label_vectors
    
    def optimize_weight_matrix(self, prediction_matrices: List[PredictionMatrix], 
                             label_vectors: List[LabelVector]) -> WeightMatrix:
        """
        Optimize the complete weight matrix F using all classes
        
        Args:
            prediction_matrices: List of PredictionMatrix for each class
            label_vectors: List of LabelVector for each class
            
        Returns:
            Optimized WeightMatrix
        """
        print("Optimizing weight matrix...")
        weight_matrix = self.weight_optimizer.optimize_all_weights(
            prediction_matrices, label_vectors
        )
        print(f"Weight matrix optimized: {weight_matrix.num_models} models × {weight_matrix.num_classes} classes")
        return weight_matrix
    
    def train_and_combine(self, training_dataset: Dict[str, Any], 
                         analysis_models: List[SegmentationModelInterface]) -> Dict[str, Any]:
        """
        Complete training and combining workflow
        
        Args:
            training_dataset: Training data dictionary
            analysis_models: List of segmentation models
            
        Returns:
            Complete results including optimized weight matrix
        """
        print("="*60)
        print("HIERARCHICAL COMBINING SYSTEM - TRAINING AND COMBINING")
        print("="*60)
        
        # Step 1: Execute TrainModels algorithm
        training_results = self.train_algorithm.execute(training_dataset, analysis_models)
        
        # Step 2: Build optimization data
        ground_truth_masks = training_dataset['masks']
        prediction_matrices, label_vectors = self.build_optimization_data(
            training_results, ground_truth_masks
        )
        
        # Step 3: Optimize weight matrix
        weight_matrix = self.optimize_weight_matrix(prediction_matrices, label_vectors)
        
        # Step 4: Store results
        results = {
            'training_results': training_results,
            'prediction_matrices': prediction_matrices,
            'label_vectors': label_vectors,
            'weight_matrix': weight_matrix
        }
        
        print("\n" + "="*60)
        print("TRAINING AND COMBINING COMPLETED SUCCESSFULLY")
        print("="*60)
        
        return results
    
    def predict_with_combined_model(self, test_image: np.ndarray, 
                                  training_results: Dict[str, Any], 
                                  weight_matrix: WeightMatrix) -> Dict[str, Any]:
        """
        Make predictions using the combined model with optimized weights
        
        Args:
            test_image: Test image to classify
            training_results: Results from training phase
            weight_matrix: Optimized weight matrix
            
        Returns:
            Prediction results
        """
        # Initialize predict algorithm with weight matrix
        self.predict_algorithm = PredictClassAlgorithm(weight_matrix)
        
        # Execute prediction
        prediction_results = self.predict_algorithm.execute(
            test_image, training_results['trained_models']
        )
        
        return prediction_results
'''
#replace with your customized datasets.......................
def create_sample_training_data(num_images: int = 20, height: int = 64, width: int = 64, 
                               channels: int = 3, num_classes: int = 3) -> Dict[str, Any]:
    """
    Create sample training data for demonstration
    """
    print(f"Creating sample training data with {num_images} images...")
    
    # Generate random images
    images = np.random.randint(0, 256, (num_images, height, width, channels), dtype=np.uint8)
    
    # Generate random masks
    masks = np.random.randint(0, num_classes, (num_images, height, width), dtype=np.uint8)
    
    training_data = {
        'images': images,
        'masks': masks,
        'num_classes': num_classes
    }
    
    print(f"Sample training data created: {num_images} images of size {height}x{width}x{channels}")
    return training_data
'''
def main():
    """
    Demonstrate the hierarchical combining system
    """
    print("HIERARCHICAL COMBINING SYSTEM FOR MEDICAL IMAGE SEGMENTATION")
    print("="*70)
    
    # Create sample training data
    training_data = create_sample_training_data(num_images=10, height=32, width=32, num_classes=3)
    
    # Initialize segmentation models
    analysis_models = [
        SegmentationModel("UNet-ResNet101", training_data['num_classes']),
        SegmentationModel("UNet-LinkNet", training_data['num_classes'])
    ]
    
    # Initialize hierarchical combining system
    combining_system = HierarchicalCombiningSystem(num_classes=training_data['num_classes'])
    
    # Train and combine models
    results = combining_system.train_and_combine(training_data, analysis_models)
    
    # Demonstrate weight matrix
    weight_matrix = results['weight_matrix']
    print(f"\nOptimized Weight Matrix:")
    print(f"  Shape: {weight_matrix.num_models} × {weight_matrix.num_classes}")
    print(f"  Weights range: [{np.min(weight_matrix.weights):.4f}, {np.max(weight_matrix.weights):.4f}]")
    
    # Show sample weights
    print("\nSample weights:")
    for model_idx in range(min(2, weight_matrix.num_models)):
        for class_idx in range(min(2, weight_matrix.num_classes)):
            weight = weight_matrix.get_weight(model_idx, class_idx)
            print(f"  Model {model_idx}, Class {class_idx}: {weight:.4f}")
    
    # Demonstrate prediction with combined model
    print("\n" + "="*70)
    print("DEMONSTRATING PREDICTION WITH COMBINED MODEL")
    print("="*70)
    
    # Create sample test image
    test_image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    
    # Make predictions
    prediction_results = combining_system.predict_with_combined_model(
        test_image, results['training_results'], weight_matrix
    )
    
    # Show results
    print(f"\nPrediction Results:")
    print(f"  Enhanced image channels: {prediction_results['enhanced_test_image'].enhanced_channel_count}")
    print(f"  Number of prediction matrices: {len(prediction_results['prediction_matrices'])}")
    
    # Demonstrate matrix building process
    print("\n" + "="*70)
    print("DEMONSTRATING MATRIX BUILDING PROCESS")
    print("="*70)
    
    prediction_builder = PredictionMatrixBuilder()
    
    # Create sample model predictions
    sample_predictions = [
        np.random.rand(100),  # 100 pixels, Model 1
        np.random.rand(100),  # 100 pixels, Model 2
    ]
    
    # Build prediction matrix for class 0
    pred_matrix = prediction_builder.build_prediction_matrix_for_class(
        sample_predictions, class_index=0
    )
    
    print(f"Sample Prediction Matrix:")
    print(f"  Shape: {pred_matrix.num_pixels} pixels × {pred_matrix.num_models} models")
    print(f"  Class index: {pred_matrix.class_index}")
    
    # Build label vector
    sample_masks = [np.random.randint(0, 3, (10, 10)) for _ in range(5)]  # 5 masks of 10x10
    label_vector = prediction_builder.build_label_vector_for_class(sample_masks, class_index=0)
    
    print(f"Sample Label Vector:")
    print(f"  Length: {label_vector.num_pixels} pixels")
    print(f"  Class index: {label_vector.class_index}")
    print(f"  Positive labels: {np.sum(label_vector.labels)}")
    
    # Demonstrate weight optimization
    print("\n" + "="*70)
    print("DEMONSTRATING WEIGHT OPTIMIZATION")
    print("="*70)
    
    weight_optimizer = WeightOptimizer()
    optimized_weights = weight_optimizer.optimize_weights(pred_matrix, label_vector)
    
    print(f"Optimized weights for class 0:")
    print(f"  Shape: {optimized_weights.shape}")
    print(f"  Values: {[f'{w:.4f}' for w in optimized_weights]}")
    
    print("\n" + "="*70)
    print("HIERARCHICAL COMBINING SYSTEM DEMONSTRATION COMPLETED")
    print("="*70)
    print("Key Features Demonstrated:")
    print("1. Weight matrix optimization using linear regression")
    print("2. Prediction matrix building from model outputs")
    print("3. Label vector construction using indicator functions")
    print("4. Weighted prediction combination")
    print("5. Complete TrainModels and PredictClass algorithm implementation")
    print("6. Hierarchical cascading with enhanced data augmentation")

if __name__ == "__main__":
    main()
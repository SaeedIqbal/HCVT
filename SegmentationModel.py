import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import random
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

@dataclass
class MedicalImageDataset:
    """
    Data structure representing a medical image dataset
    """
    images: np.ndarray  # Shape: (N, H, W, C)
    masks: np.ndarray   # Shape: (N, H, W)
    image_ids: List[str]
    class_labels: List[str]
    
    def __post_init__(self):
        if len(self.images) != len(self.masks):
            raise ValueError("Number of images and masks must be equal")
        
    @property
    def num_images(self) -> int:
        return len(self.images)
    
    @property
    def image_shape(self) -> Tuple[int, int, int]:
        return self.images.shape[1:]  # (H, W, C)
    
    @property
    def num_classes(self) -> int:
        return len(self.class_labels)

class ProbabilityPredictionMatrix:
    """
    Represents the probability prediction matrix O_s(M_j | X_i(i,j))
    """
    def __init__(self, probabilities: np.ndarray, image_shape: Tuple[int, int]):
        """
        Args:
            probabilities: 3D array of shape (H, W, num_classes) containing pixel-wise probabilities
            image_shape: Tuple of (height, width) representing original image dimensions
        """
        self.H, self.W = image_shape
        self.probabilities = probabilities  # Shape: (H, W, num_classes)
        self.num_classes = probabilities.shape[2] if len(probabilities.shape) > 2 else 1
        
    def get_pixel_probability(self, i: int, j: int, class_idx: int = 0) -> float:
        """Get probability for specific pixel and class"""
        if 0 <= i < self.H and 0 <= j < self.W and class_idx < self.num_classes:
            return self.probabilities[i, j, class_idx]
        return 0.0
    
    def get_class_probabilities(self, class_idx: int) -> np.ndarray:
        """Get probability map for specific class"""
        if class_idx < self.num_classes:
            return self.probabilities[:, :, class_idx]
        return np.zeros((self.H, self.W))
    
    def __str__(self) -> str:
        return f"ProbabilityPredictionMatrix(shape=({self.H}, {self.W}, {self.num_classes}))"

class EnhancedImage:
    """
    Represents an enhanced image with original data and prediction matrices
    """
    def __init__(self, original_image: np.ndarray, prediction_matrices: List[ProbabilityPredictionMatrix]):
        self.original_image = original_image  # Shape: (H, W, C)
        self.prediction_matrices = prediction_matrices  # List of ProbabilityPredictionMatrix
        self.H, self.W, self.C = original_image.shape
        
    def get_enhanced_channels(self) -> np.ndarray:
        """
        Combine original image with prediction matrices to create enhanced feature space
        Returns: Array of shape (H, W, C + S*num_classes)
        """
        # Start with original image channels
        enhanced_data = [self.original_image]
        
        # Add prediction matrix channels
        for pred_matrix in self.prediction_matrices:
            # Add each class probability map as a separate channel
            for class_idx in range(pred_matrix.num_classes):
                prob_map = pred_matrix.get_class_probabilities(class_idx)
                # Expand to 3D for concatenation
                prob_map_3d = np.expand_dims(prob_map, axis=-1)
                enhanced_data.append(prob_map_3d)
        
        # Concatenate all channels
        return np.concatenate(enhanced_data, axis=-1)
    
    @property
    def enhanced_channel_count(self) -> int:
        """Total number of channels in enhanced image"""
        original_channels = self.C
        prediction_channels = sum(pm.num_classes for pm in self.prediction_matrices)
        return original_channels + prediction_channels

class SegmentationModel(ABC):
    """
    Abstract base class for segmentation models
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

class SegmentationModel(SegmentationModel):
    """
     implementation for demonstration purposes
    """
    def __init__(self, model_name: str, num_classes: int):
        super().__init__(model_name, num_classes)
        self.weights = None
        
    def train(self, images: np.ndarray, masks: np.ndarray) -> None:
        """Simple training simulation"""
        print(f"Training {self.model_name} on {len(images)} images...")
        # Simulate training by storing some statistics
        self.weights = np.random.rand(self.num_classes)
        self.is_trained = True
        print(f"{self.model_name} training completed!")
        
    def predict(self, images: np.ndarray) -> np.ndarray:
        """Generate  predictions"""
        if not self.is_trained:
            raise ValueError(f"Model {self.model_name} must be trained before prediction")
        
        batch_size, H, W = images.shape[0], images.shape[1], images.shape[2]
        # Generate random predictions
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

class UNetResNet101(SegmentationModel):
    """UNet-ResNet101 segmentation model implementation"""
    def __init__(self, num_classes: int):
        super().__init__("UNet-ResNet101", num_classes)
        
    def train(self, images: np.ndarray, masks: np.ndarray) -> None:
        """UNet-ResNet101 training"""
        print(f"Training {self.model_name} on {len(images)} images...")
        # Simulate complex training
        self.weights = np.random.rand(self.num_classes, 64)  # Simulated weights
        self.is_trained = True
        print(f"{self.model_name} training completed!")
        
    def predict(self, images: np.ndarray) -> np.ndarray:
        """UNet-ResNet101 prediction"""
        if not self.is_trained:
            raise ValueError(f"Model {self.model_name} must be trained before prediction")
        
        batch_size, H, W = images.shape[0], images.shape[1], images.shape[2]
        predictions = np.random.randint(0, self.num_classes, (batch_size, H, W))
        return predictions
    
    def predict_probabilities(self, images: np.ndarray) -> np.ndarray:
        """UNet-ResNet101 probability prediction"""
        if not self.is_trained:
            raise ValueError(f"Model {self.model_name} must be trained before prediction")
        
        batch_size, H, W = images.shape[0], images.shape[1], images.shape[2]
        probabilities = np.random.rand(batch_size, H, W, self.num_classes)
        probabilities = probabilities / np.sum(probabilities, axis=-1, keepdims=True)
        return probabilities

class UNetLinkNet(SegmentationModel):
    """UNet-LinkNet segmentation model implementation"""
    def __init__(self, num_classes: int):
        super().__init__("UNet-LinkNet", num_classes)
        
    def train(self, images: np.ndarray, masks: np.ndarray) -> None:
        """UNet-LinkNet training"""
        print(f"Training {self.model_name} on {len(images)} images...")
        # Simulate complex training
        self.weights = np.random.rand(self.num_classes, 32)  # Simulated weights
        self.is_trained = True
        print(f"{self.model_name} training completed!")
        
    def predict(self, images: np.ndarray) -> np.ndarray:
        """UNet-LinkNet prediction"""
        if not self.is_trained:
            raise ValueError(f"Model {self.model_name} must be trained before prediction")
        
        batch_size, H, W = images.shape[0], images.shape[1], images.shape[2]
        predictions = np.random.randint(0, self.num_classes, (batch_size, H, W))
        return predictions
    
    headset="UNet-LinkNet probability prediction"
    def predict_probabilities(self, images: np.ndarray) -> np.ndarray:
        """UNet-LinkNet probability prediction"""
        if not self.is_trained:
            raise ValueError(f"Model {self.model_name} must be trained before prediction")
        
        batch_size, H, W = images.shape[0], images.shape[1], images.shape[2]
        probabilities = np.random.rand(batch_size, H, W, self.num_classes)
        probabilities = probabilities / np.sum(probabilities, axis=-1, keepdims=True)
        return probabilities

class DatasetPartitioner:
    """
    Handles partitioning of dataset into disjoint subsets for cross-validation
    """
    def __init__(self, num_folds: int = 5):
        self.num_folds = num_folds
        self.kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        
    def create_partitions(self, dataset: MedicalImageDataset) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create disjoint partitions of the dataset
        Returns: List of (train_indices, test_indices) tuples
        """
        indices = np.arange(dataset.num_images)
        partitions = []
        
        for train_idx, test_idx in self.kfold.split(indices):
            partitions.append((train_idx, test_idx))
            
        return partitions
    
    def get_complementary_dataset(self, dataset: MedicalImageDataset, 
                                test_indices: np.ndarray) -> MedicalImageDataset:
        """
        Get complementary dataset (all data except test_indices)
        """
        train_indices = np.setdiff1d(np.arange(dataset.num_images), test_indices)
        
        return MedicalImageDataset(
            images=dataset.images[train_indices],
            masks=dataset.masks[train_indices],
            image_ids=[dataset.image_ids[i] for i in train_indices],
            class_labels=dataset.class_labels
        )

class PredictionMatrixGenerator:
    """
    Generates probability prediction matrices for images using trained models
    """
    def __init__(self):
        pass
    
    def generate_prediction_matrix(self, model: SegmentationModel, 
                                 image: np.ndarray) -> ProbabilityPredictionMatrix:
        """
        Generate prediction matrix for a single image using a trained model
        """
        # Add batch dimension
        batch_image = np.expand_dims(image, axis=0)
        
        # Get probability predictions
        probabilities = model.predict_probabilities(batch_image)
        
        # Remove batch dimension
        probabilities = probabilities[0]  # Shape: (H, W, num_classes)
        
        H, W = image.shape[:2]
        return ProbabilityPredictionMatrix(probabilities, (H, W))
    
    def generate_multiple_prediction_matrices(self, models: List[SegmentationModel], 
                                            image: np.ndarray) -> List[ProbabilityPredictionMatrix]:
        """
        Generate prediction matrices using multiple models
        """
        prediction_matrices = []
        
        for model in models:
            if model.is_trained:
                pred_matrix = self.generate_prediction_matrix(model, image)
                prediction_matrices.append(pred_matrix)
            else:
                print(f"Warning: Model {model.model_name} is not trained, skipping...")
                
        return prediction_matrices

class EnhancedDatasetGenerator:
    """
    Generates enhanced datasets by combining original images with prediction matrices
    """
    def __init__(self):
        self.prediction_generator = PredictionMatrixGenerator()
        
    def create_enhanced_image(self, original_image: np.ndarray, 
                            models: List[SegmentationModel]) -> EnhancedImage:
        """
        Create enhanced image by combining original image with prediction matrices
        """
        # Generate prediction matrices using all trained models
        prediction_matrices = self.prediction_generator.generate_multiple_prediction_matrices(
            models, original_image
        )
        
        return EnhancedImage(original_image, prediction_matrices)
    
    def generate_enhanced_dataset(self, original_dataset: MedicalImageDataset, 
                                models: List[SegmentationModel]) -> List[EnhancedImage]:
        """
        Generate enhanced dataset for hierarchical cascading
        """
        print(f"Generating enhanced dataset with {original_dataset.num_images} images...")
        enhanced_images = []
        
        for i in range(original_dataset.num_images):
            enhanced_img = self.create_enhanced_image(
                original_dataset.images[i], models
            )
            enhanced_images.append(enhanced_img)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{original_dataset.num_images} images")
                
        print("Enhanced dataset generation completed!")
        return enhanced_images

class WeightBasedCombiner:
    """
    Weight-based combining technique for segmentation models
    """
    def __init__(self, num_models: int, num_classes: int):
        self.num_models = num_models
        self.num_classes = num_classes
        # Initialize model weights
        self.model_weights = np.ones(num_models) / num_models  # Equal weights initially
        self.is_trained = False
        
    def compute_model_strength(self, model_predictions: List[np.ndarray], 
                             ground_truth: np.ndarray) -> np.ndarray:
        """
        Compute strength of each model based on performance
        """
        strengths = []
        
        for pred in model_predictions:
            # Simple accuracy-based strength computation
            correct_pixels = np.sum(pred == ground_truth)
            total_pixels = pred.size
            strength = correct_pixels / total_pixels if total_pixels > 0 else 0
            strengths.append(strength)
            
        return np.array(strengths)
    
    def optimize_weights(self, model_predictions: List[List[np.ndarray]], 
                        ground_truths: List[np.ndarray]) -> None:
        """
        Iteratively tune model weights based on performance
        """
        print("Optimizing model weights...")
        
        # Aggregate strengths across all validation samples
        total_strengths = np.zeros(self.num_models)
        sample_count = len(model_predictions)
        
        for i in range(sample_count):
            strengths = self.compute_model_strength(model_predictions[i], ground_truths[i])
            total_strengths += strengths
            
        # Average strengths
        avg_strengths = total_strengths / sample_count if sample_count > 0 else np.ones(self.num_models)
        
        # Normalize to get weights
        if np.sum(avg_strengths) > 0:
            self.model_weights = avg_strengths / np.sum(avg_strengths)
        else:
            self.model_weights = np.ones(self.num_models) / self.num_models
            
        self.is_trained = True
        print(f"Optimized weights: {self.model_weights}")
    
    def combine_predictions(self, model_predictions: List[np.ndarray]) -> np.ndarray:
        """
        Combine predictions using learned weights
        For segmentation, we use weighted voting
        """
        if not self.is_trained:
            print("Warning: Combiner not trained, using equal weights")
            
        if len(model_predictions) == 0:
            raise ValueError("No predictions to combine")
            
        # For demonstration, we'll use a simple weighted approach
        # In practice, this would be more sophisticated
        H, W = model_predictions[0].shape
        combined_prediction = np.zeros((H, W))
        
        # Weighted combination (simplified for categorical predictions)
        weighted_sum = np.zeros((H, W, self.num_models))
        for i, pred in enumerate(model_predictions):
            weighted_sum[:, :, i] = pred * self.model_weights[i]
            
        # Take weighted average and argmax
        combined_prediction = np.argmax(np.mean(weighted_sum, axis=2), axis=-1)
        
        return combined_prediction.astype(int)

class HierarchicalCascadingSystem:
    """
    Main system implementing the hierarchical cascading architecture
    """
    def __init__(self, num_layers: int = 3, num_folds: int = 5):
        self.num_layers = num_layers
        self.num_folds = num_folds
        self.partition_manager = DatasetPartitioner(num_folds)
        self.enhanced_generator = EnhancedDatasetGenerator()
        self.models_per_layer = {}  # Store models for each layer
        self.combiners_per_layer = {}  # Store combiners for each layer
        
    def initialize_models(self, num_classes: int) -> List[SegmentationModel]:
        """
        Initialize segmentation models for a layer
        """
        models = [
            UNetResNet101(num_classes),
            UNetLinkNet(num_classes)
        ]
        return models
    
    def train_first_layer(self, dataset: MedicalImageDataset) -> List[SegmentationModel]:
        """
        Train first layer models on the original dataset
        """
        print("Training first layer models...")
        
        # Initialize models
        models = self.initialize_models(dataset.num_classes)
        
        # Train each model
        for model in models:
            model.train(dataset.images, dataset.masks)
            
        self.models_per_layer[1] = models
        return models
    
    def generate_second_layer_data(self, dataset: MedicalImageDataset, 
                                 first_layer_models: List[SegmentationModel]) -> List[EnhancedImage]:
        """
        Generate enhanced dataset for second layer training
        """
        print("Generating second layer training data...")
        
        # Create enhanced images
        enhanced_images = self.enhanced_generator.generate_enhanced_dataset(
            dataset, first_layer_models
        )
        
        return enhanced_images
    
    def train_layer_models(self, dataset: MedicalImageDataset, layer_idx: int) -> List[SegmentationModel]:
        """
        Train models for a specific layer using cross-validation
        """
        print(f"Training models for layer {layer_idx}...")
        
        # Create partitions
        partitions = self.partition_manager.create_partitions(dataset)
        
        # Initialize models for this layer
        models = self.initialize_models(dataset.num_classes)
        
        # Train models on each partition
        trained_models = []
        
        for fold_idx, (train_indices, val_indices) in enumerate(partitions):
            print(f"Training fold {fold_idx + 1}/{self.num_folds} for layer {layer_idx}...")
            
            # Get training data
            train_images = dataset.images[train_indices]
            train_masks = dataset.masks[train_indices]
            
            # Train each model on this fold
            fold_models = []
            for model in models:
                # Create a copy of the model for this fold
                fold_model = type(model)(model.num_classes)
                fold_model.train(train_images, train_masks)
                fold_models.append(fold_model)
                
            trained_models.extend(fold_models)
            
        self.models_per_layer[layer_idx] = trained_models
        return trained_models
    
    def optimize_combiner_weights(self, dataset: MedicalImageDataset, 
                                models: List[SegmentationModel], layer_idx: int) -> WeightBasedCombiner:
        """
        Optimize combiner weights for a layer
        """
        print(f"Optimizing combiner weights for layer {layer_idx}...")
        
        # Create partitions for validation
        partitions = self.partition_manager.create_partitions(dataset)
        
        # Initialize combiner
        combiner = WeightBasedCombiner(len(models), dataset.num_classes)
        
        # Collect predictions and ground truths for weight optimization
        all_predictions = []
        all_ground_truths = []
        
        prediction_generator = PredictionMatrixGenerator()
        
        for fold_idx, (train_indices, val_indices) in enumerate(partitions):
            print(f"Validating fold {fold_idx + 1}/{self.num_folds}...")
            
            # Get validation data
            val_images = dataset.images[val_indices]
            val_masks = dataset.masks[val_indices]
            
            # Generate predictions for each validation image
            fold_predictions = []
            fold_ground_truths = []
            
            for i in range(len(val_images)):
                # Get predictions from all models
                image_predictions = []
                for model in models:
                    if model.is_trained:
                        batch_image = np.expand_dims(val_images[i], axis=0)
                        pred = model.predict(batch_image)[0]  # Remove batch dimension
                        image_predictions.append(pred)
                
                if image_predictions:
                    fold_predictions.append(image_predictions)
                    fold_ground_truths.append(val_masks[i])
            
            all_predictions.extend(fold_predictions)
            all_ground_truths.extend(fold_ground_truths)
        
        # Optimize combiner weights
        combiner.optimize_weights(all_predictions, all_ground_truths)
        
        self.combiners_per_layer[layer_idx] = combiner
        return combiner
    
    def train_hierarchical_system(self, dataset: MedicalImageDataset) -> Dict[str, Any]:
        """
        Train the complete hierarchical cascading system
        """
        print("="*60)
        print("TRAINING HIERARCHICAL CASCADE SYSTEM")
        print("="*60)
        
        results = {}
        
        # Layer 1: Train on original dataset
        print("\n--- LAYER 1 TRAINING ---")
        layer1_models = self.train_first_layer(dataset)
        results['layer1_models'] = layer1_models
        
        # Optimize combiner for layer 1
        layer1_combiner = self.optimize_combiner_weights(dataset, layer1_models, 1)
        results['layer1_combiner'] = layer1_combiner
        
        # Generate enhanced data for layer 2
        print("\n--- GENERATING ENHANCED DATA FOR LAYER 2 ---")
        enhanced_images = self.generate_second_layer_data(dataset, layer1_models)
        results['enhanced_images'] = enhanced_images
        
        # Convert enhanced images to dataset format for subsequent layers
        # This is a simplified conversion - in practice, you'd need proper handling
        enhanced_dataset = self._convert_enhanced_to_dataset(enhanced_images, dataset)
        
        # Train subsequent layers
        for layer_idx in range(2, self.num_layers + 1):
            print(f"\n--- LAYER {layer_idx} TRAINING ---")
            
            # Train models for this layer
            layer_models = self.train_layer_models(enhanced_dataset, layer_idx)
            results[f'layer{layer_idx}_models'] = layer_models
            
            # Optimize combiner for this layer
            layer_combiner = self.optimize_combiner_weights(enhanced_dataset, layer_models, layer_idx)
            results[f'layer{layer_idx}_combiner'] = layer_combiner
            
            # For demonstration, we'll stop here
            # In practice, you'd continue generating enhanced data for next layer
            
        print("\n" + "="*60)
        print("HIERARCHICAL CASCADE SYSTEM TRAINING COMPLETED")
        print("="*60)
        
        return results
    
    def _convert_enhanced_to_dataset(self, enhanced_images: List[EnhancedImage], 
                                   original_dataset: MedicalImageDataset) -> MedicalImageDataset:
        """
        Convert enhanced images to dataset format for training subsequent layers
        """
        # This is a simplified conversion for demonstration
        # In practice, you'd need to handle the enhanced channel structure properly
        
        # For now, we'll just use the original images and masks
        return original_dataset

class MatrixAnalyzer:
    """
    Analyzes prediction matrices and handles computational challenges
    """
    def __init__(self):
        pass
    
    def analyze_matrix_size(self, dataset_info: Dict[str, Any]) -> Dict[str, int]:
        """
        Analyze the size of prediction matrices for computational planning
        """
        N = dataset_info['num_images']
        H = dataset_info['height']
        W = dataset_info['width']
        S = dataset_info['num_models']
        L = dataset_info['num_classes']
        
        # Calculate matrix dimensions
        total_pixels = N * H * W
        matrix_rows = total_pixels
        matrix_cols = S * L
        
        total_elements = matrix_rows * matrix_cols
        
        return {
            'matrix_rows': matrix_rows,
            'matrix_cols': matrix_cols,
            'total_elements': total_elements,
            'memory_estimate_gb': total_elements * 4 / (1024**3)  # Assuming 4 bytes per float
        }
    
    def demonstrate_computational_scaling(self):
        """
        Demonstrate computational scaling with example datasets
        """
        print("Computational Scaling Analysis:")
        print("-" * 40)
        
        # Example: NIH chest X-ray dataset
        nih_info = {
            'name': 'NIH Chest X-ray',
            'num_images': 80000,
            'height': 640,
            'width': 544,
            'num_models': 2,
            'num_classes': 14
        }
        
        nih_analysis = self.analyze_matrix_size(nih_info)
        print(f"{nih_info['name']}:")
        print(f"  Images: {nih_info['num_images']:,}")
        print(f"  Resolution: {nih_info['height']}x{nih_info['width']}")
        print(f"  Matrix rows: {nih_analysis['matrix_rows']:,}")
        print(f"  Matrix cols: {nih_analysis['matrix_cols']:,}")
        print(f"  Total elements: {nih_analysis['total_elements']:,}")
        print(f"  Memory estimate: {nih_analysis['memory_estimate_gb']:.2f} GB")
        
        # Example: VinDR-CXR dataset
        vindr_info = {
            'name': 'VinDR-CXR',
            'num_images': 18000,
            'height': 512,
            'width': 512,
            'num_models': 2,
            'num_classes': 28
        }
        
        vindr_analysis = self.analyze_matrix_size(vindr_info)
        print(f"\n{vindr_info['name']}:")
        print(f"  Images: {vindr_info['num_images']:,}")
        print(f"  Resolution: {vindr_info['height']}x{vindr_info['width']}")
        print(f"  Matrix rows: {vindr_analysis['matrix_rows']:,}")
        print(f"  Matrix cols: {vindr_analysis['matrix_cols']:,}")
        print(f"  Total elements: {vindr_analysis['total_elements']:,}")
        print(f"  Memory estimate: {vindr_analysis['memory_estimate_gb']:.2f} GB")
'''
Checking the code....... if work, then apply your customized datasets......
def create_sample_dataset(num_images: int = 100, height: int = 256, width: int = 256, 
                         channels: int = 3, num_classes: int = 5) -> MedicalImageDataset:
    """
    Create a sample medical image dataset for demonstration
    """
    print(f"Creating sample dataset with {num_images} images...")
    
    # Generate random images
    images = np.random.randint(0, 256, (num_images, height, width, channels), dtype=np.uint8)
    
    # Generate random masks
    masks = np.random.randint(0, num_classes, (num_images, height, width), dtype=np.uint8)
    
    # Generate image IDs
    image_ids = [f"img_{i:05d}" for i in range(num_images)]
    
    # Define class labels
    class_labels = [f"class_{i}" for i in range(num_classes)]
    
    dataset = MedicalImageDataset(images, masks, image_ids, class_labels)
    print(f"Sample dataset created: {dataset.num_images} images, "
          f"shape {dataset.image_shape}, {dataset.num_classes} classes")
    
    return dataset
'''
def main():
    """
    Demonstrate the hierarchical cascading system
    """
    print("HIERARCHICAL CASCADING MEDICAL IMAGE SEGMENTATION SYSTEM")
    print("="*60)
    
    # Create sample dataset
    dataset = create_sample_dataset(num_images=50, height=128, width=128, num_classes=3)
    
    # Initialize hierarchical system
    hierarchical_system = HierarchicalCascadingSystem(num_layers=2, num_folds=3)
    
    # Train the system
    results = hierarchical_system.train_hierarchical_system(dataset)
    
    # Analyze computational requirements
    print("\n" + "="*60)
    print("COMPUTATIONAL ANALYSIS")
    print("="*60)
    
    matrix_analyzer = MatrixAnalyzer()
    matrix_analyzer.demonstrate_computational_scaling()
    
    # Demonstrate enhanced image creation
    print("\n" + "="*60)
    print("ENHANCED IMAGE DEMONSTRATION")
    print("="*60)
    
    # Create sample models
    sample_models = [
        UNetResNet101(dataset.num_classes),
        UNetLinkNet(dataset.num_classes)
    ]
    
    # Train sample models
    for model in sample_models:
        sample_images = dataset.images[:10]  # Small subset for demo
        sample_masks = dataset.masks[:10]
        model.train(sample_images, sample_masks)
    
    # Create enhanced image
    sample_image = dataset.images[0]
    enhanced_generator = EnhancedDatasetGenerator()
    enhanced_img = enhanced_generator.create_enhanced_image(sample_image, sample_models)
    
    print(f"Original image shape: {sample_image.shape}")
    print(f"Enhanced image channels: {enhanced_img.enhanced_channel_count}")
    print(f"Number of prediction matrices: {len(enhanced_img.prediction_matrices)}")
    
    for i, pred_matrix in enumerate(enhanced_img.prediction_matrices):
        print(f"  Prediction matrix {i+1}: {pred_matrix}")
    
    # Demonstrate probability prediction matrix
    print("\n" + "="*60)
    print("PROBABILITY PREDICTION MATRIX DEMONSTRATION")
    print("="*60)
    
    if enhanced_img.prediction_matrices:
        pred_matrix = enhanced_img.prediction_matrices[0]
        print(f"Matrix dimensions: {pred_matrix.H}x{pred_matrix.W}x{pred_matrix.num_classes}")
        
        # Show sample probabilities
        sample_positions = [(0, 0), (10, 10), (20, 20)]
        for i, j in sample_positions:
            if i < pred_matrix.H and j < pred_matrix.W:
                prob = pred_matrix.get_pixel_probability(i, j, 0)
                print(f"  P(class_0 | pixel({i},{j})) = {prob:.4f}")
    
    print("\n" + "="*60)
    print("SYSTEM DEMONSTRATION COMPLETED")
    print("="*60)
    print("Key Features Demonstrated:")
    print("1. Hierarchical cascading architecture with multiple layers")
    print("2. Dataset partitioning for cross-validation")
    print("3. Enhanced image generation with prediction matrices")
    print("4. Weight-based model combination")
    print("5. Computational scaling analysis")
    print("6. Probability prediction matrix handling")

if __name__ == "__main__":
    main()
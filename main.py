import os
import cv2
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

class MedicalImagePreprocessor:
    def __init__(self, base_path='/home/phd/datasets'):
        self.base_path = base_path
        self.chest_xray_path = os.path.join(base_path, 'chest_xray')
        self.vindr_cxr_path = os.path.join(base_path, 'vindr_cxr')
        self.target_size = (512, 512)
        
    def load_dicom_image(self, dicom_path: str) -> Optional[np.ndarray]:
        """
        Load DICOM image and convert to PNG format
        Handles photometric interpretation for inverted pixel values
        """
        try:
            # Read DICOM file
            dicom = pydicom.dcmread(dicom_path)
            
            # Get pixel array
            image = dicom.pixel_array
            
            # Handle photometric interpretation
            if hasattr(dicom, 'PhotometricInterpretation'):
                if dicom.PhotometricInterpretation == "MONOCHROME1":
                    # Invert pixel values for MONOCHROME1
                    image = np.max(image) - image
            
            return image
        except Exception as e:
            print(f"Error loading DICOM image {dicom_path}: {str(e)}")
            return None
    
    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
        """
        Resize image to target size using bilinear interpolation
        """
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        return resized
    
    def normalize_image_johnson(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image using Johnson et al. approach:
        1. Subtract minimum pixel value
        2. Divide by maximum pixel value in shifted image
        3. Scale to 0-255 range and convert to uint8
        """
        # Step 1: Subtract minimum pixel value
        min_val = np.min(image)
        shifted_image = image.astype(np.float32) - min_val
        
        # Step 2: Divide by maximum pixel value in shifted image
        max_val = np.max(shifted_image)
        if max_val > 0:
            normalized_image = shifted_image / max_val
        else:
            normalized_image = shifted_image
        
        # Step 3: Scale to 0-255 and convert to uint8
        final_image = (normalized_image * 255).astype(np.uint8)
        
        return final_image
    
    def apply_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """
        Apply histogram equalization to enhance contrast
        """
        if len(image.shape) == 3:
            # Convert to grayscale if RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply histogram equalization
        equalized = cv2.equalizeHist(image)
        return equalized
    
    def preprocess_chest_xray_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Complete preprocessing pipeline for ChestX-ray PNG images
        """
        try:
            # Load PNG image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Could not load image: {image_path}")
                return None
            
            # Resize to 512x512
            resized = self.resize_image(image, self.target_size)
            
            # Normalize using Johnson method
            normalized = self.normalize_image_johnson(resized)
            
            # Apply histogram equalization
            enhanced = self.apply_histogram_equalization(normalized)
            
            return enhanced
            
        except Exception as e:
            print(f"Error preprocessing ChestX-ray image {image_path}: {str(e)}")
            return None
    
    def preprocess_vindr_cxr_image(self, dicom_path: str) -> Optional[np.ndarray]:
        """
        Complete preprocessing pipeline for VinDr-CXR DICOM images
        """
        try:
            # Load DICOM image
            image = self.load_dicom_image(dicom_path)
            if image is None:
                return None
            
            # Resize to 512x512
            resized = self.resize_image(image, self.target_size)
            
            # Normalize using Johnson method
            normalized = self.normalize_image_johnson(resized)
            
            # Apply histogram equalization
            enhanced = self.apply_histogram_equalization(normalized)
            
            return enhanced
            
        except Exception as e:
            print(f"Error preprocessing VinDr-CXR image {dicom_path}: {str(e)}")
            return None
    
    def process_dataset_batch(self, image_paths: List[str], dataset_type: str = 'chest_xray') -> List[np.ndarray]:
        """
        Process a batch of images from either dataset
        """
        processed_images = []
        
        print(f"Processing batch of {len(image_paths)} {dataset_type} images...")
        
        for i, image_path in enumerate(image_paths):
            if dataset_type == 'chest_xray':
                processed = self.preprocess_chest_xray_image(image_path)
            elif dataset_type == 'vindr_cxr':
                processed = self.preprocess_vindr_cxr_image(image_path)
            else:
                print(f"Unknown dataset type: {dataset_type}")
                continue
            
            if processed is not None:
                processed_images.append(processed)
            
            # Progress indicator
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(image_paths)} images")
        
        print(f"Successfully processed {len(processed_images)} out of {len(image_paths)} images")
        return processed_images
    
    def demonstrate_preprocessing_pipeline(self, sample_chest_path: str = None, sample_dicom_path: str = None):
        """
        Demonstrate the complete preprocessing pipeline
        """
        print("="*60)
        print("MEDICAL IMAGE PREPROCESSING PIPELINE DEMONSTRATION")
        print("="*60)
        
        # Create sample images if paths not provided
        if sample_chest_path is None or not os.path.exists(sample_chest_path):
            print("Creating sample ChestX-ray image for demonstration...")
            sample_chest = np.random.randint(0, 255, (1024, 1024), dtype=np.uint8)
            cv2.circle(sample_chest, (512, 512), 100, 128, -1)
            cv2.rectangle(sample_chest, (200, 200), (824, 824), 200, 5)
            sample_chest_img = self.resize_image(sample_chest, self.target_size)
        else:
            sample_chest_img = cv2.imread(sample_chest_path, cv2.IMREAD_GRAYSCALE)
            if sample_chest_img is not None:
                sample_chest_img = self.resize_image(sample_chest_img, self.target_size)
            else:
                sample_chest_img = np.random.randint(0, 255, self.target_size, dtype=np.uint8)
        
        # Apply preprocessing steps
        print("\n1. Original Image")
        print(f"   Shape: {sample_chest_img.shape}")
        print(f"   Min pixel value: {np.min(sample_chest_img)}")
        print(f"   Max pixel value: {np.max(sample_chest_img)}")
        
        # Johnson normalization
        normalized = self.normalize_image_johnson(sample_chest_img)
        print("\n2. After Johnson Normalization")
        print(f"   Shape: {normalized.shape}")
        print(f"   Min pixel value: {np.min(normalized)}")
        print(f"   Max pixel value: {np.max(normalized)}")
        
        # Histogram equalization
        enhanced = self.apply_histogram_equalization(normalized)
        print("\n3. After Histogram Equalization")
        print(f"   Shape: {enhanced.shape}")
        print(f"   Min pixel value: {np.min(enhanced)}")
        print(f"   Max pixel value: {np.max(enhanced)}")
        
        # Visualize preprocessing steps
        try:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(sample_chest_img, cmap='gray')
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            axes[1].imshow(normalized, cmap='gray')
            axes[1].set_title('Johnson Normalized')
            axes[1].axis('off')
            
            axes[2].imshow(enhanced, cmap='gray')
            axes[2].set_title('Histogram Equalized')
            axes[2].axis('off')
            
            plt.suptitle('Medical Image Preprocessing Pipeline')
            plt.tight_layout()
            plt.show()
        except:
            print("Matplotlib not available for visualization")
        
        print("\nPreprocessing pipeline demonstration completed successfully!")
        return sample_chest_img, normalized, enhanced

class HierarchicalCascadedProcessor:
    """
    Implementation of the Hierarchical Cascaded Vision-Transformers algorithm
    """
    def __init__(self, num_stages: int = 3):
        self.num_stages = num_stages
        self.weight_matrices = {}
        self.transformer_models = {}
        
    def augment_data(self, X_train: np.ndarray, predictions: List[np.ndarray]) -> np.ndarray:
        """
        Stage 1: Data Augmentation
        Augment image: X*_i ← X_i ∪ ⋃_{s=1}^{S} O^s_y(X_i)
        """
        print("Stage 1: Data Augmentation")
        X_augmented = []
        
        for i in range(len(X_train)):
            # Start with original image
            augmented = X_train[i].copy()
            
            # Concatenate predictions from all stages
            for s in range(self.num_stages):
                if s < len(predictions):
                    augmented = np.concatenate([augmented, predictions[s][i]], axis=-1)
            
            X_augmented.append(augmented)
        
        return np.array(X_augmented)
    
    def optimize_weights(self, A_star: np.ndarray, Y_s: np.ndarray, stage: int) -> np.ndarray:
        """
        Stage 2: Weight Optimization per Stage
        Learn weights: W_s ← argmin_{W_s} ||A*_s W_s - Y_s||^2
        """
        print(f"Stage 2: Optimizing weights for stage {stage}")
        try:
            # Least squares optimization
            W_s = np.linalg.lstsq(A_star, Y_s, rcond=None)[0]
            return W_s
        except Exception as e:
            print(f"Error in weight optimization for stage {stage}: {str(e)}")
            # Return identity matrix as fallback
            return np.eye(A_star.shape[1])[:A_star.shape[1], :Y_s.shape[1]]
    
    def combine_predictions(self, I_test: np.ndarray, predictions: List[np.ndarray], weights: List[np.ndarray]) -> np.ndarray:
        """
        Stage 3: Prediction Fusion
        Combine predictions: C^s(I_test(i,j)) ← ∑_{s=1}^{S} w_{s,j} · O^s_y(I*_test(i,j))
        """
        print("Stage 3: Prediction Fusion")
        combined_results = []
        
        for i in range(len(I_test)):
            combined = np.zeros_like(predictions[0][i]) if len(predictions) > 0 else np.zeros(I_test[i].shape)
            
            for s in range(min(self.num_stages, len(predictions), len(weights))):
                if s < len(predictions) and s < len(weights):
                    # Weighted sum of predictions
                    weighted_pred = weights[s] * predictions[s][i]
                    combined += weighted_pred
            
            combined_results.append(combined)
        
        return np.array(combined_results)
    
    def analyze(self, X_train: np.ndarray, X_test: np.ndarray, Y_train: np.ndarray, 
                S: int = 3, N: int = 1000) -> np.ndarray:
        """
        Main analysis function implementing the Hierarchical Cascaded Vision-Transformers algorithm
        """
        print("Starting Hierarchical Cascaded Vision-Transformers Analysis")
        print(f"Parameters: S={S} stages, N={N} samples")
        
        # Initialize synthetic predictions for demonstration
        # In real implementation, these would come from actual Vision Transformer models
        print("Initializing synthetic predictions for demonstration...")
        synthetic_predictions_train = []
        synthetic_predictions_test = []
        
        for s in range(S):
            # Generate synthetic predictions with same shape as input but with additional channels
            pred_shape_train = list(X_train.shape)
            pred_shape_train[-1] = X_train.shape[-1] + (s + 1) * 10  # Add channels
            pred_train = np.random.rand(*pred_shape_train[:2])  # Simplified for demo
            synthetic_predictions_train.append(pred_train)
            
            pred_shape_test = list(X_test.shape)
            pred_shape_test[-1] = X_test.shape[-1] + (s + 1) * 10
            pred_test = np.random.rand(*pred_shape_test[:2])  # Simplified for demo
            synthetic_predictions_test.append(pred_test)
        
        # Stage 1: Data Augmentation
        X_augmented = self.augment_data(X_train, synthetic_predictions_train)
        print(f"Augmented training data shape: {X_augmented.shape}")
        
        # Stage 2: Weight Optimization per Stage
        optimized_weights = []
        for s in range(S):
            # Create synthetic A*_s matrix for demonstration
            A_star_s = np.random.rand(X_augmented.shape[0], max(1, X_augmented.shape[-1] // 2))
            Y_s = Y_train if len(Y_train.shape) > 1 else Y_train.reshape(-1, 1)
            
            W_s = self.optimize_weights(A_star_s, Y_s, s)
            optimized_weights.append(W_s)
            print(f"Optimized weight matrix for stage {s} shape: {W_s.shape}")
        
        # Stage 3: Prediction Fusion
        final_results = self.combine_predictions(X_test, synthetic_predictions_test, optimized_weights)
        print(f"Final analysis results shape: {final_results.shape}")
        
        print("Hierarchical Cascaded Vision-Transformers Analysis completed!")
        return final_results

def main():
    """Main function to demonstrate preprocessing and hierarchical processing"""
    
    # Initialize preprocessor
    preprocessor = MedicalImagePreprocessor()
    
    # Check dataset paths
    print("Checking dataset paths...")
    preprocessor_paths = [
        preprocessor.chest_xray_path,
        preprocessor.vindr_cxr_path
    ]
    
    for path in preprocessor_paths:
        if os.path.exists(path):
            print(f"✓ Found: {path}")
        else:
            print(f"✗ Not found: {path}")
            print(f"  Please ensure datasets are located at: {path}")
    
    # Demonstrate preprocessing pipeline
    preprocessor.demonstrate_preprocessing_pipeline()
    
    # Demonstrate Hierarchical Cascaded Vision-Transformers algorithm
    print("\n" + "="*60)
    print("DEMONSTRATING HIERARCHICAL CASCADED VISION-TRANSFORMERS")
    print("="*60)
    
    # Initialize hierarchical processor
    hc_processor = HierarchicalCascadedProcessor(num_stages=3)
    
    # Create sample data for demonstration
    #print("Creating sample data for demonstration...")
    #X_train = np.random.rand(100, 512, 512)  # 100 training images of 512x512
    #X_test = np.random.rand(20, 512, 512)    # 20 test images of 512x512
    #Y_train = np.random.randint(0, 2, (100, 14))  # Binary labels for 14 diseases
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Training labels shape: {Y_train.shape}")
    
    # Run analysis
    results = hc_processor.analyze(X_train, X_test, Y_train, S=3, N=100)
    
    print(f"\nAnalysis completed successfully!")
    print(f"Results shape: {results.shape}")
    
    print("\n" + "="*60)
    print("PREPROCESSING AND ANALYSIS DEMONSTRATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
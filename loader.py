import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class MedicalImageDataset:
    def __init__(self, base_path='/home/phd/datasets'):
        self.base_path = base_path
        self.chest_xray_path = os.path.join(base_path, 'chest_xray')
        self.vindr_cxr_path = os.path.join(base_path, 'vindr_cxr')
        
        # Define disease labels for both datasets
        self.chest_xray_labels = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 
            'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 
            'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia', 'No Finding'
        ]
        
        self.vindr_cxr_labels = [
            'Aortic_enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly', 
            'Clavicle_fracture', 'Consolidation', 'Edema', 'Emphysema', 
            'Enlarged_pulmonary_artery', 'Interstitial_lung_disease', 'Infiltration', 
            'Lung_opacity', 'Lung_cavity', 'Lung_cyst', 'Mediastinal_shift', 
            'Nodule/Mass', 'Pleural_effusion', 'Pleural_thickening', 'Pneumothorax', 
            'Pulmonary_fibrosis', 'Rib_fracture', 'Other_lesion', 
            'Chronic_obstructive_pulmonary_disease', 'Lung_tumor', 'Pneumonia', 
            'Tuberculosis', 'Other_disease', 'No finding'
        ]
    
    def load_chest_xray_data(self):
        """
        Load ChestX-ray dataset information
        Dataset composition: 112,120 images from 30,805 patients
        Training: 86,524 images
        Test: 25,596 images
        """
        print("Loading ChestX-ray dataset...")
        print(f"Dataset path: {self.chest_xray_path}")
        
        # Dataset statistics
        total_images = 112120
        total_patients = 30805
        train_images = 86524
        test_images = 25596
        
        print(f"Total images: {total_images}")
        print(f"Total patients: {total_patients}")
        print(f"Training images: {train_images}")
        print(f"Test images: {test_images}")
        print(f"Number of disease labels: {len(self.chest_xray_labels)}")
        print("Disease labels:", self.chest_xray_labels)
        
        return {
            'total_images': total_images,
            'total_patients': total_patients,
            'train_images': train_images,
            'test_images': test_images,
            'labels': self.chest_xray_labels
        }
    
    def load_vindr_cxr_data(self):
        """
        Load VinDr-CXR dataset information
        Dataset composition: 18,000 images classified by radiologists
        Training: 15,000 images
        Test: 3,000 images
        """
        print("\nLoading VinDr-CXR dataset...")
        print(f"Dataset path: {self.vindr_cxr_path}")
        
        # Dataset statistics
        total_images = 18000
        train_images = 15000
        test_images = 3000
        
        print(f"Total images: {total_images}")
        print(f"Training images: {train_images}")
        print(f"Test images: {test_images}")
        print(f"Number of disease labels: {len(self.vindr_cxr_labels)}")
        print("Disease labels:", self.vindr_cxr_labels)
        
        return {
            'total_images': total_images,
            'train_images': train_images,
            'test_images': test_images,
            'labels': self.vindr_cxr_labels
        }
    
    def display_dataset_info(self):
        """Display comprehensive information about both datasets"""
        print("="*60)
        print("MEDICAL CHEST X-RAY DATASET ANALYSIS")
        print("="*60)
        
        # Load ChestX-ray dataset info
        chest_data = self.load_chest_xray_data()
        
        # Load VinDr-CXR dataset info
        vindr_data = self.load_vindr_cxr_data()
        
        print("\n" + "="*60)
        print("DATASET COMPARISON")
        print("="*60)
        print(f"{'Metric':<25} {'ChestX-ray':<15} {'VinDr-CXR':<15}")
        print("-"*60)
        print(f"{'Total Images':<25} {chest_data['total_images']:<15} {vindr_data['total_images']:<15}")
        print(f"{'Training Images':<25} {chest_data['train_images']:<15} {vindr_data['train_images']:<15}")
        print(f"{'Test Images':<25} {chest_data['test_images']:<15} {vindr_data['test_images']:<15}")
        print(f"{'Number of Labels':<25} {len(chest_data['labels']):<15} {len(vindr_data['labels']):<15}")
        print(f"{'Annotated by':<25} {'NLP':<15} {'Radiologists':<15}")
        
        return chest_data, vindr_data
    
    def check_dataset_paths(self):
        """Check if dataset directories exist"""
        print("\nChecking dataset paths...")
        
        paths_to_check = [
            self.chest_xray_path,
            self.vindr_cxr_path
        ]
        
        for path in paths_to_check:
            if os.path.exists(path):
                print(f"✓ Found: {path}")
                # Count files in directory
                try:
                    file_count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
                    print(f"  Files in directory: {file_count}")
                except:
                    print(f"  Unable to count files in {path}")
            else:
                print(f"✗ Not found: {path}")
                print(f"  Please ensure the dataset is located at: {path}")
    
    def preprocess_image(self, image_path, target_size=(224, 224)):
        """
        Preprocess a single image with histogram equalization
        """
        try:
            # Read image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            
            # Resize image
            img = cv2.resize(img, target_size)
            
            # Apply histogram equalization
            img_eq = cv2.equalizeHist(img)
            
            return img_eq
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None
    
    def demonstrate_preprocessing(self, sample_image_path=None):
        """
        Demonstrate image preprocessing techniques
        """
        print("\nDemonstrating image preprocessing techniques...")
        
        # If no sample image provided, create a sample demonstration
        if sample_image_path is None or not os.path.exists(sample_image_path):
            print("Creating sample image for demonstration...")
            # Create a sample image with varying intensities
            sample_img = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
            # Add some structured patterns
            cv2.circle(sample_img, (112, 112), 50, 128, -1)
            cv2.rectangle(sample_img, (50, 50), (174, 174), 200, 3)
        else:
            sample_img = cv2.imread(sample_image_path, cv2.IMREAD_GRAYSCALE)
            if sample_img is not None:
                sample_img = cv2.resize(sample_img, (224, 224))
            else:
                print("Could not load sample image, creating synthetic one...")
                sample_img = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        
        # Apply histogram equalization
        img_eq = cv2.equalizeHist(sample_img)
        
        # Display comparison (if matplotlib is available)
        try:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(sample_img, cmap='gray')
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            axes[1].imshow(img_eq, cmap='gray')
            axes[1].set_title('Histogram Equalized')
            axes[1].axis('off')
            
            plt.suptitle('Image Preprocessing Demonstration')
            plt.tight_layout()
            plt.show()
        except:
            print("Matplotlib not available for visualization")
        
        print("Preprocessing demonstration completed.")
        return sample_img, img_eq

def main():
    """Main function to demonstrate dataset loading and analysis"""
    
    # Initialize dataset handler
    dataset_handler = MedicalImageDataset()
    
    # Check if dataset paths exist
    dataset_handler.check_dataset_paths()
    
    # Display comprehensive dataset information
    chest_data, vindr_data = dataset_handler.display_dataset_info()
    
    # Demonstrate preprocessing techniques
    dataset_handler.demonstrate_preprocessing()
    
    print("\n" + "="*60)
    print("DATASET ANALYSIS COMPLETE")
    print("="*60)
    print("Next steps:")
    print("1. Ensure datasets are properly organized in the specified paths")
    print("2. Implement data loading functions for actual image processing")
    print("3. Add label loading and preprocessing pipelines")
    print("4. Implement training and evaluation workflows")

if __name__ == "__main__":
    main()
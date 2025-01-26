import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import logging
import sys
import cv2

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_extraction.face_processor import FaceProcessor

class GlobalModelTester:
    def __init__(self, 
                 global_model_dir: str = "data/global_model",
                 test_images_dir: str = "../../data/test_images"):
        self.global_model_dir = global_model_dir
        self.test_images_dir = test_images_dir
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize face processor for test images
        self.face_processor = FaceProcessor()
        
    def load_latest_model(self):
        """Load the most recent global model"""
        try:
            # Get all model files
            model_files = [f for f in os.listdir(self.global_model_dir) 
                         if f.startswith("global_model_")]
            
            if not model_files:
                self.logger.error("No global model files found!")
                return None
            
            # Get the most recent file
            latest_model = max(model_files)
            model_path = os.path.join(self.global_model_dir, latest_model)
            
            # Load the model
            model_data = np.load(model_path, allow_pickle=True)
            self.logger.info(f"Loaded model: {latest_model}")
            
            return model_data['student_embeddings'].item()
    
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return None

    def save_result_image(self, image_path: str, similarity: float, face):
        """Save the test image with similarity score and face box"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                self.logger.error(f"Could not load image for saving: {image_path}")
                return
            
            if face is not None:
                # Draw face bounding box
                x1, y1, x2, y2 = [int(b) for b in face.bbox]
                
                # Set color based on threshold (0.5)
                threshold = 0.5
                is_match = similarity > threshold
                color = (0, 255, 0) if is_match else (0, 0, 255)  # Green for match, Red for no match
                
                # Draw rectangle around face
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # Add text with similarity and threshold
                text_lines = [
                    f"Similarity: {similarity:.2f}",
                    f"Threshold: {threshold:.2f}",
                    "MATCH" if is_match else "NO MATCH"
                ]
                
                # Position text
                y_position = 30
                for text in text_lines:
                    cv2.putText(img, text, (10, y_position), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    y_position += 30
            
            # Create results directory if it doesn't exist
            results_dir = "../../data/test_results"
            os.makedirs(results_dir, exist_ok=True)
            
            # Save image with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(results_dir, 
                                     f"result_{timestamp}_{os.path.basename(image_path)}")
            cv2.imwrite(output_path, img)
            self.logger.info(f"Result saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving result image: {str(e)}")

    def process_test_image(self, image_path):
        """Process a single test image and return embedding and face"""
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                self.logger.error(f"Could not load image: {image_path}")
                return None, None
            
            # Get face embedding
            faces = self.face_processor.app.get(img)
            
            if not faces:
                self.logger.error("No face detected in test image")
                return None, None
            
            # Get the largest face
            face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            test_embedding = face.embedding
            
            return test_embedding, face
            
        except Exception as e:
            self.logger.error(f"Error processing test image {image_path}: {str(e)}")
            return None, None

    def compare_embeddings(self, test_embedding, model_embedding):
        """Compare test embedding with model embedding"""
        try:
            # Normalize embeddings
            test_norm = np.linalg.norm(test_embedding)
            model_norm = np.linalg.norm(model_embedding)
            
            test_normalized = test_embedding / test_norm
            model_normalized = model_embedding / model_norm
            
            # Calculate cosine similarity
            similarity = np.dot(test_normalized, model_normalized)
            
            return {
                'similarity': similarity,
                'test_norm': test_norm,
                'model_norm': model_norm,
                'test_embedding': test_embedding,
                'model_embedding': model_embedding
            }
            
        except Exception as e:
            self.logger.error(f"Error comparing embeddings: {str(e)}")
            return None

    def visualize_results(self, results, output_dir="test_results"):
        """Visualize test results"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create similarity plot
        plt.figure(figsize=(12, 6))
        images = list(results.keys())
        similarities = [results[img]['similarity'] for img in images]
        
        plt.bar(range(len(images)), similarities)
        plt.xticks(range(len(images)), [os.path.basename(img) for img in images], rotation=45)
        plt.title('Similarity Scores for Test Images')
        plt.ylabel('Cosine Similarity')
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, f'test_results_{timestamp}.png')
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path

def main():
    """Test the global model with test images"""
    print("\nGlobal Model Testing")
    print("===================")
    
    tester = GlobalModelTester()
    
    # Load latest model
    model_data = tester.load_latest_model()
    if model_data is None:
        print("Failed to load model!")
        return
    
    print(f"\nModel Info:")
    # The model_data is a dictionary of student embeddings
    print(f"Number of students: {len(model_data)}")
    
    # Get the first student's embedding (since we only have one in this case)
    student_id = list(model_data.keys())[0]
    student_embedding = model_data[student_id]
    print(f"Using student ID: {student_id}")
    
    # Process all test images
    test_results = {}
    test_files = [f for f in os.listdir(tester.test_images_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"\nProcessing {len(test_files)} test images...")
    
    for test_file in test_files:
        image_path = os.path.join(tester.test_images_dir, test_file)
        print(f"\nTesting image: {test_file}")
        
        # Get embedding and face for test image
        test_embedding, face = tester.process_test_image(image_path)
        if test_embedding is None:
            print(f"Failed to process {test_file}")
            continue
        
        # Compare with model
        comparison = tester.compare_embeddings(test_embedding, student_embedding)
        if comparison is None:
            print(f"Failed to compare embeddings for {test_file}")
            continue
        
        test_results[image_path] = comparison
        similarity = comparison['similarity']
        print(f"Similarity score: {similarity:.4f}")
        
        # Save result image with face box and similarity score
        tester.save_result_image(image_path, similarity, face)
    
    if test_results:
        # Visualize results
        plot_file = tester.visualize_results(test_results)
        print(f"\nResults visualization saved as: {plot_file}")
        
        # Print summary
        print("\nTest Summary:")
        print("-------------")
        for image_path, result in test_results.items():
            print(f"{os.path.basename(image_path)}: {result['similarity']:.4f}")
    else:
        print("\nNo test results generated!")

if __name__ == "__main__":
    main() 
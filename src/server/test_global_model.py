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
                 test_images_dir: str = "../../data/test-images"):
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

    def save_result_image(self, image_path: str, faces, img, match_results: list):
        """Save test image with face boxes and match information"""
        try:
            # Draw results on image
            for face, matches in zip(faces, match_results):
                if matches is None:
                    continue
                
                # Get face box
                x1, y1, x2, y2 = [int(b) for b in face.bbox]
                
                # Find best match
                best_match = max(matches.items(), key=lambda x: x[1])
                student_id, similarity = best_match
                
                # Set color based on similarity threshold
                threshold = 0.5
                is_match = similarity > threshold
                color = (0, 255, 0) if is_match else (0, 0, 255)
                
                # Draw face box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # Add match information
                if is_match:
                    text = f"Student {student_id}: {similarity:.2f}"
                else:
                    text = f"No match: {similarity:.2f}"
                
                cv2.putText(img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, color, 2)
            
            # Save result
            results_dir = os.path.join(os.path.dirname(self.test_images_dir), "test_results")
            os.makedirs(results_dir, exist_ok=True)
            
            output_path = os.path.join(results_dir, f"result_{os.path.basename(image_path)}")
            cv2.imwrite(output_path, img)
            self.logger.info(f"Saved result to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving result image: {str(e)}")

    def process_test_image(self, image_path):
        """Process a single test image and return embeddings and faces"""
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                self.logger.error(f"Could not load image: {image_path}")
                return None, None
            
            # Get faces and embeddings
            faces = self.face_processor.app.get(img)
            
            if not faces:
                self.logger.error("No faces detected in test image")
                return None, None
            
            # Return all faces and their embeddings
            return faces, img
            
        except Exception as e:
            self.logger.error(f"Error processing test image {image_path}: {str(e)}")
            return None, None

    def compare_embeddings(self, test_embedding: np.ndarray, student_embeddings: dict) -> dict:
        """Compare test embedding with all student embeddings"""
        results = {}
        try:
            # Normalize test embedding
            test_embedding = test_embedding / np.linalg.norm(test_embedding)
            
            # Compare with each student embedding
            for student_id, student_embedding in student_embeddings.items():
                # Normalize student embedding
                student_embedding = np.array(student_embedding)
                student_embedding = student_embedding / np.linalg.norm(student_embedding)
                
                # Calculate similarity
                similarity = np.dot(test_embedding, student_embedding)
                results[student_id] = similarity
            
            return results
            
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
    print(f"Number of students in model: {len(model_data)}")
    
    # Process all test images
    test_files = [f for f in os.listdir(tester.test_images_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"\nProcessing {len(test_files)} test images...")
    
    for test_file in test_files:
        image_path = os.path.join(tester.test_images_dir, test_file)
        print(f"\nTesting image: {test_file}")
        
        # Get faces and embeddings from test image
        faces, img = tester.process_test_image(image_path)
        if faces is None or img is None:
            print(f"Failed to process {test_file}")
            continue
        
        print(f"Found {len(faces)} faces in image")
        
        # Compare each face with all student embeddings
        match_results = []
        for face in faces:
            # Compare with all student embeddings
            matches = tester.compare_embeddings(face.embedding, model_data)
            if matches is None:
                print(f"Failed to compare embeddings for a face")
                match_results.append(None)
                continue
            
            # Find best match
            best_match = max(matches.items(), key=lambda x: x[1])
            student_id, similarity = best_match
            
            if similarity > 0.5:
                print(f"Face matched with Student {student_id} (similarity: {similarity:.2f})")
            else:
                print(f"Face has no good match (best similarity: {similarity:.2f})")
            
            match_results.append(matches)
        
        # Save result image with all faces and their matches
        tester.save_result_image(image_path, faces, img, match_results)

if __name__ == "__main__":
    main() 
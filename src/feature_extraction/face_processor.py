import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import os
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime

class FaceProcessor:
    def __init__(self):
        """Initialize Face Processor with InsightFace"""
        # Setup logging first
        self.logger = logging.getLogger(__name__)
        
        # Set CUDA paths to system installation
        cuda_path = os.environ.get('CUDA_PATH', 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8')
        os.environ['PATH'] = f"{cuda_path}/bin;{os.environ['PATH']}"
        os.environ['CUDA_PATH'] = cuda_path
        
        try:
            # Try with basic CPU first to ensure model loading
            self.app = FaceAnalysis(
                name="buffalo_l",
                root="models",
                providers=['CPUExecutionProvider']
            )
            self.logger.info("Initial CPU initialization successful")
            
            # Now try to switch to CUDA
            self.app = FaceAnalysis(
                name="buffalo_l",
                root="models",
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.logger.info("Successfully switched to CUDA provider")
        except Exception as e:
            self.logger.warning(f"Using CPU only due to CUDA error: {str(e)}")
            # Keep the CPU initialization if CUDA failed
        
        # Prepare the model with smaller detection size for better performance
        try:
            self.app.prepare(ctx_id=0, det_size=(320, 320))
            self.logger.info("Model preparation completed")
        except Exception as e:
            self.logger.error(f"Failed to prepare model: {str(e)}")
            raise
        
    def process_student_images(self, person_dir: str) -> Optional[np.ndarray]:
        """
        Process images for a single person and generate embedding
        
        Args:
            person_dir: Directory containing images for one person
        Returns:
            Average embedding for the person
        """
        embeddings = []
        
        # Get all valid image files
        valid_extensions = ('.jpg', '.jpeg', '.png')
        image_files = [f for f in os.listdir(person_dir) 
                      if f.lower().endswith(valid_extensions)]
        
        if not image_files:
            self.logger.warning(f"No valid images found in {person_dir}")
            return None
        
        self.logger.info(f"Processing {len(image_files)} images for {os.path.basename(person_dir)}")
        
        for img_file in image_files:
            try:
                # Load and preprocess image
                img_path = os.path.join(person_dir, img_file)
                img = cv2.imread(img_path)
                
                if img is None:
                    self.logger.error(f"Failed to load image: {img_path}")
                    continue
                
                # Resize if image is too large
                max_size = 640  # Reduced max size
                if max(img.shape) > max_size:
                    scale = max_size / max(img.shape)
                    img = cv2.resize(img, None, fx=scale, fy=scale)
                
                # Get face embedding
                faces = self.app.get(img)
                
                if not faces:
                    self.logger.warning(f"No face detected in {img_file}")
                    continue
                
                # Get the largest face
                face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                embeddings.append(face.embedding)
                self.logger.info(f"Processed {img_file}")
                
            except Exception as e:
                self.logger.error(f"Error processing {img_file}: {str(e)}")
                continue
        
        if not embeddings:
            self.logger.error(f"No valid embeddings generated for {os.path.basename(person_dir)}")
            return None
        
        # Calculate average embedding
        average_embedding = np.mean(embeddings, axis=0)
        average_embedding = average_embedding / np.linalg.norm(average_embedding)
        
        return average_embedding
    
    def process_all_persons(self, base_dir: str) -> Dict[str, np.ndarray]:
        """
        Process images for all persons in the dataset
        
        Args:
            base_dir: Base directory containing subdirectories for each person
        Returns:
            Dictionary mapping person names to their embeddings
        """
        person_embeddings = {}
        
        # Get all person directories
        person_dirs = [d for d in os.listdir(base_dir) 
                      if os.path.isdir(os.path.join(base_dir, d))]
        
        if not person_dirs:
            self.logger.error(f"No person directories found in {base_dir}")
            return person_embeddings
        
        total_persons = len(person_dirs)
        self.logger.info(f"Found {total_persons} persons in dataset")
        
        for idx, person_dir in enumerate(person_dirs, 1):
            try:
                person_path = os.path.join(base_dir, person_dir)
                self.logger.info(f"Processing person {idx}/{total_persons}: {person_dir}")
                
                embedding = self.process_student_images(person_path)
                
                if embedding is not None:
                    person_embeddings[person_dir] = embedding
                    # Save individual embedding immediately
                    save_path = os.path.join("data", "embeddings", f"{person_dir}_embedding.npy")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    np.save(save_path, embedding)
                    self.logger.info(f"Saved embedding for {person_dir}")
                
            except Exception as e:
                self.logger.error(f"Failed to process {person_dir}: {str(e)}")
                continue
            
            # Log progress
            self.logger.info(f"Progress: {idx}/{total_persons} persons processed")
        
        return person_embeddings
    
    def process_classroom_photo(self, 
                              image_path: str,
                              min_face_size: int = 50) -> List[np.ndarray]:
        """
        Process classroom photo and extract all face embeddings
        
        Args:
            image_path: Path to classroom photo
            min_face_size: Minimum face size to consider
            
        Returns:
            List of face embeddings
        """
        embeddings = []
        
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                self.logger.error(f"Could not load classroom image: {image_path}")
                return embeddings
            
            # Detect faces
            faces = self.app.get(img)
            
            self.logger.info(f"Detected {len(faces)} faces in classroom photo")
            
            # Process each face
            for face in faces:
                face_width = face.bbox[2] - face.bbox[0]
                if face_width >= min_face_size:
                    embeddings.append(face.embedding)
                else:
                    self.logger.warning("Skipped a face due to small size")
            
        except Exception as e:
            self.logger.error(f"Error processing classroom photo: {str(e)}")
        
        return embeddings

def main():
    """Test the face processor"""
    processor = FaceProcessor()
    
    # Use absolute path to the base directory containing all person folders
    base_dir = os.path.abspath("../../data/students_dataset")
    
    # Debug: Check if directory exists
    if not os.path.exists(base_dir):
        print(f"Error: Directory not found: {base_dir}")
        return
    
    print(f"\nUsing directory: {base_dir}")
    print("Starting feature extraction...")
    
    # Process all persons
    person_embeddings = processor.process_all_persons(base_dir)
    
    if person_embeddings:
        print(f"\nFeature extraction successful for {len(person_embeddings)} persons!")
        
        # Create embeddings directory
        embeddings_dir = os.path.abspath("../../data/embeddings")
        os.makedirs(embeddings_dir, exist_ok=True)
        
        # Save embeddings for each person
        for person_name, embedding in person_embeddings.items():
            embedding_path = os.path.join(embeddings_dir, f"{person_name}_embedding.npy")
            np.save(embedding_path, embedding)
            print(f"Saved embedding for {person_name}")
        
        print("\nAll embeddings saved successfully!")

if __name__ == "__main__":
    main() 
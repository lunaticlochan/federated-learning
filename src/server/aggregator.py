import numpy as np
import os
from typing import List, Dict, Optional
import logging
from datetime import datetime

class FeatureAggregator:
    def __init__(self, model_save_dir: str = "data/federated_model"):
        """Initialize Feature Aggregator"""
        self.model_save_dir = model_save_dir
        os.makedirs(model_save_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Store single student embedding
        self.student_embedding: Optional[np.ndarray] = None
        self.student_id: Optional[str] = None
    
    def add_student_embedding(self, student_id: str, embedding: np.ndarray) -> bool:
        """Add single student embedding"""
        try:
            # Verify embedding shape
            if embedding.shape != (512,):
                self.logger.error(f"Invalid embedding shape from student {student_id}: {embedding.shape}")
                return False
            
            # Store normalized embedding
            self.student_embedding = embedding / np.linalg.norm(embedding)
            self.student_id = student_id
            
            self.logger.info(f"Added embedding for student {student_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding student embedding: {str(e)}")
            return False
    
    def aggregate_embeddings(self, embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """Aggregate multiple embeddings into one model"""
        try:
            # Store all embeddings
            self.student_embeddings = embeddings
            
            # Calculate average embedding
            all_embeddings = np.stack(list(embeddings.values()))
            average_embedding = np.mean(all_embeddings, axis=0)
            
            # Normalize the average embedding
            average_embedding = average_embedding / np.linalg.norm(average_embedding)
            
            self.logger.info(f"Successfully aggregated {len(embeddings)} embeddings")
            return average_embedding
            
        except Exception as e:
            self.logger.error(f"Error aggregating embeddings: {str(e)}")
            return None
    
    def save_model(self) -> bool:
        """Save aggregated model"""
        try:
            if not self.student_embeddings:
                self.logger.error("No embeddings to save")
                return False
            
            # Create a dictionary with embeddings and metadata
            model_data = {
                'student_embeddings': self.student_embeddings,
                'num_students': len(self.student_embeddings),
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
            }
            
            # Save to file
            save_path = os.path.join(
                self.model_save_dir, 
                f"global_model_{model_data['timestamp']}.npz"
            )
            
            np.savez(save_path, **model_data)
            self.logger.info(f"Saved model with {len(self.student_embeddings)} students")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            return False

def main():
    """Test the feature aggregator with single student"""
    aggregator = FeatureAggregator()
    
    print("\nSingle Student Feature Test")
    print("=========================")
    
    # Load the actual student embedding
    feature_path = os.path.join("data", "embeddings", "student_embedding.npy")
    if os.path.exists(feature_path):
        student_embedding = np.load(feature_path)
        print(f"\nLoaded student embedding shape: {student_embedding.shape}")
        
        # Add student embedding
        if aggregator.add_student_embedding("student_01", student_embedding):
            print("Added student embedding successfully")
            
            # Save model
            if aggregator.save_model():
                print("Saved student model successfully")
    else:
        print("\nNo student embedding found. Please run feature extraction first.")

if __name__ == "__main__":
    main() 
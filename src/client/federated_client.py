import socket
import numpy as np
import logging
import os
import sys
import json
from datetime import datetime
from typing import Dict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.protocol import FederatedProtocol, Message, MessageType
from client.feature_manager import FeatureManager

class FederatedClient:
    def __init__(
        self, 
        student_id: str,
        host: str = 'localhost',
        port: int = 5000,
        embeddings_dir: str = "../feature_extraction/data/embeddings"
    ):
        self.student_id = student_id
        self.host = host
        self.port = port
        self.embeddings_dir = os.path.abspath(embeddings_dir)
        
        # Initialize feature manager
        self.feature_manager = FeatureManager(self.embeddings_dir)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def send_message(self, client_socket: socket.socket, message: Message):
        """Send message with length prefix"""
        data = message.to_json().encode()
        message_length = len(data)
        client_socket.send(str(message_length).zfill(10).encode())
        client_socket.send(data)
    
    def receive_message(self, client_socket: socket.socket) -> Message:
        """Receive message with length prefix"""
        message_length = int(client_socket.recv(10).decode())
        message_data = client_socket.recv(message_length).decode()
        return Message.from_json(message_data)
    
    def send_features(self, features: Dict[str, np.ndarray]) -> bool:
        """Connect to server and send features"""
        try:
            # Create socket and connect
            self.logger.info("Connecting to server...")
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((self.host, self.port))
            
            # Send connect message
            connect_msg = FederatedProtocol.create_connect_message(self.student_id)
            self.send_message(client_socket, connect_msg)
            
            # Wait for acknowledgment
            response = self.receive_message(client_socket)
            if not (response.type == MessageType.ACKNOWLEDGE and response.data["success"]):
                self.logger.error("Connection failed")
                return False
            
            self.logger.info("Connected successfully")
            
            # Send features for each student
            self.logger.info(f"Sending {len(features)} student embeddings...")
            feature_msg = FederatedProtocol.create_feature_message(
                self.student_id,
                features  # Send features directly
            )
            self.send_message(client_socket, feature_msg)
            
            # Wait for acknowledgment
            response = self.receive_message(client_socket)
            success = (response.type == MessageType.ACKNOWLEDGE 
                      and response.data["success"])
            
            if success:
                self.logger.info("Features sent successfully")
            else:
                self.logger.error("Failed to send features")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error: {str(e)}")
            return False
            
        finally:
            client_socket.close()
            self.logger.info("Connection closed")

def main():
    """Run the client"""
    print("\nFederated Client")
    print("===============")
    
    try:
        # Initialize client
        client = FederatedClient("client_01")
        
        # Load all embeddings using feature manager
        features = client.feature_manager.get_features()
        
        if features is None or len(features) == 0:
            print("Error: No embeddings found!")
            print(f"Please check the directory: {client.embeddings_dir}")
            return
        
        print(f"\nLoaded {len(features)} embeddings")
        for student_id, embedding in features.items():
            print(f"Student {student_id}: shape {embedding.shape}")
        
        # Send all features to server
        if client.send_features(features):
            print("\nAll embeddings sent successfully!")
        else:
            print("\nFailed to send embeddings!")
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nPlease ensure:")
        print("1. The embeddings directory exists")
        print("2. The directory contains *_embedding.npy files")
        print("3. You have permission to access the directory")

if __name__ == "__main__":
    main() 
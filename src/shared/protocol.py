from dataclasses import dataclass
from enum import Enum
import json
import numpy as np
from typing import Dict, Any

class MessageType(Enum):
    CONNECT = "connect"
    SEND_FEATURES = "send_features"
    ACKNOWLEDGE = "acknowledge"
    ERROR = "error"

@dataclass
class Message:
    type: MessageType
    data: Dict[str, Any]
    
    def to_json(self) -> str:
        """Convert message to JSON string"""
        return json.dumps({
            "type": self.type.value,
            "data": self.data
        })
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Message':
        """Create message from JSON string"""
        data = json.loads(json_str)
        return cls(
            type=MessageType(data["type"]),
            data=data["data"]
        )

class FederatedProtocol:
    @staticmethod
    def create_connect_message(student_id: str) -> Message:
        """Create connection message"""
        return Message(
            type=MessageType.CONNECT,
            data={"student_id": student_id}
        )
    
    @staticmethod
    def create_feature_message(student_id: str, features: Dict[str, Any]) -> Message:
        """Create message for sending features"""
        # Convert numpy arrays to lists if needed
        features_dict = {}
        for sid, embedding in features.items():
            if isinstance(embedding, np.ndarray):
                features_dict[sid] = embedding.tolist()
            else:
                features_dict[sid] = embedding
        
        return Message(
            type=MessageType.SEND_FEATURES,
            data={
                "student_id": student_id,
                "features": features_dict
            }
        )
    
    @staticmethod
    def create_ack_message(success: bool, message: str) -> Message:
        """Create acknowledgment message"""
        return Message(
            type=MessageType.ACKNOWLEDGE,
            data={
                "success": success,
                "message": message
            }
        ) 
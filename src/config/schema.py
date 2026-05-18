from pydantic import BaseModel, Field
from typing import List, Dict, Any

class ModelConfig(BaseModel):
    type: str = Field(..., description="Type of the model (e.g., cnn1d)")
    parameters: Dict[str, Any] = Field(..., description="Parameters for the model")

class TrainingConfig(BaseModel):
    batch_size: int = Field(..., description="Batch size for training")
    learning_rate: float = Field(..., description="Learning rate for the optimizer")
    epochs: int = Field(..., description="Number of epochs for training")

class EvaluationConfig(BaseModel):
    threshold: float = Field(..., description="Threshold for anomaly detection")
    min_duration: int = Field(..., description="Minimum duration for event detection")

class ConfigSchema(BaseModel):
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig

def validate_config(config: Dict[str, Any]) -> ConfigSchema:
    return ConfigSchema(**config)
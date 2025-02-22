import numpy as np
from typing import List, Dict, Any
import torch
import pickle

class FederatedAggregator:
    """
    A federated learning aggregator that combines model weights from multiple edge devices.
    Supports both PyTorch models and numpy arrays.
    """
    
    def __init__(self, aggregation_method: str = "weighted_average"):
        """
        Initialize the aggregator.
        
        Args:
            aggregation_method (str): Method to use for aggregation ('weighted_average' or 'equal_average')
        """
        self.aggregation_method = aggregation_method
        
    def aggregate_weights(self, 
                         model_weights: List[Dict[str, Any]], 
                         sample_sizes: List[int] = None) -> Dict[str, Any]:
        """
        Aggregate weights from multiple models.
        
        Args:
            model_weights (List[Dict]): List of model state dictionaries
            sample_sizes (List[int]): Number of samples used to train each model
            
        Returns:
            Dict: Aggregated model weights
        """
        if len(model_weights) < 2:
            raise ValueError("Need at least two models to aggregate")
            
        if sample_sizes is None:
            # If no sample sizes provided, use equal weighting
            sample_sizes = [1] * len(model_weights)
            
        if len(sample_sizes) != len(model_weights):
            raise ValueError("Number of sample sizes must match number of models")
            
        # Calculate weights for each model based on sample sizes
        total_samples = sum(sample_sizes)
        weights = [size / total_samples for size in sample_sizes]
        
        aggregated_weights = {}
        
        # Get the first model's keys
        for key in model_weights[0].keys():
            # Initialize with zeros of the same shape and type as the first model
            if isinstance(model_weights[0][key], torch.Tensor):
                aggregated_weights[key] = torch.zeros_like(model_weights[0][key])
            else:
                aggregated_weights[key] = np.zeros_like(model_weights[0][key])
            
            # Weighted sum of all models
            for model_idx, model_dict in enumerate(model_weights):
                if isinstance(model_dict[key], torch.Tensor):
                    aggregated_weights[key] += weights[model_idx] * model_dict[key]
                else:
                    aggregated_weights[key] += weights[model_idx] * model_dict[key]
                    
        return aggregated_weights
    
    def save_aggregated_model(self, weights: Dict[str, Any], filepath: str):
        """
        Save aggregated weights to a file.
        
        Args:
            weights (Dict): Aggregated model weights
            filepath (str): Path to save the weights
        """
        with open(filepath, 'wb') as f:
            pickle.dump(weights, f)
            
    def load_aggregated_model(self, filepath: str) -> Dict[str, Any]:
        """
        Load aggregated weights from a file.
        
        Args:
            filepath (str): Path to load the weights from
            
        Returns:
            Dict: Loaded model weights
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)

# Example usage with PyTorch models
def example_usage():
    # Create dummy models and data
    model1_weights = {'layer1': torch.randn(10, 10),
                     'layer2': torch.randn(5, 5)}
    model2_weights = {'layer1': torch.randn(10, 10),
                     'layer2': torch.randn(5, 5)}
    
    # Initialize aggregator
    aggregator = FederatedAggregator()
    
    # Aggregate weights
    aggregated_weights = aggregator.aggregate_weights(
        model_weights=[model1_weights, model2_weights],
        sample_sizes=[100, 150]  # Assuming model1 was trained on 100 samples and model2 on 150
    )
    
    # Save aggregated weights
    aggregator.save_aggregated_model(aggregated_weights, 'aggregated_model.pkl')
    
    # Load aggregated weights
    loaded_weights = aggregator.load_aggregated_model('aggregated_model.pkl')
    
    return aggregated_weights
# Federated_Learning_on_IoT

## Aggregator code (main_aggregator.py):
This implementation provides a flexible federated aggregator that can:

1. Aggregate weights from multiple models (though your use case focuses on two)
2. Support both PyTorch tensors and numpy arrays
3. Use weighted averaging based on the number of samples each model was trained on
4. Save and load aggregated models

For your Raspberry Pi setup, you would:

1. Train models on your edge nodes
2. Transfer the model weights to the Raspberry Pi (via Bluetooth/WiFi)
3. Use this aggregator to combine the weights
4. Save the aggregated model for future use

## Here's how you would use it in practice:

// On the Raspberry Pi
aggregator = FederatedAggregator()

// After receiving weights from edge devices
aggregated_weights = aggregator.aggregate_weights(
    model_weights=[edge1_weights, edge2_weights],
    sample_sizes=[edge1_samples, edge2_samples]
)

// Save the aggregated model
aggregator.save_aggregated_model(aggregated_weights, 'aggregated_model.pkl')
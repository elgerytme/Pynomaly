"""Basic usage example for neuro-symbolic AI package."""

from neuro_symbolic import NeuroSymbolicService


def main():
    """Demonstrate basic neuro-symbolic AI functionality."""
    
    # Initialize the service
    service = NeuroSymbolicService()
    
    # Create a neuro-symbolic model
    model = service.create_model(
        name="reasoning_system",
        neural_backbone="transformer", 
        symbolic_reasoner="first_order_logic"
    )
    
    print(f"Created model: {model.name} (ID: {model.id})")
    
    # Add symbolic constraints to model
    model.add_symbolic_constraint({
        "name": "variance_rule",
        "type": "logical",
        "rule": "IF variance > threshold THEN flag_as_notable",
        "confidence": 0.9
    })
    
    print(f"Added symbolic constraints to model")
    
    # Simulate training (normally would use real data)
    training_data = {"features": [[1, 2, 3], [4, 5, 6]], "labels": [0, 1]}
    service.train_model(model.id, training_data)
    
    print("Model trained successfully")
    
    # Make a prediction with explanation
    test_input = {"features": [7, 8, 9]}
    result = service.predict_with_explanation(model.id, test_input)
    
    print("\nPrediction Results:")
    print(f"  Prediction: {result.prediction}")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Explanation: {result.get_explanation_summary()}")
    
    # List all models
    models = service.list_models()
    print(f"\nTotal models registered: {len(models)}")


if __name__ == "__main__":
    main()
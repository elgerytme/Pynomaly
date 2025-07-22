"""Basic usage example for neuro-symbolic AI package."""

from neuro_symbolic import NeuroSymbolicService, KnowledgeGraph


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
    
    # Create a simple knowledge graph
    kg = KnowledgeGraph.create("reasoning_rules")
    kg.add_triple("Standard", "hasProperty", "LowVariance")
    kg.add_triple("Unusual", "hasProperty", "HighVariance")
    kg.add_triple("HighVariance", "indicates", "NotablePattern")
    
    # Register the knowledge graph
    service.knowledge_graphs[kg.id] = kg
    
    # Attach knowledge to model
    service.attach_knowledge_to_model(model.id, kg.id)
    
    print(f"Attached knowledge graph with {len(kg.triples)} triples")
    
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
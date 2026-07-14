import torch
from models.visual.transformer_encoder import VisualTransformerEncoder

def test_encoder():
    print("Testing the Visual Transformer Encoder limits securely...")

    # Validate mapping conditions
    batch_size = 4
    temporal_span = 150
    dimension = 393
    target_embedding = 256

    encoder = VisualTransformerEncoder(
        input_dim=dimension, 
        d_model=target_embedding,
        num_layers=4,
        nhead=8
    )

    # 1. Simulate structural tensor extraction directly imitating preprocessing outputs
    x = torch.randn(batch_size, temporal_span, dimension, requires_grad=True)

    # 2. Push inputs via encoder block
    output = encoder(x)

    # 3. Print structural integrity assertions
    print(f"Input Shape: {list(x.shape)}")
    print(f"Output Shape: {list(output.shape)}")
    print(f"Output dtype: {output.dtype}")

    nan_count = torch.isnan(output).sum().item()
    inf_count = torch.isinf(output).sum().item()
    print(f"NaN count: {nan_count}")
    print(f"Inf count: {inf_count}")

    # 4. Strict topology bounding validation
    assert output.shape == (batch_size, temporal_span, target_embedding), \
        f"Validation failed natively. Result shape {output.shape} violates expected {(batch_size, temporal_span, target_embedding)} constraints."
    
    # 5. Evaluate Gradient tracking validity asserting native topology connects securely backward.
    loss = output.mean()
    loss.backward()

    assert x.grad is not None, "Tensor graph traversal logic fractured! Loss gradients failed propagating upstream!"

    # 6. Success Output Print statement
    print("\nVISUAL ENCODER TEST PASSED")

if __name__ == "__main__":
    test_encoder()

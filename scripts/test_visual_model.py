import torch
from models.visual.visual_model import VisualModel

def test_visual_model():
    print("Testing the generalized Unified Visual Model constraints securely...")

    # 1. Instantiate the comprehensive graph
    model = VisualModel()
    
    # 2. Simulate raw preprocessing bounding arrays
    batch_size = 4
    x = torch.randn(batch_size, 150, 393, requires_grad=True)
    
    # 3. Push graph natively executing End-To-End 
    with torch.no_grad():
        encoded = model.encoder(x)
        
    logits = model(x)
    
    # 4. Print validations mapping dimensional boundaries
    print(f"Input Shape: {list(x.shape)}")
    print(f"Encoder Output Shape: {list(encoded.shape)}")
    print(f"Logits Shape: {list(logits.shape)}")
    print(f"Output dtype: {logits.dtype}")

    nan_count = torch.isnan(logits).sum().item()
    inf_count = torch.isinf(logits).sum().item()
    print(f"NaN Count: {nan_count}")
    print(f"Inf Count: {inf_count}")

    # 5. Assert Graph limits natively mapping
    assert logits.shape == (batch_size, 2), f"Validation failed natively. Result shape {logits.shape} violates expected {(batch_size, 2)} bounds."
    
    # 6. Evaluate Gradient propagation ensuring non-broken graph links 
    loss = logits.mean()
    loss.backward()

    valid_grads = True
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            print(f"Missing gradient bounds tracked back to {name}")
            valid_grads = False
            
    assert valid_grads, "Not all gradients tracked successfully securely."
    assert x.grad is not None, "Tensor input logic fractured! Loss gradients failed propagating upstream!"

    # 7. Success print
    print("\nVISUAL MODEL TEST PASSED")

if __name__ == "__main__":
    test_visual_model()

import torch


from squareCUDA import squareCUDA_Model



def test_module():
    model = squareCUDA_Model()
    x = torch.randn(1000, device='cuda', dtype=torch.float32, requires_grad=True)

    y_custom = model(x)

    y_ref = x ** 2

    forward_ok = torch.allclose(y_custom, y_ref, atol=1e-6)
    print(f"Forward match: {forward_ok}")

    loss = y_custom.sum()
    loss.backward()

    print("Input:", x[:5])
    print("Output:", y_custom[:5])
    print("Grad:", x.grad[:5])
    print("Expected grad ≈ 2 * x:", (2 * x)[:5])


def test_native():
    torch.autograd.set_detect_anomaly(True)

    x = torch.randn(1000, device='cuda', dtype=torch.float32, requires_grad=True)

    y_custom = squareCUDA.square_forward(x)

    y_ref = x ** 2
    
    forward_ok = torch.allclose(y_custom, y_ref, atol=1e-6)
    print(f"Forward match: {forward_ok}")

    grad_output = torch.ones_like(y_custom)

    grad_input_custom = squareCUDA.square_backward(grad_output, x)

    y_ref.sum().backward()
    grad_input_ref = x.grad.clone()

    backward_ok = torch.allclose(grad_input_custom, grad_input_ref, atol=1e-6)
    print(f"Backward match: {backward_ok}")


if __name__ == '__main__':

    print("test native")
    test_native()

    print("test module")
    test_module()


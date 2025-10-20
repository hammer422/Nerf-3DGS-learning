
import torch.nn as nn
from torch.autograd import Function

from . import _C


class _SquareFunction(Function):
    @staticmethod
    def forward(ctx, input):
        result = _C.square_forward(input)
        # 保存 input 用于 backward（如果 backward 需要）
        ctx.save_for_backward(input)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        # 获取保存的 input
        input, = ctx.saved_tensors
        grad_input = _C.square_backward(grad_output, input)
        return grad_input


def _squareCUDA_apply(inp):
    return _SquareFunction.apply(inp)


class squareCUDA_Model(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, inp):
        return _squareCUDA_apply(inp)

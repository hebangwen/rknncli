"""Operator metadata and color configuration."""

CATEGORY_COLORS = {
    'Layer': '#cfe8ff',
    'Activation': '#ffe3c7',
    'Normalization': '#e5d9ff',
    'Pool': '#d6f5d6',
    'Shape': '#fff3bf',
    'Tensor': '#dff7ff',
    'Math': '#ffd6e7',
    'Data': '#e6ffed',
    'Control': '#e5e7eb',
    'Other': '#eef2ff',
}

SPECIAL_OP_COLORS = {
    'InputOperator': '#d5f5e3',
    'OutputOperator': '#fdebd0',
}

OPERATOR_CATEGORIES = {
    'ActivationLayer': 'Activation',
    'Add': 'Other',
    'AveragePool': 'Pool',
    'BatchNormalization': 'Normalization',
    'BatchNormalizationLayer': 'Normalization',
    'Concat': 'Tensor',
    'Conv': 'Layer',
    'ConvAddRelu': 'Layer',
    'ConvClip': 'Layer',
    'ConvExSwish': 'Layer',
    'ConvLeakyRelu': 'Layer',
    'ConvLeakyReluAdd': 'Layer',
    'ConvRelu': 'Layer',
    'ConvReluAdd': 'Layer',
    'ConvSigmoid': 'Layer',
    'ConvTranspose': 'Layer',
    'ConvolutionLayer': 'Layer',
    'HardSigmoid': 'Activation',
    'LeakyRelu': 'Activation',
    'LeakyReluLayer': 'Activation',
    'MaxPool': 'Pool',
    'Pad': 'Tensor',
    'PoolingLayer2': 'Pool',
    'Relu': 'Activation',
    'Reshape': 'Shape',
    'Sigmoid': 'Activation',
    'Slice': 'Tensor',
    'SoftMax2': 'Activation',
    'Softmax': 'Activation',
    'Softmax2Layer': 'Activation',
    'Split': 'Tensor',
    'TensorScale': 'Layer',
    'TensorTranspose': 'Transform',
    'Transpose': 'Transform',
    'VSI_NN_OP_ADD': 'Other',
    'VSI_NN_OP_BATCH_NORM': 'Normalization',
    'VSI_NN_OP_CONCAT': 'Tensor',
    'VSI_NN_OP_CONV2D': 'Layer',
    'VSI_NN_OP_CONV_RELU': 'Layer',
    'VSI_NN_OP_CONV_RELU_POOL': 'Layer',
    'VSI_NN_OP_DECONVOLUTION': 'Layer',
    'VSI_NN_OP_FCL': 'Layer',
    'VSI_NN_OP_FCL_RELU': 'Layer',
    'VSI_NN_OP_INSTANCE_NORM': 'Normalization',
    'VSI_NN_OP_LEAKY_RELU': 'Activation',
    'VSI_NN_OP_LEAKY_SIGMOID': 'Activation',
    'VSI_NN_OP_LRN': 'Normalization',
    'VSI_NN_OP_LSTM': 'Layer',
    'VSI_NN_OP_MISH': 'Activation',
    'VSI_NN_OP_PERMUTE': 'Shape',
    'VSI_NN_OP_POOL': 'Pool',
    'VSI_NN_OP_PRELU': 'Activation',
    'VSI_NN_OP_RELU': 'Activation',
    'VSI_NN_OP_RELUN': 'Activation',
    'VSI_NN_OP_RESHAPE': 'Shape',
    'VSI_NN_OP_RESIZE': 'Shape',
    'VSI_NN_OP_SIGMOID': 'Activation',
    'VSI_NN_OP_SOFTMAX': 'Activation',
    'VSI_NN_OP_STRIDED_SLICE': 'Tensor',
    'VSI_NN_OP_SWISH': 'Activation',
    'exSoftmax13': 'Activation',
    'exSwish': 'Activation',
}


def get_node_color(op_type: str) -> str:
    """Return a fill color for the given operator type."""
    if op_type in SPECIAL_OP_COLORS:
        return SPECIAL_OP_COLORS[op_type]
    category = OPERATOR_CATEGORIES.get(op_type, "Other")
    return CATEGORY_COLORS.get(category, CATEGORY_COLORS["Other"])

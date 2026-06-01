import nn_core

class Module(nn_core.Module):
    """Python base class for composing NNEngine layers.

    Subclasses automatically register child layers assigned to `self` 
    and implement a forward method that accepts a Tensor object.

    Example:
        >>> class MyNet(Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.fc = nn.DenseLayer(16, 32) # Auto-registered!
        ...
        ...     def forward(self, x):
        ...         return self.fc(x)
    """

    def __init__(self):
        """Initialize the native module base class."""
        super().__init__()

    def __setattr__(self, name, value):
        """Automatically register layers to the C++ backend."""
        if isinstance(value, nn_core.Layer) or isinstance(value, nn_core.Module):
            self.add_module(value)
        super().__setattr__(name, value)
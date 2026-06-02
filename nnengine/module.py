import nn_core

class Module(nn_core.Module):
    """Python base class for composing NNEngine layers.

    Subclasses automatically register child layers assigned to `self` 
    and implement a forward method that accepts a Tensor object.
    
    This abstracts away the manual C++ module registry process.
    """

    def __init__(self):
        """Initialize the native module base class."""
        super().__init__()

    def __setattr__(self, name, value):
        """Intercept assignments and register them natively in C++.
        
        Args:
            name (str): The attribute name assigned to the layer.
            value (Any): The layer or module instance to bind.
            
        Raises:
            AttributeError: If a user attempts to reassign an already registered layer.
        """
        if isinstance(value, nn_core.Layer) or isinstance(value, nn_core.Module):
            if hasattr(self, name):
                raise AttributeError(f"Layer '{name}' is already registered. Reassignment is not allowed.")
            self.register_module(name, value)
            
        super().__setattr__(name, value)
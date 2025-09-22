import numpy as np
import json
import random
import numpy
import torch


def seed_everything(seed: int = 0):
    """
    Sets the seed for generating random numbers in PyTorch, NumPy, and Python.
    
    This function ensures reproducibility by setting fixed seeds for all random number
    generators used in the project.
    
    Parameters:
        seed (int, optional): The seed value to use. Defaults to 0.
        
    Returns:
        None
        
    Example:
        >>> seed_everything(42)
        # All subsequent random operations will now be deterministic
    """
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_lr_scheduler_with_warmup(optimizer, num_warmup_steps=None, num_steps=None, warmup_proportion=None,
                                 last_step=-1):
    """
    Creates a learning rate scheduler with a warmup phase.
    
    This function returns a PyTorch LambdaLR scheduler that linearly increases the
    learning rate from 0 to the initial learning rate over the first num_warmup_steps,
    and then maintains the learning rate constant afterward.
    
    Parameters:
        optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate
        num_warmup_steps (int, optional): The number of warmup steps. If None, calculated from
                                         num_steps and warmup_proportion
        num_steps (int, optional): The total number of training steps
        warmup_proportion (float, optional): The proportion of total steps to use for warmup
        last_step (int, optional): The index of the last step. Default is -1 (scheduler initialized)
        
    Returns:
        torch.optim.lr_scheduler.LambdaLR: The learning rate scheduler
        
    Raises:
        ValueError: If neither num_warmup_steps nor both num_steps and warmup_proportion are provided
        
    Example:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        >>> scheduler = get_lr_scheduler_with_warmup(optimizer, num_steps=1000, warmup_proportion=0.1)
    """
    if num_warmup_steps is None and (num_steps is None or warmup_proportion is None):
        raise ValueError(
            'Either num_warmup_steps or num_steps and warmup_proportion should be provided.')

    if num_warmup_steps is None:
        num_warmup_steps = int(num_steps * warmup_proportion)

    def get_lr_multiplier(step):
        if step < num_warmup_steps:
            return (step + 1) / (num_warmup_steps + 1)
        else:
            return 1

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=get_lr_multiplier, last_epoch=last_step)

    return lr_scheduler


def numpy_encoder(obj):
    """
    Custom JSON encoder for NumPy data types.
    
    This function helps serialize NumPy data types into JSON-compatible Python
    types. It handles NumPy integers, floats, and arrays by converting them to their
    standard Python equivalents.
    
    Parameters:
        obj: The object to encode
        
    Returns:
        The encoded object as a JSON-serializable Python type
        
    Raises:
        TypeError: If the object cannot be serialized
        
    Example:
        >>> import json
        >>> data = {'array': np.array([1, 2, 3])}
        >>> json.dumps(data, default=numpy_encoder)
        '{"array": [1, 2, 3]}'
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # let the encoder raise the usual TypeError for other types
    raise TypeError(f"Unserializable object {obj} of type {type(obj)}")


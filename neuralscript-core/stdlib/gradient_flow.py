# neuralscript-core/stdlib/gradient_flow.py
import numpy as np

# Mock Tape class
class GradientTape:
    def __init__(self):
        self.active = True
        self.watched_variables = []
        self.operations = []
        print("StdLib: GradientTape started.")

    def watch(self, variable_name):
        if self.active:
            self.watched_variables.append(variable_name)
            print(f"StdLib: GradientTape watching variable '{variable_name}'.")
        else:
            print("StdLib: Warning - Tape is not active, cannot watch variable.")
            
    def record_operation(self, op_name, inputs, output_name):
        if self.active:
            self.operations.append({'op': op_name, 'inputs': inputs, 'output': output_name})
            # print(f"StdLib: GradientTape recorded op '{op_name}'.") # Can be verbose
            
    def stop(self):
        self.active = False
        print("StdLib: GradientTape stopped.")

    def __repr__(self):
        return f"GradientTape(active={self.active}, watched={len(self.watched_variables)}, ops={len(self.operations)})"


# --- StdLib Gradient Functions ---
_active_tapes = [] # Global list of active tapes for simplicity in MVP

def start_gradient_tape():
    """Starts a new gradient tape and returns it."""
    tape = GradientTape()
    _active_tapes.append(tape) # Manage active tapes if needed globally
    return tape

def stop_gradient_tape(tape_object):
    """Stops the given gradient tape."""
    if not isinstance(tape_object, GradientTape):
        raise TypeError("Argument must be a GradientTape object.")
    tape_object.stop()
    if tape_object in _active_tapes:
        _active_tapes.remove(tape_object)
    return None # Or return status

def watch_variable(tape_object, variable_name: str):
    """Instructs the tape to watch a specific variable (by name)."""
    if not isinstance(tape_object, GradientTape):
        raise TypeError("First argument must be a GradientTape object.")
    # In a real system, this would involve the runtime passing the actual variable or its ID.
    # For placeholder, just passing name.
    tape_object.watch(variable_name)
    return None

def compute_gradients(tape_object, target_variable_name: str, source_variable_names: list):
    """
    Computes gradients of target_variable w.r.t. source_variables using the tape.
    Placeholder: Returns mock gradients.
    """
    if not isinstance(tape_object, GradientTape):
        raise TypeError("First argument must be a GradientTape object.")
    if tape_object.active:
        print("StdLib: Warning - Gradients computed on an active tape. Usually tape is stopped first.")
        
    print(f"StdLib: Attempting to compute gradients for target '{target_variable_name}' w.r.t {source_variable_names} using {tape_object}")
    
    mock_gradients = {}
    for src_var_name in source_variable_names:
        if src_var_name in tape_object.watched_variables:
            # Generate mock gradient based on variable name hash for some variation
            mock_gradients[src_var_name] = np.array([hash(src_var_name) % 100 / 100.0, 
                                                     (hash(src_var_name) >> 8) % 100 / 100.0]) 
            print(f"StdLib: Mock gradient for '{src_var_name}' = {mock_gradients[src_var_name]}")
        else:
            print(f"StdLib: Variable '{src_var_name}' was not watched by this tape, no gradient.")
            mock_gradients[src_var_name] = None # Or raise error

    return mock_gradients # This would be a dictionary of {var_name: gradient_array}

def apply_gradients(parameters: dict, gradients: dict):
    """
    Applies computed gradients to parameters.
    Placeholder: Prints parameters and their mock updated values.
    parameters: dict of {name: value_array}
    gradients: dict of {name: gradient_array}
    """
    print(f"StdLib: Attempting to apply gradients.")
    updated_parameters = {}
    learning_rate = 0.01 # Mock learning rate
    
    for name, param_value in parameters.items():
        if name in gradients and gradients[name] is not None:
            if not isinstance(param_value, np.ndarray) or not isinstance(gradients[name], np.ndarray):
                print(f"StdLib: Skipping gradient application for '{name}': parameter or gradient is not a numpy array.")
                updated_parameters[name] = param_value # Keep original
                continue
            if param_value.shape != gradients[name].shape:
                 print(f"StdLib: Skipping gradient application for '{name}': shape mismatch {param_value.shape} vs {gradients[name].shape}.")
                 updated_parameters[name] = param_value
                 continue

            updated_value = param_value - learning_rate * gradients[name]
            updated_parameters[name] = updated_value
            print(f"StdLib: Parameter '{name}' mock updated. Old mean: {np.mean(param_value)}, New mean: {np.mean(updated_value)}")
        else:
            updated_parameters[name] = param_value # No gradient, keep original
            print(f"StdLib: No gradient for parameter '{name}', kept original.")
            
    return updated_parameters # Returns a dict of updated parameters


STD_LIB_GRADIENT_FUNCTIONS = {
    "start_tape": start_gradient_tape,
    "stop_tape": stop_gradient_tape,
    "watch_variable": watch_variable, # Watch a variable (by name for now)
    "compute_gradients": compute_gradients,
    "apply_gradients": apply_gradients,
}

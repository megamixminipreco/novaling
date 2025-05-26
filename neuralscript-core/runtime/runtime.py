# neuralscript-core/runtime/runtime.py
import numpy as np
# Assuming compiler.py and its ASTNode definition are in the parent directory
# For testing, we might need to adjust paths or ensure ASTNode is available.

# --- Minimal ASTNode definition for runtime tests if not importing from compiler ---
# This should ideally be a shared component.
class ASTNode:
    def __init__(self, node_type, children=None, value=None, **kwargs):
        self.type = node_type
        self.value = value
        if node_type == 'NeuralOperation' and 'arguments' in kwargs: # Compatibility with compiler's test ASTs
            self.children = kwargs['arguments'] if kwargs['arguments'] is not None else []
        elif children is not None:
            self.children = children
        else:
            self.children = []
        self.__dict__.update(kwargs)

# --- Import StdLib ---
# This import path assumes 'neuralscript-core' is the root package or on PYTHONPATH
# For direct execution of runtime.py, this might need adjustment or be handled by __main__
try:
    from .stdlib.vector_operations import STD_LIB_VECTOR_FUNCTIONS
except ImportError:
    # Fallback for direct execution if neuralscript-core is not in PYTHONPATH
    # This is common for modular development testing.
    # print("Runtime: Could not perform relative import for stdlib.vector_operations. Attempting direct.")
    try:
        from stdlib.vector_operations import STD_LIB_VECTOR_FUNCTIONS
    except ImportError:
        # print("Runtime: Could not import STD_LIB_VECTOR_FUNCTIONS. Stdlib calls might fail.")
        STD_LIB_VECTOR_FUNCTIONS = {} # Ensure it exists to prevent crash

class Runtime:
    def __init__(self):
        self.variables = {}  # For storing variable values (simple global scope for now)
        self.stack = []      # Operand stack for calculations
        self.call_stack = [] # For managing function call contexts (return PC, local vars)
        self.pc = 0          # Program Counter
        self.bytecode = []
        self.functions = {}  # To store metadata about user-defined functions {name: {pc_start, params}}
        self.labels = {}     # To store PC for labels {label_name: pc}
        
        # Standard Library
        self.stdlib_functions = {}
        self.op_handlers = {
            '+': lambda a, b: a + b,
            '-': lambda a, b: a - b,
            '*': lambda a, b: a * b,
            '/': lambda a, b: a / b if b != 0 else float('inf'), # Basic division, handle zero
            '==': lambda a, b: a == b,
            '!=': lambda a, b: a != b,
            '<': lambda a, b: a < b,
            '>': lambda a, b: a > b,
            '<=': lambda a, b: a <= b,
            '>=': lambda a, b: a >= b,
        }
        # Register stdlib modules
        if STD_LIB_VECTOR_FUNCTIONS: # Only if successfully imported
            self.register_stdlib_module(STD_LIB_VECTOR_FUNCTIONS)


    def register_stdlib_module(self, module_functions):
        for name, func in module_functions.items():
            if name in self.stdlib_functions:
                print(f"Warning: Stdlib function {name} multiply defined. Overwriting.")
            self.stdlib_functions[name] = func
            # print(f"Runtime: Registered stdlib function '{name}'.") # Optional: for debugging

    def _preprocess_labels_and_functions(self, bytecode):
        self.labels = {}
        self.functions = {}
        for i, (instr, *args) in enumerate(bytecode):
            if instr == 'LABEL':
                self.labels[args[0]] = i
            elif instr == 'FUNC_BEGIN':
                func_name = args[0]
                param_names = args[1] if len(args) > 1 else []
                self.functions[func_name] = {'pc_start': i, 'params': param_names}
                # print(f"Runtime: Registered function '{func_name}' at PC {i} with params {param_names}")


    def execute(self, bytecode):
        self.bytecode = bytecode
        self._preprocess_labels_and_functions(bytecode)
        self.pc = 0
        
        # Find the first instruction that is not a LABEL or FUNC_BEGIN to start execution,
        # unless we are jumping directly to a function (e.g. main, not implemented yet)
        # This simple loop skips over function definitions at the start of bytecode.
        while self.pc < len(self.bytecode):
            instr_type = self.bytecode[self.pc][0]
            if instr_type not in ['LABEL', 'FUNC_BEGIN']:
                break
            self.pc += 1
        
        # Main execution loop
        while self.pc < len(self.bytecode):
            instr, *args = self.bytecode[self.pc]
            # print(f"DEBUG: PC={self.pc}, Instr: {instr}, Args: {args}, Stack: {self.stack}, Vars: {self.variables}")

            consumed_instruction = True # Assume PC advances by 1 unless instr modifies it

            if instr == 'LOAD_CONST':
                self.stack.append(args[0])
            elif instr == 'LOAD_VAR':
                var_name = args[0]
                if var_name not in self.variables:
                    raise NameError(f"Variable '{var_name}' not defined.")
                self.stack.append(self.variables[var_name])
            elif instr == 'STORE_VAR':
                var_name = args[0]
                if not self.stack:
                    raise IndexError("Stack is empty, cannot store variable.")
                self.variables[var_name] = self.stack.pop()
            elif instr == 'MAKE_VECTOR':
                num_elements = args[0]
                if len(self.stack) < num_elements:
                    raise IndexError(f"Not enough elements on stack to make vector of size {num_elements}")
                elements = self.stack[-num_elements:]
                self.stack = self.stack[:-num_elements] # Pop elements
                self.stack.append(np.array(elements))
            elif instr == 'CALL_OP':
                op_symbol = args[0]
                num_op_args = args[1]
                if len(self.stack) < num_op_args:
                    raise IndexError(f"Not enough operands on stack for operation '{op_symbol}'")
                
                op_args = self.stack[-num_op_args:]
                self.stack = self.stack[:-num_op_args] # Pop operands

                if op_symbol not in self.op_handlers:
                    raise NotImplementedError(f"Operation '{op_symbol}' not implemented.")
                result = self.op_handlers[op_symbol](*op_args) # Unpack args for the handler
                self.stack.append(result)
            
            elif instr == 'NEURAL_OP':
                op_name = args[0]
                # For a non-stack machine version, arg_names_or_values = args[1]
                # For a stack machine, num_args = args[1], and we'd pop from stack.
                # The compiler currently produces resolved names/values in args[1].
                # This runtime part needs to be consistent with the compiler's NEURAL_OP output.
                # Assuming compiler gives ['var_name', literal_val, ...]
                
                resolved_args = []
                arg_spec = args[1] # This is the list like ['q', 'k', 'v'] from compiler
                for item in arg_spec:
                    if isinstance(item, str) and item in self.variables: # It's a variable name
                        resolved_args.append(self.variables[item])
                    else: # It's a literal value
                        resolved_args.append(item)

                # Placeholder for actual neural operation dispatch
                print(f"Runtime: Executing NEURAL_OP '{op_name}' with args {resolved_args} (result is placeholder)")
                if op_name == "ATTEND": # Example
                    if len(resolved_args) != 3: raise ValueError("ATTEND expects 3 arguments")
                    # query, key, value = resolved_args
                    # result = np.dot(query, key.T) * value # Highly simplified placeholder
                    result = np.array([np.sum(arg) for arg in resolved_args]) # Placeholder
                    self.stack.append(result) # Push placeholder result
                else:
                    self.stack.append(np.array([0.0])) # Default placeholder for other ops


            elif instr == 'JUMP':
                self.pc = self.labels[args[0]]
                consumed_instruction = False # PC set directly
            elif instr == 'JUMP_IF_FALSE':
                if not self.stack: raise IndexError("Stack empty for JUMP_IF_FALSE condition.")
                condition = self.stack.pop()
                if not condition:
                    self.pc = self.labels[args[0]]
                    consumed_instruction = False
            
            elif instr == 'FUNC_BEGIN':
                # This instruction is mainly for preprocessing; skip during normal execution.
                # Runtime should jump over function bodies unless called.
                # Find the corresponding FUNC_END label and jump PC after it
                # This is a simple way to skip function bodies if they are inlined.
                func_name = args[0]
                end_label_name = f"FUNC_{func_name}_END" # Compiler needs to use consistent naming
                
                # This simplistic skip might be problematic if functions are not defined at top level
                # or if there are nested functions. Preprocessing already stored start PC.
                # The main execution loop at the start of execute() should skip these.
                # If execution somehow lands here, it means it's trying to execute a function body directly.
                # This might be okay if the function is implicitly called (e.g. 'main'), or an error.
                # print(f"Warning: PC landed inside FUNC_BEGIN for {func_name}. Skipping to its END label.")
                # self.pc = self.labels[end_label_name] +1 # +1 to go after the label
                # consumed_instruction = False
                pass # PC will advance, effectively skipping this if not jumped over

            elif instr == 'CALL_FUNC':
                func_name = args[0]
                arg_count = args[1]

                if func_name not in self.functions: # Not a user-defined function
                    if func_name in self.stdlib_functions: # Check stdlib
                        if len(self.stack) < arg_count:
                            raise IndexError(f"Not enough arguments on stack for stdlib call {func_name}")
                        
                        args_for_stdlib = self.stack[-arg_count:]
                        self.stack = self.stack[:-arg_count] # Pop args
                        
                        try:
                            # print(f"DEBUG: Calling stdlib {func_name} with args {args_for_stdlib}")
                            result = self.stdlib_functions[func_name](*args_for_stdlib)
                            self.stack.append(result)
                        except Exception as e:
                            print(f"Error during stdlib function '{func_name}' execution: {e}")
                            # Potentially push a specific error object or re-raise
                            raise # Re-raise for now
                        
                        # self.pc += 1 # Advance PC after stdlib call
                        # consumed_instruction = True # PC already advanced or will be by loop
                        # continue # Skip rest of user-function call logic
                    else:
                        raise NameError(f"Function '{func_name}' not defined as user function or stdlib.")
                else: # User-defined function
                    if len(self.stack) < arg_count:
                        raise IndexError(f"Not enough arguments on stack for call to {func_name}")
                    
                    func_info = self.functions[func_name]
                    if len(func_info['params']) != arg_count:
                        raise ValueError(f"Function {func_name} expects {len(func_info['params'])} args, got {arg_count}")

                    # Setup new call frame
                    # Frame: (return_pc, current_locals_snapshot)
                    # Simple local scope: new dict for each call. More complex would involve chaining.
                    current_locals = {} 
                    
                    # Pop args from stack and assign to params in new local scope
                    # Args are popped in reverse order of params
                    for param_name in reversed(func_info['params']):
                        current_locals[param_name] = self.stack.pop()
                    
                    # Store current variables (for restoring outer scope) and push new locals
                    self.call_stack.append({'return_pc': self.pc + 1, 'variables': self.variables})
                    self.variables = current_locals # Switch to new local scope
                    
                    self.pc = func_info['pc_start'] # Jump to function's start
                    # PC will then naturally skip FUNC_BEGIN and go to first actual instruction
                    consumed_instruction = False 


            elif instr == 'RETURN_VALUE':
                if not self.stack: raise IndexError("Stack empty for RETURN_VALUE.")
                return_value = self.stack.pop()
                
                if not self.call_stack: raise RuntimeError("Call stack empty, cannot return.")
                
                # Restore previous frame
                previous_frame = self.call_stack.pop()
                self.variables = previous_frame['variables'] # Restore outer scope's variables
                self.pc = previous_frame['return_pc']
                
                self.stack.append(return_value) # Push return value onto caller's stack
                consumed_instruction = False

            elif instr == 'RETURN_VOID':
                if not self.call_stack: raise RuntimeError("Call stack empty, cannot return.")

                previous_frame = self.call_stack.pop()
                self.variables = previous_frame['variables']
                self.pc = previous_frame['return_pc']
                # No return value to push for VOID return
                consumed_instruction = False
            
            elif instr == 'LABEL':
                pass # Labels are targets for jumps, do nothing when PC flows over them

            else:
                raise NotImplementedError(f"Unknown bytecode instruction: {instr}")

            if consumed_instruction:
                self.pc += 1
        
        # print("Runtime execution finished.")
        # print("Final Stack:", self.stack)
        # print("Final Variables:", self.variables)


# --- Main for testing Runtime ---
if __name__ == '__main__':
    # Minimal Compiler for generating test bytecode (copied from compiler.py's main)
    class Compiler:
        def __init__(self): self.bytecode = []; self.label_count = 0
        def _generate_label(self, prefix="L"): self.label_count += 1; return f"{prefix}{self.label_count}"
        def compile(self, node):
            self.bytecode = []; self.label_count = 0
            if node is None: return []
            self.visit(node)
            return self.bytecode
        def visit(self, node):
            if node is None: return None
            method_name = f'visit_{node.type}'; visitor = getattr(self, method_name, self.generic_visit)
            return visitor(node)
        def generic_visit(self, node): raise Exception(f'No visit_{node.type} for mini-compiler, value: {getattr(node,"value","N/A")}')
        def visit_Program(self, node): 
            if node.children: 
                for stmt in node.children: self.visit(stmt)
        def visit_NumberLiteral(self, node): self.bytecode.append(('LOAD_CONST', node.value)); return {'type':'literal','value':node.value}
        def visit_Identifier(self, node): self.bytecode.append(('LOAD_VAR', node.name)); return {'type':'variable','name':node.name}
        def visit_VectorLiteral(self, node): 
            for el in node.children: self.visit(el)
            self.bytecode.append(('MAKE_VECTOR', len(node.children))); return {'type':'vector'}
        def visit_Assignment(self, node): self.visit(node.children[0]); self.bytecode.append(('STORE_VAR', node.name))
        def visit_BinaryOperation(self, node): self.visit(node.children[0]); self.visit(node.children[1]); self.bytecode.append(('CALL_OP', node.operator, 2)); return {'type':'binary_op'}
        def visit_If(self, node):
            self.visit(node.children[0]) # Condition
            else_lbl = self._generate_label("IFELSE"); end_lbl = self._generate_label("IFEND")
            self.bytecode.append(('JUMP_IF_FALSE', else_lbl if node.has_else else end_lbl))
            self.visit(node.children[1]) # Then branch
            if node.has_else: self.bytecode.append(('JUMP', end_lbl))
            self.bytecode.append(('LABEL', else_lbl))
            if node.has_else: self.visit(node.children[2]) # Else branch
            self.bytecode.append(('LABEL', end_lbl))
        def visit_FunctionDef(self, node):
            start_lbl = self._generate_label(f"FUNC_{node.name}_START"); end_lbl = self._generate_label(f"FUNC_{node.name}_END")
            self.bytecode.append(('LABEL', start_lbl))
            self.bytecode.append(('FUNC_BEGIN', node.name, [p.name for p in node.params if hasattr(p, 'name')]))
            # Body of FunctionDef is a ProgramNode in parser, but list of stmts in prompt's test factory
            if node.children and isinstance(node.children[0], ASTNode) and node.children[0].type == 'Program':
                 self.visit(node.children[0])
            else: # Assuming children is list of statements for test factory
                 for stmt in node.children: self.visit(stmt)
            self.bytecode.append(('LABEL', end_lbl))
        def visit_Return(self, node):
            if node.children and node.children[0] is not None: self.visit(node.children[0]); self.bytecode.append(('RETURN_VALUE',))
            else: self.bytecode.append(('RETURN_VOID',))
        def visit_Call(self, node):
            for arg in node.children: self.visit(arg)
            self.bytecode.append(('CALL_FUNC', node.callee, len(node.children)))
            return {'type':'call_result'}


    # AST Node factories (simplified for testing)
    def ProgramNode(stmts): return ASTNode('Program', children=stmts)
    def AssignmentNode(name, expr): return ASTNode('Assignment', children=[expr], name=name)
    def IdentifierNode(name): return ASTNode('Identifier', name=name)
    def NumberLiteralNode(val): return ASTNode('NumberLiteral', value=val)
    def VectorLiteralNode(elements): return ASTNode('VectorLiteral', children=elements)
    def CallNode(callee, args): return ASTNode('Call', children=args, callee=callee)
    def BinaryOperationNode(left, op, right): return ASTNode('BinaryOperation', children=[left,right], operator=op)
    def FunctionDefNode(name, params, body_stmts, rt=None): return ASTNode('FunctionDef', children=body_stmts, name=name, params=params, return_type=rt)
    def ReturnNode(expr=None): # For RETURN; use ReturnNode(). For RETURN val; use ReturnNode(val_node)
        return ASTNode('Return', children=[expr] if expr is not None else [])


    # --- Test Case 1: Basic Arithmetic and Variable Assignment ---
    print("--- Runtime Test Case 1: Basic Ops ---")
    compiler1 = Compiler()
    runtime1 = Runtime()
    ast1 = ProgramNode([
        AssignmentNode(name='x', value_expr=NumberLiteralNode(10)),
        AssignmentNode(name='y', value_expr=NumberLiteralNode(20)),
        AssignmentNode(name='z', value_expr=BinaryOperationNode(
            IdentifierNode('x'), '+', IdentifierNode('y')
        ))
    ])
    bytecode1 = compiler1.compile(ast1)
    # print("Bytecode 1:", bytecode1)
    runtime1.execute(bytecode1)
    assert runtime1.variables['x'] == 10
    assert runtime1.variables['y'] == 20
    assert runtime1.variables['z'] == 30
    print("Test Case 1 Passed. z =", runtime1.variables['z'])

    # --- Test Case 2: Vector Creation and Assignment ---
    print("\n--- Runtime Test Case 2: Vector Ops ---")
    compiler2 = Compiler()
    runtime2 = Runtime()
    ast2 = ProgramNode([
        AssignmentNode(name='v', value_expr=VectorLiteralNode([
            NumberLiteralNode(1.0), NumberLiteralNode(2.5), NumberLiteralNode(3.0)
        ]))
    ])
    bytecode2 = compiler2.compile(ast2)
    # print("Bytecode 2:", bytecode2)
    runtime2.execute(bytecode2)
    assert isinstance(runtime2.variables['v'], np.ndarray)
    assert np.array_equal(runtime2.variables['v'], np.array([1.0, 2.5, 3.0]))
    print("Test Case 2 Passed. v =", runtime2.variables['v'])

    # --- Test Case 3: Simple Function Definition and Call ---
    print("\n--- Runtime Test Case 3: Function Def & Call ---")
    compiler3 = Compiler()
    runtime3 = Runtime()
    ast3 = ProgramNode([
        FunctionDefNode(name="add_five", params=[IdentifierNode("val")], body_stmts=[
            ReturnNode(expr=BinaryOperationNode(IdentifierNode("val"), "+", NumberLiteralNode(5)))
        ]),
        AssignmentNode(name="result", value_expr=CallNode(
            callee="add_five", args=[NumberLiteralNode(10)]
        ))
    ])
    bytecode3 = compiler3.compile(ast3)
    # print("Bytecode 3:", bytecode3)
    runtime3.execute(bytecode3)
    assert runtime3.variables['result'] == 15
    print("Test Case 3 Passed. result =", runtime3.variables['result'])

    # --- Test Case 4: Function Call with Local Scopes and Outer Scope Preservation ---
    print("\n--- Runtime Test Case 4: Scopes & Function Call ---")
    compiler4 = Compiler()
    runtime4 = Runtime()
    ast4 = ProgramNode([
        AssignmentNode(name='a', value_expr=NumberLiteralNode(100)), # Outer scope 'a'
        FunctionDefNode(name="scoped_func", params=[IdentifierNode("a")], body_stmts=[ # Inner scope 'a' (param)
            AssignmentNode(name='b', value_expr=BinaryOperationNode(
                IdentifierNode("a"), "*", NumberLiteralNode(2) # Uses inner 'a'
            )),
            ReturnNode(expr=IdentifierNode('b'))
        ]),
        AssignmentNode(name="res_b", value_expr=CallNode(
            callee="scoped_func", args=[NumberLiteralNode(5)] # Call with 5, so inner 'a' is 5
        )),
        # After call, check if outer 'a' is preserved and 'b' is not in outer scope
    ])
    bytecode4 = compiler4.compile(ast4)
    # print("Bytecode 4:", bytecode4)
    runtime4.execute(bytecode4)
    assert runtime4.variables['a'] == 100, f"Outer scope 'a' was {runtime4.variables.get('a')}"
    assert 'b' not in runtime4.variables, f"'b' leaked to outer scope: {runtime4.variables.get('b')}"
    assert runtime4.variables['res_b'] == 10, f"Result of scoped_func was {runtime4.variables.get('res_b')}"
    print("Test Case 4 Passed. res_b =", runtime4.variables.get('res_b'), ", outer 'a' =", runtime4.variables.get('a'))


    # --- Test Case 5: StdLib Vector Function Call ---
    print("\n--- Runtime Test Case 5: StdLib vector_dot call ---")
    compiler_std = Compiler() 
    runtime_std = Runtime()

    ast5_setup = ProgramNode([
        AssignmentNode(name='v1', value_expr=VectorLiteralNode([
            NumberLiteralNode(1), NumberLiteralNode(2), NumberLiteralNode(3)
        ])),
        AssignmentNode(name='v2', value_expr=VectorLiteralNode([
            NumberLiteralNode(4), NumberLiteralNode(5), NumberLiteralNode(6)
        ]))
    ])
    bytecode5_setup = compiler_std.compile(ast5_setup)
    runtime_std.execute(bytecode5_setup)
    # print("Vars after setup:", runtime_std.variables)
    assert np.array_equal(runtime_std.variables['v1'], np.array([1.,2.,3.]))
    assert np.array_equal(runtime_std.variables['v2'], np.array([4.,5.,6.]))

    ast5_call = ProgramNode([
        AssignmentNode(name='dot_res', value_expr=CallNode(
            callee="vector_dot", 
            args=[IdentifierNode('v1'), IdentifierNode('v2')]
        ))
    ])
    bytecode5_call = compiler_std.compile(ast5_call)
    # print("Bytecode for call:", bytecode5_call)
    
    runtime_std.execute(bytecode5_call)
    # print("Vars after stdlib call:", runtime_std.variables)
    # Expected dot product: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert runtime_std.variables.get('dot_res') == 32.0
    
    print("Stdlib vector_dot test passed.")

# neuralscript-core/runtime/runtime.py
import numpy as np
try:
    from .neural_memory import NeuralMemory 
except ImportError:
    NeuralMemory = None 

try:
    from .stdlib.vector_operations import STD_LIB_VECTOR_FUNCTIONS
except ImportError:
    try: from stdlib.vector_operations import STD_LIB_VECTOR_FUNCTIONS 
    except ImportError: STD_LIB_VECTOR_FUNCTIONS = {}; # print("Runtime Warning: stdlib.vector_operations not found.")

try:
    from .stdlib.attention_mechanisms import STD_LIB_ATTENTION_FUNCTIONS 
except ImportError:
    try: from stdlib.attention_mechanisms import STD_LIB_ATTENTION_FUNCTIONS 
    except ImportError: STD_LIB_ATTENTION_FUNCTIONS = {}; # print("Runtime Warning: stdlib.attention_mechanisms not found.")

try:
    from .stdlib.probability_distributions import STD_LIB_DISTRIBUTION_FACTORIES_AND_OPS
except ImportError:
    try: from stdlib.probability_distributions import STD_LIB_DISTRIBUTION_FACTORIES_AND_OPS
    except ImportError: STD_LIB_DISTRIBUTION_FACTORIES_AND_OPS = {}; # print("Runtime Warning: stdlib.probability_distributions not found.")

try:
    from .stdlib.gradient_flow import STD_LIB_GRADIENT_FUNCTIONS # New import
except ImportError:
    try: from stdlib.gradient_flow import STD_LIB_GRADIENT_FUNCTIONS
    except ImportError: STD_LIB_GRADIENT_FUNCTIONS = {}; # print("Runtime Warning: stdlib.gradient_flow not found.")


class RuntimeErrorNS(Exception):
    def __init__(self, message, pc=None, instruction=None):
        full_message = "RuntimeErrorNS: "
        if pc is not None:
            full_message += f"[PC:{pc}] "
        if instruction is not None:
            full_message += f"Instruction: {instruction} - "
        full_message += message
        super().__init__(full_message)
        self.pc = pc
        self.instruction = instruction

class Runtime:
    def __init__(self):
        self.stack = []
        self.variables = {} 
        self.bytecode = []
        self.pc = 0
        self.labels = {}
        self.call_stack = [] 
        self.functions = {} 
        self.neural_memory = NeuralMemory() if NeuralMemory else None
        
        self.stdlib_functions = {} 
        if STD_LIB_VECTOR_FUNCTIONS: 
            self.register_stdlib_module(STD_LIB_VECTOR_FUNCTIONS)
        if STD_LIB_ATTENTION_FUNCTIONS: 
            self.register_stdlib_module(STD_LIB_ATTENTION_FUNCTIONS) 
        if STD_LIB_DISTRIBUTION_FACTORIES_AND_OPS: 
            self.register_stdlib_module(STD_LIB_DISTRIBUTION_FACTORIES_AND_OPS)
        if STD_LIB_GRADIENT_FUNCTIONS: # New registration
            self.register_stdlib_module(STD_LIB_GRADIENT_FUNCTIONS)

        self.op_handlers = {
            '+': lambda a, b: a + b, '-': lambda a, b: a - b,
            '*': lambda a, b: a * b, '/': lambda a, b: a / b if b != 0 else float('inf'),
            '==': lambda a, b: a == b, '!=': lambda a, b: a != b,
            '<': lambda a, b: a < b, '>': lambda a, b: a > b,
            '<=': lambda a, b: a <= b, '>=': lambda a, b: a >= b,
        }

    def register_stdlib_module(self, module_functions):
        for name, func in module_functions.items():
            self.stdlib_functions[name] = func

    def _preprocess_labels_and_functions(self, bytecode):
        self.labels = {}
        self.functions = {} 
        for instruction_idx, instruction in enumerate(bytecode):
            op_code = instruction[0]
            if op_code == 'LABEL':
                self.labels[instruction[1]] = instruction_idx 
            elif op_code == 'FUNC_BEGIN':
                self.functions[instruction[1]] = instruction_idx 


    def execute(self, bytecode):
        self.bytecode = bytecode
        if not self.bytecode: return None
        self._preprocess_labels_and_functions(bytecode)
        self.pc = 0
        self.stack = []
        # self.variables = {} # Commented out to persist global variables across multiple execute calls in tests
        self.call_stack = [] 

        initial_pc_set = False
        for i, instruction in enumerate(self.bytecode):
            if instruction[0] not in ['LABEL', 'FUNC_BEGIN']:
                self.pc = i
                initial_pc_set = True
                break
        if not initial_pc_set and self.bytecode: 
            return None


        while self.pc < len(self.bytecode):
            instruction = self.bytecode[self.pc]
            op_code = instruction[0]
            
            if op_code == 'LABEL' or op_code == 'FUNC_BEGIN':
                self.pc += 1
                continue
            
            current_pc_for_error = self.pc 
            consumed_instruction = True

            try:
                if op_code == 'LOAD_CONST':
                    self.stack.append(instruction[1])
                elif op_code == 'LOAD_VAR':
                    var_name = instruction[1]
                    val = None; found = False
                    if self.call_stack and var_name in self.call_stack[-1]['locals']:
                         val = self.call_stack[-1]['locals'][var_name]; found = True
                    elif var_name in self.variables:
                        val = self.variables[var_name]; found = True
                    
                    if not found: raise RuntimeErrorNS(f"Variable '{var_name}' not found.", current_pc_for_error, instruction)
                    self.stack.append(val)

                elif op_code == 'STORE_VAR':
                    if not self.stack: raise RuntimeErrorNS("Stack is empty for STORE_VAR.", current_pc_for_error, instruction)
                    var_name = instruction[1]
                    value_to_store = self.stack.pop()
                    if self.call_stack: self.call_stack[-1]['locals'][var_name] = value_to_store
                    else: self.variables[var_name] = value_to_store

                elif op_code == 'MAKE_VECTOR':
                    count = instruction[1]
                    if len(self.stack) < count: raise RuntimeErrorNS(f"Stack underflow for MAKE_VECTOR (need {count}, have {len(self.stack)}).", current_pc_for_error, instruction)
                    elements = self.stack[-count:]
                    self.stack = self.stack[:-count]
                    self.stack.append(np.array(elements))

                elif op_code == 'CALL_OP':
                    op_symbol, arg_count = instruction[1], instruction[2]
                    if len(self.stack) < arg_count: raise RuntimeErrorNS(f"Stack underflow for '{op_symbol}' (need {arg_count}, have {len(self.stack)}).", current_pc_for_error, instruction)
                    args_val = self.stack[-arg_count:] 
                    self.stack = self.stack[:-arg_count]
                    if op_symbol not in self.op_handlers: raise RuntimeErrorNS(f"Operator '{op_symbol}' not implemented.", current_pc_for_error, instruction)
                    try:
                        if arg_count == 2: result = self.op_handlers[op_symbol](args_val[0], args_val[1])
                        elif arg_count == 1: result = self.op_handlers[op_symbol](args_val[0]) 
                        else: raise RuntimeErrorNS(f"Unsupported arg_count {arg_count} for CALL_OP '{op_symbol}'.", current_pc_for_error, instruction)
                        self.stack.append(result)
                    except ZeroDivisionError: raise RuntimeErrorNS(f"Division by zero during '{op_symbol}'.", current_pc_for_error, instruction) from None
                    except TypeError as te: raise RuntimeErrorNS(f"Type error during '{op_symbol}': {te}. Args: {args_val}", current_pc_for_error, instruction) from te
                
                elif op_code == 'NEURAL_OP':
                    op_name, arg_sources = instruction[1], instruction[2]
                    resolved_args = []
                    for src_idx, source_name_or_val in enumerate(arg_sources):
                        val_arg = None; found_arg = False
                        if isinstance(source_name_or_val, (int, float, str, bool, np.ndarray)): 
                            val_arg=source_name_or_val; found_arg=True
                        else: 
                            if self.call_stack and source_name_or_val in self.call_stack[-1]['locals']: 
                                val_arg = self.call_stack[-1]['locals'][source_name_or_val]; found_arg=True
                            elif source_name_or_val in self.variables: 
                                val_arg = self.variables[source_name_or_val]; found_arg=True
                        if not found_arg: raise RuntimeErrorNS(f"Variable '{source_name_or_val}' (arg {src_idx} for NEURAL_OP '{op_name}') not found.", current_pc_for_error, instruction)
                        resolved_args.append(val_arg)
                    
                    if op_name == "ATTEND":
                        if len(resolved_args) != 3: raise RuntimeErrorNS(f"NEURAL_OP 'ATTEND' expects 3 resolved arguments, got {len(resolved_args)}.", current_pc_for_error, instruction)
                        self.stack.append(f"mock_attention_output({resolved_args[0]},{resolved_args[1]},{resolved_args[2]})") 
                    elif op_name == "MERGE":
                        if len(resolved_args) != 2: raise RuntimeErrorNS(f"NEURAL_OP 'MERGE' expects 2 resolved arguments, got {len(resolved_args)}.", current_pc_for_error, instruction)
                        self.stack.append(f"mock_merge_output({resolved_args[0]},{resolved_args[1]})") 
                    else: self.stack.append(f"mock_output_for_{op_name}") 

                elif op_code == 'JUMP':
                    self.pc = self.labels[instruction[1]]; consumed_instruction = False
                elif op_code == 'JUMP_IF_FALSE':
                    if not self.stack: raise RuntimeErrorNS("Stack empty for JUMP_IF_FALSE condition.", current_pc_for_error, instruction)
                    condition = self.stack.pop()
                    if not condition: self.pc = self.labels[instruction[1]]; consumed_instruction = False
                
                elif op_code == 'CALL_FUNC':
                    func_name, arg_count = instruction[1], instruction[2]
                    is_stdlib = func_name in self.stdlib_functions
                    is_userdef = func_name in self.functions

                    if not is_stdlib and not is_userdef: raise RuntimeErrorNS(f"Function '{func_name}' not defined.", current_pc_for_error, instruction)
                    if len(self.stack) < arg_count: raise RuntimeErrorNS(f"Stack underflow for CALL_FUNC '{func_name}' (need {arg_count}, have {len(self.stack)}).", current_pc_for_error, instruction)
                    
                    args_for_call = self.stack[-arg_count:]
                    self.stack = self.stack[:-arg_count] 

                    if is_stdlib:
                        try:
                            result = self.stdlib_functions[func_name](*args_for_call)
                            self.stack.append(result) 
                        except Exception as e: raise RuntimeErrorNS(f"Error in stdlib function '{func_name}': {type(e).__name__} - {e}", current_pc_for_error, instruction) from e
                    else: 
                        func_begin_pc = self.functions[func_name] 
                        func_def_instruction = self.bytecode[func_begin_pc] 
                        param_names = func_def_instruction[2] 

                        if len(param_names) != arg_count: raise RuntimeErrorNS(f"Function '{func_name}' defined at BC PC {func_begin_pc} expected {len(param_names)} args, got {arg_count}.", current_pc_for_error, instruction)
                        
                        frame = {'return_pc': self.pc + 1, 'locals': {}, 'name': func_name}
                        for i in range(arg_count): frame['locals'][param_names[i]] = args_for_call[i]
                        
                        self.call_stack.append(frame)
                        self.pc = func_begin_pc + 1; consumed_instruction = False 
                
                elif op_code == 'RETURN_VALUE':
                    if not self.call_stack: raise RuntimeErrorNS("RETURN outside of function call.", current_pc_for_error, instruction)
                    if not self.stack: raise RuntimeErrorNS("Stack empty on RETURN_VALUE (no value to return).", current_pc_for_error, instruction)
                    # Value to return is already on top of the stack.
                    frame = self.call_stack.pop(); self.pc = frame['return_pc']; consumed_instruction = False
                
                elif op_code == 'RETURN_VOID':
                    if not self.call_stack: raise RuntimeErrorNS("RETURN (void) outside of function call.", current_pc_for_error, instruction)
                    frame = self.call_stack.pop(); self.pc = frame['return_pc']; consumed_instruction = False
                else:
                    raise RuntimeErrorNS(f"Unknown Opcode '{op_code}' encountered.", current_pc_for_error, instruction)

                if consumed_instruction: self.pc += 1
            
            except RuntimeErrorNS as e_ns: raise e_ns 
            except KeyError as ke: 
                 raise RuntimeErrorNS(f"Name or key not found: {ke}", current_pc_for_error, instruction) from ke
            except IndexError as ie: 
                 raise RuntimeErrorNS(f"Index error (likely stack underflow): {ie}", current_pc_for_error, instruction) from ie
            except TypeError as te: 
                 raise RuntimeErrorNS(f"Type error during operation: {te}", current_pc_for_error, instruction) from te
            except Exception as e_py: 
                raise RuntimeErrorNS(f"Unexpected Python error during execution: {type(e_py).__name__} - {e_py}", current_pc_for_error, instruction) from e_py
        
        return self.stack[-1] if self.stack else None


if __name__ == '__main__':
    # --- ASTNode definition for __main__ (should match one used by Parser if importing ASTs) ---
    class ASTNode: # Minimal ASTNode for test setup
        def __init__(self, node_type, children=None, value=None, **kwargs):
            self.type = node_type; self.value = value
            self.children = children if children is not None else []
            self.__dict__.update(kwargs)
            self.line = kwargs.get('line', -1) 
            self.col = kwargs.get('col', -1)   
            if node_type == 'NeuralOperation' and 'arguments' in kwargs: 
                self.children = kwargs['arguments'] or []


    # --- Minimal Compiler for generating test bytecode ---
    class CompilerStub: # Simplified compiler for test cases
        def __init__(self): self.bytecode = []; self.label_count = 0
        def _generate_label(self, prefix="L"): self.label_count += 1; return f"{prefix}{self.label_count}"
        def compile(self, node): # node is ProgramNode
            self.bytecode = []
            if node and node.children:
                for stmt_node in node.children: self.visit(stmt_node)
            return self.bytecode
        def visit(self, node): # Simplified dispatcher
            method_name = f'visit_{node.type}'; getattr(self, method_name, self.generic_visit)(node)
        def generic_visit(self, node): raise NotImplementedError(f"MiniCompiler cannot visit {node.type}")
        def visit_Assignment(self, node): self.visit(node.children[0]); self.bytecode.append(('STORE_VAR', node.name))
        def visit_NumberLiteral(self, node): self.bytecode.append(('LOAD_CONST', node.value))
        def visit_VectorLiteral(self, node): 
            for el in node.children: self.visit(el)
            self.bytecode.append(('MAKE_VECTOR', len(node.children)))
        def visit_Identifier(self, node): self.bytecode.append(('LOAD_VAR', node.name))
        def visit_Call(self, node):
            for arg in node.children: self.visit(arg)
            self.bytecode.append(('CALL_FUNC', node.callee, len(node.children)))
        def visit_StringLiteral(self, node): # Added for watch_variable test
             self.bytecode.append(('LOAD_CONST', node.value))
        def visit_ListLiteral(self, node): # Added for compute_gradients test (if needed)
            # This is a conceptual MAKE_LIST; actual implementation might differ.
            # For testing, LOAD_CONST with a Python list is simpler if runtime supports it.
            # If not, this would involve loading each element then calling MAKE_LIST.
            pass # Placeholder, as test uses direct LOAD_CONST for list

    # AST Node factories for __main__ tests
    def ProgramNode(stmts): return ASTNode('Program', children=stmts)
    def AssignmentNode(name, expr): return ASTNode('Assignment', children=[expr], name=name)
    def IdentifierNode(name): return ASTNode('Identifier', name=name)
    def NumberLiteralNode(val): return ASTNode('NumberLiteral', value=val)
    def VectorLiteralNode(elements): return ASTNode('VectorLiteral', children=elements)
    def StringLiteralNode(val): return ASTNode('StringLiteral', value=val) # Added
    def ListLiteralNode(elements): return ASTNode('ListLiteral', children=elements) # Added
    def CallNode(callee, args): return ASTNode('Call', children=args, callee=callee)


    compiler = CompilerStub()
    runtime = Runtime() # Runtime now registers stdlib automatically

    print("--- Runtime Test: StdLib gradient flow (placeholder) ---")
    runtime.variables = {} 
    runtime.stack = []

    # 1. tape1 = start_tape();
    ast_start_tape = ProgramNode([AssignmentNode(name='tape1', value_expr=CallNode(callee="start_tape", args=[]))])
    bc_start_tape = compiler.compile(ast_start_tape)
    runtime.execute(bc_start_tape)
    assert 'tape1' in runtime.variables
    assert hasattr(runtime.variables['tape1'], 'watch') 

    # 2. my_param = VECTOR [1.0, 2.0]; 
    ast_param_setup = ProgramNode([
        AssignmentNode(name='my_param', value_expr=VectorLiteralNode([NumberLiteralNode(1.0), NumberLiteralNode(2.0)]))
    ])
    bc_param_setup = compiler.compile(ast_param_setup)
    runtime.execute(bc_param_setup)
    assert 'my_param' in runtime.variables

    # 3. watch_variable(tape1, "my_param");
    ast_watch = ProgramNode([
        CallNode(callee="watch_variable", args=[IdentifierNode('tape1'), StringLiteralNode("my_param")]) 
    ])
    bc_watch = compiler.compile(ast_watch)
    runtime.execute(bc_watch)
    assert "my_param" in runtime.variables['tape1'].watched_variables

    # 4. grads = compute_gradients(tape1, "loss_var", ["my_param"]);
    runtime.variables['loss_var_placeholder'] = NumberLiteralNode(0.5) # Mock target variable for completeness
    
    bc_compute_grads = [
        ('LOAD_VAR', 'tape1'),
        ('LOAD_CONST', 'loss_var_placeholder'), 
        ('LOAD_CONST', ['my_param']),         
        ('CALL_FUNC', 'compute_gradients', 3),
        ('STORE_VAR', 'grads')
    ]
    runtime.execute(bc_compute_grads)
    assert 'grads' in runtime.variables
    assert 'my_param' in runtime.variables['grads']
    assert isinstance(runtime.variables['grads']['my_param'], np.ndarray)

    # 5. updated_params = apply_gradients({"my_param": my_param}, grads);
    params_dict_val = {'my_param': runtime.variables['my_param']} 
    
    bc_apply_grads = [
        ('LOAD_CONST', params_dict_val), 
        ('LOAD_VAR', 'grads'),       
        ('CALL_FUNC', 'apply_gradients', 2),
        ('STORE_VAR', 'updated_p')
    ]
    runtime.execute(bc_apply_grads)
    assert 'updated_p' in runtime.variables
    assert 'my_param' in runtime.variables['updated_p']
    assert np.all(runtime.variables['updated_p']['my_param'] < params_dict_val['my_param'])


    # 6. stop_tape(tape1);
    ast_stop_tape = ProgramNode([CallNode(callee="stop_tape", args=[IdentifierNode('tape1')])])
    bc_stop_tape = compiler.compile(ast_stop_tape)
    runtime.execute(bc_stop_tape)
    assert runtime.variables['tape1'].active == False

    print("Stdlib gradient flow placeholder tests completed.")

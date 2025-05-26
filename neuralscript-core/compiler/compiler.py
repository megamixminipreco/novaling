# neuralscript-core/compiler/compiler.py

class ASTNode: # As provided in the prompt for Subtask 22
    def __init__(self, node_type, children=None, value=None, **kwargs):
        self.type = node_type
        self.value = value
        self.children = children if children is not None else []
        self.__dict__.update(kwargs)
        # Ensure 'token' is processed correctly if present, otherwise use direct line/col from kwargs
        token = kwargs.get('token')
        # Ensure line/col are initialized, defaulting to -1 if not found.
        self.line = -1
        self.col = -1
        if token and isinstance(token, dict):
            self.line = token.get('line', -1)
            self.col = token.get('col', -1)
        else: # Fallback to direct kwargs if token is not a dict or not present
            self.line = kwargs.get('line', -1)
            self.col = kwargs.get('col', -1)


class CompilerError(Exception):
    def __init__(self, message, line=None, col=None):
        full_message = "CompilerError: "
        if line is not None and col is not None and line != -1 and col != -1 : # Only add if valid
            full_message += f"Line {line}:Col {col}: "
        full_message += message
        super().__init__(full_message)
        self.line = line
        self.col = col

class Compiler:
    def __init__(self):
        self.bytecode = []
        self.symbol_table = {} 
        self.label_count = 0
        self.current_func_params = [] # Tracks parameters of the current function being compiled

    def _generate_label(self, prefix="L"):
        self.label_count += 1
        return f"{prefix}{self.label_count}"

    def compile(self, node):
        self.bytecode = []
        self.symbol_table = {} 
        self.current_func_params = [] # Reset for each compilation
        try:
            if node: 
                 self.visit(node)
            else:
                 # Provide a default line/col or indicate it's a general error
                 raise CompilerError("Cannot compile from a None AST root.", line=0, col=0) 
        except CompilerError as e:
            print(e) 
            raise 
        except Exception as e:
            err_line = getattr(node, 'line', 0) if node else 0
            err_col = getattr(node, 'col', 0) if node else 0
            print(f"Unexpected error during compilation: {type(e).__name__} - {e} at around Line {err_line}:Col {err_col}")
            raise CompilerError(f"Unexpected compilation error: {e}", err_line, err_col) from e
        return self.bytecode

    def visit(self, node):
        if node is None: 
            current_context_line = -1 
            current_context_col = -1
            raise CompilerError("Attempted to visit a None AST node. This indicates an AST construction issue.",
                                line=current_context_line, col=current_context_col)

        method_name = f'visit_{node.type}'
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise CompilerError(f"No visit_{node.type} method defined for compiler.", node.line, node.col)

    def visit_Program(self, node):
        for stmt_node in node.children:
            if stmt_node: 
                self.visit(stmt_node)

    def visit_Identifier(self, node):
        if node.name not in self.current_func_params and node.name not in self.symbol_table :
            is_function_ref = node.name in self.symbol_table and \
                              isinstance(self.symbol_table.get(node.name), dict) and \
                              self.symbol_table[node.name].get('type') == 'function'
            if not is_function_ref:
                 raise CompilerError(f"Undefined variable or symbol '{node.name}'.", node.line, node.col)
        
        self.bytecode.append(('LOAD_VAR', node.name))
        return {'type': 'variable', 'name': node.name}

    def visit_Assignment(self, node):
        value_repr = self.visit(node.children[0]) # Compile RHS first
        
        if not self.current_func_params: # Global scope assignment
            self.symbol_table[node.name] = {'type': 'variable', 'line': node.line, 'col': node.col}
        
        self.bytecode.append(('STORE_VAR', node.name))
        return {'type': 'assignment_complete', 'var_name': node.name}


    def visit_NeuralOperation(self, node):
        arg_reprs = []
        # Arity checks
        if node.name == "ATTEND" and len(node.children) != 3:
            raise CompilerError(f"Neural operation '{node.name}' expects 3 arguments, got {len(node.children)}.", 
                              node.line, node.col)
        if node.name == "MERGE" and len(node.children) != 2: 
            raise CompilerError(f"Neural operation '{node.name}' expects 2 arguments, got {len(node.children)}.",
                              node.line, node.col)
            
        for child_idx, arg_node in enumerate(node.children):
            if arg_node is None: 
                raise CompilerError(f"Argument {child_idx+1} for neural op '{node.name}' is None (AST error).", node.line, node.col)
            arg_reprs.append(self.visit(arg_node))

        arg_names_or_values = []
        for r_idx, r in enumerate(arg_reprs):
            child_node = node.children[r_idx] 
            if r is None or 'type' not in r : 
                 raise CompilerError(f"Invalid argument representation for argument {r_idx+1} of neural op '{node.name}'.",
                                   child_node.line, child_node.col)

            if r['type'] == 'variable': arg_names_or_values.append(r['name'])
            elif r['type'] == 'literal': arg_names_or_values.append(r['value'])
            else: 
                 raise CompilerError(f"Unexpected argument representation type '{r.get('type')}' for neural op '{node.name}'.",
                                   child_node.line, child_node.col)

        self.bytecode.append(('NEURAL_OP', node.name, arg_names_or_values))
        return {'type': 'neural_op_result', 'op_name': node.name}

    def visit_FunctionDef(self, node):
        if node.name in self.symbol_table and self.symbol_table[node.name].get('type') == 'function':
             original_def_info = self.symbol_table[node.name]
             orig_line = original_def_info.get('line', 'N/A')
             orig_col = original_def_info.get('col', 'N/A')
             raise CompilerError(f"Function '{node.name}' redefined. Original definition at Line {orig_line}:Col {orig_col}.", 
                               node.line, node.col)
        
        param_names = []
        # Ensure node.params exists and is iterable, as TestASTNode might not always set it from kwargs if empty
        node_params = getattr(node, 'params', []) 
        if node_params: # Check if node_params is not None and is iterable
            for p_node in node_params: 
                if not isinstance(p_node, ASTNode) and not isinstance(p_node, TestASTNode): # Added TestASTNode for __main__
                    raise CompilerError(f"Invalid parameter type for function '{node.name}'. Expected ASTNode/TestASTNode.",
                                      node.line, node.col)
                if p_node.type != 'Identifier':
                    raise CompilerError(f"Invalid parameter definition for function '{node.name}'. Expected Identifier type.",
                                      p_node.line, p_node.col) 
                param_names.append(p_node.name)
        
        self.symbol_table[node.name] = {'type': 'function', 'params': param_names, 'line': node.line, 'col': node.col}
        
        old_params = self.current_func_params
        self.current_func_params = param_names 

        func_label_start = self._generate_label(f"FUNC_{node.name}_START")
        self.bytecode.append(('LABEL', func_label_start))
        self.bytecode.append(('FUNC_BEGIN', node.name, self.current_func_params))
        
        # FunctionDefNode body handling (from parser or __main__ TestASTNode)
        # Parser wraps body in ProgramNode, so children[0] is ProgramNode.
        # TestASTNode in __main__ passes body statements directly as children.
        body_stmts_to_visit = []
        if node.children and isinstance(node.children[0], (ASTNode, TestASTNode)) and node.children[0].type == 'Program':
            body_stmts_to_visit = node.children[0].children # It's a ProgramNode, visit its children
        elif node.children: # Assuming direct list of statements (from TestASTNode in __main__)
            body_stmts_to_visit = node.children
        
        for stmt in body_stmts_to_visit: 
            if stmt: self.visit(stmt)
        
        if not self.bytecode or self.bytecode[-1][0] not in ['RETURN_VALUE', 'RETURN_VOID']:
             self.bytecode.append(('RETURN_VOID',))

        func_label_end = self._generate_label(f"FUNC_{node.name}_END") 
        self.bytecode.append(('LABEL', func_label_end))
        
        self.current_func_params = old_params 
        return {'type': 'function_definition', 'name': node.name}


    def visit_Call(self, node):
        is_known_user_function = node.callee in self.symbol_table and \
                                 isinstance(self.symbol_table[node.callee], dict) and \
                                 self.symbol_table[node.callee].get('type') == 'function'

        if is_known_user_function:
            func_info = self.symbol_table[node.callee]
            expected_param_count = len(func_info['params'])
            actual_arg_count = len(node.children) 
            if actual_arg_count != expected_param_count:
                raise CompilerError(
                    f"Function '{node.callee}' defined at Line {func_info['line']}:Col {func_info['col']} "
                    f"expects {expected_param_count} arguments, but {actual_arg_count} were provided.",
                    node.line, node.col 
                )
        
        for arg_node in node.children: 
            self.visit(arg_node) 

        self.bytecode.append(('CALL_FUNC', node.callee, len(node.children)))
        return {'type': 'call_result', 'func_name': node.callee}

    def visit_NumberLiteral(self, node): 
        self.bytecode.append(('LOAD_CONST', node.value))
        return {'type': 'literal', 'value': node.value}
    def visit_StringLiteral(self, node):
        self.bytecode.append(('LOAD_CONST', node.value))
        return {'type': 'literal', 'value': node.value}
    def visit_BooleanLiteral(self, node):
        self.bytecode.append(('LOAD_CONST', node.value))
        return {'type': 'literal', 'value': node.value}
    def visit_VectorLiteral(self, node):
        for el_node in node.children: self.visit(el_node)
        self.bytecode.append(('MAKE_VECTOR', len(node.children)))
        return {'type': 'vector', 'size': len(node.children)}
    def visit_BinaryOperation(self, node):
        self.visit(node.children[0]) 
        self.visit(node.children[1]) 
        self.bytecode.append(('CALL_OP', node.operator, 2))
        return {'type': 'binary_op_result', 'op': node.operator}
    def visit_Return(self, node):
        if node.children and node.children[0]: 
            self.visit(node.children[0])
            self.bytecode.append(('RETURN_VALUE',))
        else: 
            self.bytecode.append(('RETURN_VOID',))
        return {'type': 'return_statement'}
    def visit_If(self, node): 
        self.visit(node.children[0]) # Condition
        else_label = self._generate_label("IF_ELSE")
        end_if_label = self._generate_label("IF_END")
        actual_jump_target = else_label if node.has_else else end_if_label
        self.bytecode.append(('JUMP_IF_FALSE', actual_jump_target))
        
        # Then branch (node.children[1] is a ProgramNode or list of statements for TestASTNode)
        then_branch_node = node.children[1]
        if then_branch_node:
            if then_branch_node.type == 'Program': self.visit(then_branch_node) 
            else: self.visit(then_branch_node) # Assume it's a single statement node if not ProgramNode
        else:
            raise CompilerError("Malformed 'then' branch in IfNode: branch is None.", node.line, node.col)


        if node.has_else: self.bytecode.append(('JUMP', end_if_label))
        self.bytecode.append(('LABEL', else_label))
        if node.has_else:
            if len(node.children) > 2:
                else_branch_node = node.children[2]
                if else_branch_node:
                    if else_branch_node.type == 'Program': self.visit(else_branch_node)
                    else: self.visit(else_branch_node) # Assume single statement node
                else:
                     raise CompilerError("Malformed 'else' branch in IfNode: branch is None though has_else is True.", node.line, node.col)
            else: # has_else is True but no else branch node found
                 raise CompilerError("Inconsistent IfNode: has_else is True but no else branch child found.", node.line, node.col)

        self.bytecode.append(('LABEL', end_if_label))
        return {'type': 'if_statement'}


if __name__ == '__main__':
    # Using a local TestASTNode for __main__ to avoid circular deps or complex imports for testing
    class TestASTNode: # Renamed to avoid conflict with top-level ASTNode
        def __init__(self, node_type, children=None, value=None, line=-1, col=-1, **kwargs):
            self.type = node_type
            self.value = value
            self.line = line if line != -1 else kwargs.get('token', {}).get('line', -1) # Prioritize direct line/col
            self.col = col if col != -1 else kwargs.get('token', {}).get('col', -1)
            
            self.children = []
            if children: 
                for child_data in children:
                    if isinstance(child_data, dict):
                        self.children.append(TestASTNode(**child_data))
                    else: 
                        self.children.append(child_data)
            
            self.__dict__.update(kwargs)
            # Special handling for params in FunctionDefNode for tests
            if 'params' in kwargs and kwargs['params'] is not None:
                self.params = [TestASTNode(**p_data) if isinstance(p_data, dict) else p_data for p_data in kwargs['params']]
            
            # For NeuralOperationNode from prompt's test case
            if node_type == 'NeuralOperation' and 'arguments' in kwargs:
                self.children = [] # Override children if 'arguments' is present
                for arg_data in kwargs['arguments']:
                     if isinstance(arg_data, dict): self.children.append(TestASTNode(**arg_data))
                     else: self.children.append(arg_data)


    compiler_test = Compiler()

    print("--- Testing Compiler: Undefined Variable ---")
    ast_undef_var = TestASTNode('Program', line=1,col=1,children=[
        TestASTNode('Assignment', line=1,col=1,name='x', children=[
            TestASTNode('BinaryOperation', line=1,col=5,operator='+', children=[
                TestASTNode('Identifier', line=1,col=5,name='y'), # y is undefined
                TestASTNode('NumberLiteral', line=1,col=9,value=10.0)
            ])
        ])
    ])
    try:
        compiler_test.compile(ast_undef_var)
    except CompilerError as e:
        print(f"Caught expected error: {e}")
    except Exception as e:
        print(f"Caught UNEXPECTED error: {type(e).__name__} - {e}")


    print("\n--- Testing Compiler: Function Arity Mismatch ---")
    param_a_node_data = {'type':'Identifier', 'name':'a', 'line':1, 'col':12}
    ast_arity = TestASTNode('Program', line=1,col=1,children=[
        TestASTNode('FunctionDef', name='my_func', params=[param_a_node_data], line=1,col=1,children=[ # Body
            TestASTNode('Return', line=1,col':17,children=[{'type':'Identifier','name':'a','line':1,'col':24}])
        ]),
        TestASTNode('Assignment', name='res', line':2,'col':1,children=[
            TestASTNode('Call', callee='my_func', line':2,'col':7,children=[ 
                {'type':'NumberLiteral','value':10.0, 'line':2,'col':17},
                {'type':'NumberLiteral','value':20.0, 'line':2,'col':21} 
            ])
        ])
    ])
    try:
        compiler_test = Compiler() 
        compiler_test.compile(ast_arity)
    except CompilerError as e:
        print(f"Caught expected error: {e}")
    except Exception as e:
        print(f"Caught UNEXPECTED error: {type(e).__name__} - {e}")

    print("\n--- Testing Compiler: Neural Op Arity Mismatch ---")
    ast_attend_arity = TestASTNode('Program', line=1,col=1,children=[
        TestASTNode('Assignment', name='output', line=1,col':1,children=[
            TestASTNode('NeuralOperation', name='ATTEND', line=1,col':10, arguments=[ 
                {'type':'Identifier','name':'q', 'line':1,'col':17} 
            ])
        ])
    ])
    try:
        compiler_test = Compiler()
        compiler_test.symbol_table['q'] = {'type': 'variable', 'line':0, 'col':0} 
        compiler_test.compile(ast_attend_arity)
    except CompilerError as e:
        print(f"Caught expected error: {e}")
    except Exception as e:
        print(f"Caught UNEXPECTED error: {type(e).__name__} - {e}")

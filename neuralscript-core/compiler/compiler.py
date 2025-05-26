# neuralscript-core/compiler/compiler.py

# For type hinting and ASTNode structure
# Ideally, ASTNode would be in a shared location. For now, define minimally or assume.
class ASTNode: # Minimal definition for type hint reference
    def __init__(self, node_type, children=None, value=None, **kwargs):
        self.type = node_type
        self.value = value
        # Handle 'arguments' kwarg for NeuralOperationNode in __main__ test
        # This ensures that if 'arguments' is passed as a kwarg (like in the __main__ test case factory for NeuralOperationNode),
        # it's correctly assigned to self.children.
        if node_type == 'NeuralOperation' and 'arguments' in kwargs:
            self.children = kwargs['arguments'] if kwargs['arguments'] is not None else []
        elif children is not None:
            self.children = children
        else:
            self.children = []
        self.__dict__.update(kwargs)

class Compiler:
    def __init__(self):
        self.bytecode = []
        self.symbol_table = {} # Simple symbol table for now
        self.label_count = 0   # For generating unique labels for jumps

    def _generate_label(self, prefix="L"):
        self.label_count += 1
        return f"{prefix}{self.label_count}"

    def compile(self, node):
        self.bytecode = [] # Reset for new compilation
        self.label_count = 0 # Reset label count for new compilation
        if node is None:
            return []
        self.visit(node)
        return self.bytecode

    def visit(self, node):
        if node is None: # Important for optional parts of AST
            return None
        method_name = f'visit_{node.type}'
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise Exception(f'No visit_{node.type} method defined for compiler, value: {node.value}')

    def visit_Program(self, node):
        for stmt in node.children:
            self.visit(stmt)

    def visit_NumberLiteral(self, node):
        # For expressions, literals might load their value onto a conceptual stack/temp var
        # Or return their representation for direct use by parent nodes.
        # For now, let's assume expressions will generate bytecode that leaves result in a known place (e.g. last_expr_result)
        # This is a simplification. A stack machine is more common.
        self.bytecode.append(('LOAD_CONST', node.value))
        # Return a representation that can be used by parent nodes
        return {'type': 'literal', 'value': node.value}


    def visit_StringLiteral(self, node):
        self.bytecode.append(('LOAD_CONST', node.value))
        return {'type': 'literal', 'value': node.value}


    def visit_BooleanLiteral(self, node):
        self.bytecode.append(('LOAD_CONST', node.value))
        return {'type': 'literal', 'value': node.value}

    def visit_Identifier(self, node):
        # When an identifier is visited in an expression context, it means its value should be loaded.
        self.bytecode.append(('LOAD_VAR', node.name))
        return {'type': 'variable', 'name': node.name}


    def visit_VectorLiteral(self, node):
        element_reprs = []
        for element_node in node.children:
            repr_val = self.visit(element_node)
            # If visit returns a dict like {'type': 'literal', 'value': ...} or {'type': 'variable', 'name': ...}
            # we might need to ensure elements are loaded onto stack before MAKE_VECTOR
            # For now, assume visit for literals/vars appends LOAD_CONST/LOAD_VAR
            element_reprs.append(repr_val) # Collect representations

        # After all elements are processed (and their LOAD opcodes generated):
        self.bytecode.append(('MAKE_VECTOR', len(node.children)))
        return {'type': 'vector', 'size': len(node.children)}


    def visit_Assignment(self, node):
        # node.name is the variable name (string)
        # node.children[0] is the expression ASTNode to be assigned
        
        # Compile the expression on the right-hand side
        # This should leave the result of the expression in a known state
        # (e.g., last item on bytecode stack, or a specific temp variable if not using stack machine)
        value_repr = self.visit(node.children[0])

        # After the expression is compiled and its value is conceptually available:
        # The STORE_VAR instruction will tell the runtime to take that available value
        # and store it in the variable.
        self.bytecode.append(('STORE_VAR', node.name))
        # No specific return needed for statements usually, or return a status.

    def visit_NeuralOperation(self, node):
        # node.name is the operation name (e.g., "ATTEND")
        # node.children are the argument ASTNodes
        # (populated correctly by ASTNode.__init__ if 'arguments' kwarg was used by factory)
        arg_reprs = []
        for arg_node in node.children:
            repr_val = self.visit(arg_node) # Compiles each argument
            # This will append LOAD_VAR or LOAD_CONST for each arg.
            arg_reprs.append(repr_val) # Collect representations if needed by NEURAL_OP bytecode

        # The NEURAL_OP bytecode instruction itself might just take the number of arguments,
        # assuming they are on the stack. Or, it could take the names/representations.
        # Let's use the names/representations for clarity in this non-stack-machine version.
        # The runtime will need to resolve these.
        arg_names_or_values = []
        for r in arg_reprs:
            if r is None: # Should not happen if args compiled correctly
                 raise ValueError(f"Argument to {node.name} compiled to None.")
            if r.get('type') == 'variable':
                arg_names_or_values.append(r['name'])
            elif r.get('type') == 'literal':
                arg_names_or_values.append(r['value'])
            # Add more types if necessary (e.g. if an expression result is stored in a temp var)

        self.bytecode.append(('NEURAL_OP', node.name, arg_names_or_values))
        # This operation's result is now conceptually the "last computed value"
        return {'type': 'neural_op_result', 'op_name': node.name}


    def visit_BinaryOperation(self, node):
        # node.children[0] is left operand, node.children[1] is right
        # node.operator is the operator string like '+', '=='
        
        # Compile left operand
        left_repr = self.visit(node.children[0])
        # Compile right operand
        right_repr = self.visit(node.children[1])

        # The values for left and right are now on the conceptual stack (due to LOAD_VAR/LOAD_CONST)
        # The CALL_OP will pop them, perform operation, and push result.
        self.bytecode.append(('CALL_OP', node.operator, 2)) # 2 arguments
        return {'type': 'binary_op_result', 'op': node.operator}

    def visit_FunctionDef(self, node):
        # node.name, node.params (list of IdentifierNodes), node.children (body stmts), node.return_type
        func_label_start = self._generate_label(f"FUNC_{node.name}_START")
        func_label_end = self._generate_label(f"FUNC_{node.name}_END")

        # Instruction to jump over function body in main code flow
        # For now, let's assume functions are hoisted or handled by runtime separately.
        # A simple scheme: emit function bytecode separately or make runtime aware of FUNC_BEGIN/END.
        
        self.bytecode.append(('LABEL', func_label_start))
        self.bytecode.append(('FUNC_BEGIN', node.name, [p.name for p in node.params])) # Store param names

        # TODO: Handle symbol table for function scope if implementing nested scopes
        # For now, assume parameters are handled by runtime based on FUNC_BEGIN

        for stmt in node.children: # node.children is the body
            self.visit(stmt)

        self.bytecode.append(('LABEL', func_label_end))
        # Implicit return if no explicit return at the end of function (runtime might handle this)
        return {'type': 'function_definition', 'name': node.name}

    def visit_Return(self, node):
        # node.children is a list. For "RETURN expr;", children=[expr_node]. For "RETURN;", children=[]
        if node.children: # If there's a child list, it implies an expression (even if it's None for RETURN None)
            value_repr = self.visit(node.children[0]) # Compile the expression to be returned
            # The result of this expression is now the "last computed value"
            self.bytecode.append(('RETURN_VALUE',)) # Assumes value is on stack or last_expr_result
        else: # RETURN (no value, children list is empty)
            self.bytecode.append(('RETURN_VOID',))
        return {'type': 'return_statement'}


    def visit_Call(self, node):
        # node.callee (string), node.children (list of argument ASTNodes)
        arg_reprs = []
        for arg_node in node.children:
            arg_repr = self.visit(arg_node) # Compiles each argument
            # LOAD_VAR/LOAD_CONST for each arg would have been emitted
            arg_reprs.append(arg_repr)

        # The CALL bytecode instruction will tell the runtime to find the function
        # and pass the arguments (which are assumed to be on the stack or passed via registers).
        self.bytecode.append(('CALL_FUNC', node.callee, len(node.children)))
        return {'type': 'call_result', 'func_name': node.callee}


    def visit_If(self, node):
        # node.children[0] = condition
        # node.children[1] = then_branch (ProgramNode)
        # node.children[2] = else_branch (ProgramNode), if present (check node.has_else)

        # Compile condition
        # The result of the condition (boolean) should be the "last computed value"
        self.visit(node.children[0]) 
        
        else_label = self._generate_label("IF_ELSE")
        end_if_label = self._generate_label("IF_END")

        # If condition is false, jump to else_label. If no else, jump to end_if_label.
        actual_jump_target_for_false = else_label if node.has_else else end_if_label
        self.bytecode.append(('JUMP_IF_FALSE', actual_jump_target_for_false))

        # Compile then branch
        self.visit(node.children[1]) # This is a ProgramNode for the 'then' block
        if node.has_else: # If there's an else block, jump over it from end of 'then' block
            self.bytecode.append(('JUMP', end_if_label))
        
        self.bytecode.append(('LABEL', else_label)) # Else branch starts here (or end_if if no else)
        if node.has_else:
            self.visit(node.children[2]) # This is a ProgramNode for the 'else' block
        
        self.bytecode.append(('LABEL', end_if_label)) # End of IF statement
        return {'type': 'if_statement'}


# --- Main for testing ---
if __name__ == '__main__':
    # We need a way to get an AST. For testing, manually create one or use Parser.
    # Minimal ASTNode and other node types for manual creation:
    # (Re-define necessary Node classes from Parser for standalone testing if not importing)
    class ASTNode: # Redefined for __main__ test purposes
        def __init__(self, node_type, children=None, value=None, **kwargs):
            self.type = node_type; self.value = value; 
            # For NeuralOperationNode test case in __main__ where 'arguments' kwarg is used
            if node_type == 'NeuralOperation' and 'arguments' in kwargs:
                 self.children = kwargs['arguments'] or []
            else: # For other nodes or if NeuralOp factory passes args as children directly
                 self.children = children or []
            self.__dict__.update(kwargs)
        # Basic repr for debugging in __main__
        def __repr__(self, level=0): # Simplified __repr__ for __main__
            ret = "\t" * level + f"ASTNode(type='{self.type}'"
            if self.value is not None: ret += f", value='{self.value}'"
            for k_attr in ['name', 'operator', 'callee', 'params', 'return_type', 'has_else', 'arguments']:
                 if hasattr(self, k_attr) and getattr(self, k_attr) is not None:
                    val_attr = getattr(self, k_attr)
                    if k_attr in ['params', 'arguments'] and isinstance(val_attr, list):
                         list_repr_val = "[" + ", ".join([str(item.value if hasattr(item,'value') else item.name if hasattr(item,'name') else item.type) for item in val_attr if isinstance(item, ASTNode)]) + "]"
                         ret += f", {k_attr}={list_repr_val}"
                    elif not isinstance(val_attr, list):
                        ret += f", {k_attr}='{val_attr}'"
            if self.children and not (self.type == 'NeuralOperation' and hasattr(self, 'arguments') and self.children == getattr(self, 'arguments', None)):
                ret += ", children=[\n"
                for child in self.children:
                    if child: ret += child.__repr__(level + 1) + ",\n"
                    else: ret += "\t" * (level + 1) + "None,\n"
                ret += "\t" * level + "]"
            ret += ")"
            return ret

    def ProgramNode(stmts): return ASTNode('Program', children=stmts)
    def AssignmentNode(name, expr): return ASTNode('Assignment', children=[expr], name=name)
    def IdentifierNode(name): return ASTNode('Identifier', name=name)
    def NumberLiteralNode(val): return ASTNode('NumberLiteral', value=val)
    def VectorLiteralNode(elements): return ASTNode('VectorLiteral', children=elements)
    # NeuralOperationNode factory for test: 'arguments' kwarg used for children compatibility with top-level ASTNode
    # The factory parameter 'args' here should match the kwarg 'arguments' used in the call site for clarity.
    def NeuralOperationNode(name, arguments): return ASTNode('NeuralOperation', name=name, arguments=arguments) 
    def BinaryOperationNode(left, op, right): return ASTNode('BinaryOperation', children=[left, right], operator=op)
    # IfNode factory for test: then/else branches are ProgramNodes
    def IfNode(condition, then_branch, else_branch=None): 
        children = [condition, then_branch]
        if else_branch: children.append(else_branch)
        return ASTNode('If', children=children, has_else=else_branch is not None)
    def BooleanLiteralNode(val): return ASTNode("BooleanLiteral", value=val)
    # FunctionDefNode factory for test: 'body' (list of statements) becomes children
    def FunctionDefNode(name, params, body, rt=None): return ASTNode("FunctionDef", children=body, name=name, params=params, return_type=rt)
    # ReturnNode factory in the prompt:
    # ReturnNode(expr=None) -> children=[None] -> RETURN_VALUE (runtime handles None)
    # ReturnNode() -> children=[] -> RETURN_VOID
    def ReturnNode(expr=None): 
        # This factory differs from prompt's test case (test case uses expr_tuple)
        # Making it consistent with how it's likely used and how visit_Return is structured
        if 'expr' in locals() and expr is not None: # Explicitly RETURN expr or RETURN None
             return ASTNode("Return", children=[expr])
        return ASTNode("Return", children=[]) # RETURN; (void)

    def CallNode(callee, args): return ASTNode("Call", children=args, callee=callee)


    compiler = Compiler()

    # Test Case 1: Simple Assignment: x = 10 + 5
    ast1 = ProgramNode([
        AssignmentNode(name='x', value_expr=BinaryOperationNode(
            NumberLiteralNode(10), '+', NumberLiteralNode(5)
        ))
    ])
    bytecode1 = compiler.compile(ast1)
    print("--- AST 1 (x = 10 + 5) ---")
    # print(ast1) # Needs full __repr__ for ASTNode
    print("--- Bytecode 1 ---")
    for instr in bytecode1: print(instr)
    # Expected:
    # ('LOAD_CONST', 10)
    # ('LOAD_CONST', 5)
    # ('CALL_OP', '+', 2)
    # ('STORE_VAR', 'x')

    # Test Case 2: Neural Operation: res = ATTEND q, k, v
    ast2 = ProgramNode([
        AssignmentNode(name='res', value_expr=NeuralOperationNode(
            name='ATTEND',
            arguments=[IdentifierNode('q'), IdentifierNode('k'), IdentifierNode('v')]
        ))
    ])
    bytecode2 = compiler.compile(ast2)
    print("\n--- AST 2 (res = ATTEND q, k, v) ---")
    print("--- Bytecode 2 ---")
    for instr in bytecode2: print(instr)
    # Expected:
    # ('LOAD_VAR', 'q')
    # ('LOAD_VAR', 'k')
    # ('LOAD_VAR', 'v')
    # ('NEURAL_OP', 'ATTEND', ['q', 'k', 'v']) # or indices if using stack machine model
    # ('STORE_VAR', 'res')

    # Test Case 3: If statement
    ast3 = ProgramNode([
        IfNode(
            condition=BinaryOperationNode(IdentifierNode('x'), '==', NumberLiteralNode(10)),
            then_branch=ProgramNode([AssignmentNode('y', NumberLiteralNode(1))]),
            else_branch=ProgramNode([AssignmentNode('y', NumberLiteralNode(0))])
        )
    ])
    bytecode3 = compiler.compile(ast3)
    print("\n--- AST 3 (IF x == 10 THEN y = 1 ELSE y = 0) ---")
    print("--- Bytecode 3 ---")
    for instr in bytecode3: print(instr)
    # Expected structure:
    # ('LOAD_VAR', 'x')
    # ('LOAD_CONST', 10)
    # ('CALL_OP', '==', 2)
    # ('JUMP_IF_FALSE', 'L1_IF_ELSE') or similar
    #   -- then branch --
    # ('LOAD_CONST', 1)
    # ('STORE_VAR', 'y')
    # ('JUMP', 'L2_IF_END')
    # ('LABEL', 'L1_IF_ELSE')
    #   -- else branch --
    # ('LOAD_CONST', 0)
    # ('STORE_VAR', 'y')
    # ('LABEL', 'L2_IF_END')
    
    # Test Case 4: Function Definition and Call
    ast4 = ProgramNode([
        FunctionDefNode(name="my_func", params=[IdentifierNode("a")], body=[
            # Using updated ReturnNode factory: ReturnNode(expr)
            ReturnNode(BinaryOperationNode(IdentifierNode("a"), "+", NumberLiteralNode(1))) 
        ]),
        AssignmentNode(name="result", value_expr=CallNode(callee="my_func", args=[NumberLiteralNode(5)]))
    ])
    bytecode4 = compiler.compile(ast4)
    print("\n--- AST 4 (Function Def & Call) ---")
    print("--- Bytecode 4 ---")
    for instr in bytecode4: print(instr)
    # Expected:
    # ('LABEL', 'FUNC_my_func_START')
    # ('FUNC_BEGIN', 'my_func', ['a'])
    # ('LOAD_VAR', 'a')
    # ('LOAD_CONST', 1)
    # ('CALL_OP', '+', 2)
    # ('RETURN_VALUE',)
    # ('LABEL', 'FUNC_my_func_END')
    # ('LOAD_CONST', 5)
    # ('CALL_FUNC', 'my_func', 1)
    # ('STORE_VAR', 'result')

```
This sets up a basic compiler structure.
Key simplifications for this initial step:
- The "bytecode" is a list of tuples, not actual binary code.
- It's not strictly a stack machine in its current form for `NEURAL_OP` and `STORE_VAR`; it's a bit more direct, assuming the runtime can handle resolving variable names passed as arguments or results of previous operations. A true stack machine would involve more PUSH/POP and specific opcodes for variable storage/retrieval relative to stack positions.
- Symbol table usage is minimal; full scope management (local vs. global, function scopes) is a more advanced topic.
- Error handling is basic.
The `visit_` methods for literals and identifiers now return a representation that the parent node (like `visit_Assignment` or `visit_NeuralOperation`) can use to understand what was loaded. This is a step towards managing expression results.
The `__main__` block has test cases for assignment, neural operations, if statements, and function definition/calls.

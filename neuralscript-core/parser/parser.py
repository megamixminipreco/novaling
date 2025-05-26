# neuralscript-core/parser/parser.py
# (Assuming ASTNode is already defined as in the previous version, or defined here)

class ASTNode:
    def __init__(self, node_type, children=None, value=None, **kwargs):
        self.type = node_type
        self.value = value # For literals, identifiers, operator type
        self.children = children if children is not None else []
        self.__dict__.update(kwargs) # For additional named attributes

    def __repr__(self, level=0):
        ret = "\t" * level + f"ASTNode(type='{self.type}'"
        if self.value is not None:
            ret += f", value='{self.value}'"
        
        # Print other specific attributes common in new nodes
        for attr in ['name', 'operator', 'callee', 'params', 'return_type']:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                # For params, which could be a list of ASTNodes, handle differently
                if attr == 'params' and isinstance(getattr(self, attr), list):
                    param_repr = "[" + ", ".join([p.__repr__(level+1) for p in getattr(self, attr)]) + "]"
                    ret += f", {attr}={param_repr}"
                else:
                    ret += f", {attr}='{getattr(self, attr)}'"

        if self.children:
            ret += ", children=[\n"
            for child in self.children:
                if child: # Check if child is not None
                    ret += child.__repr__(level + 1) + ",\n"
                else:
                    ret += "\t" * (level + 1) + "None,\n" # Represent None child
            ret += "\t" * level + "]"
        ret += ")"
        return ret

# --- New/Updated AST Node type aliases for clarity (optional, but good practice) ---
# These could also be actual classes inheriting from ASTNode if more specific logic is needed

def ProgramNode(statements):
    return ASTNode(node_type='Program', children=statements)

def AssignmentNode(target, value_expr):
    # Ensure target is an ASTNode if it's an identifier, or allow string for simplicity
    # For now, assume target is an identifier string, value_expr is an ASTNode
    return ASTNode(node_type='Assignment', name=target, children=[value_expr])

def IdentifierNode(name):
    return ASTNode(node_type='Identifier', name=name)

def NumberLiteralNode(value):
    return ASTNode(node_type='NumberLiteral', value=value)

def StringLiteralNode(value):
    return ASTNode(node_type='StringLiteral', value=value)

def BooleanLiteralNode(value): # value is Python bool True/False
    return ASTNode(node_type='BooleanLiteral', value=value)

def VectorLiteralNode(elements):
    return ASTNode(node_type='VectorLiteral', children=elements)

def MatrixLiteralNode(rows): # Assuming rows is a list of VectorLiteralNode or similar
    return ASTNode(node_type='MatrixLiteral', children=rows)
    
def BinaryOperationNode(left, op, right):
    return ASTNode(node_type='BinaryOperation', operator=op, children=[left, right])

def NeuralOperationNode(operation_name, arguments):
    # operation_name is a string like "ATTEND", arguments is a list of ASTNodes
    return ASTNode(node_type='NeuralOperation', name=operation_name, children=arguments)

def FunctionDefNode(name, params, body, return_type=None):
    # name: string, params: list of IdentifierNodes (or strings), body: list of statement ASTNodes
    return ASTNode(node_type='FunctionDef', name=name, params=params, return_type=return_type, children=body)

def ReturnNode(expression):
    return ASTNode(node_type='Return', children=[expression] if expression else [])

def IfNode(condition, then_branch, else_branch=None):
    children = [condition, then_branch]
    if else_branch:
        children.append(else_branch)
    return ASTNode(node_type='If', children=children, has_else=else_branch is not None)

def CallNode(callee_name, arguments): # callee_name as string for simplicity
    return ASTNode(node_type='Call', callee=callee_name, children=arguments)


class Parser:
    def __init__(self, tokens):
        self.tokens = [token for token in tokens if token[0] not in ['WHITESPACE', 'COMMENT']] # Filter out
        self.pos = 0
        self.current_token = self.tokens[self.pos] if self.tokens else (None, None)

    def _advance(self):
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current_token = self.tokens[self.pos]
        else:
            self.current_token = (None, None) # (Type, Value)

    def _peek(self):
        if self.pos + 1 < len(self.tokens):
            return self.tokens[self.pos + 1]
        return (None, None)

    def _expect(self, token_type, token_value=None):
        ttype, tval = self.current_token
        if ttype == token_type:
            if token_value is None or tval == token_value:
                token = self.current_token
                self._advance()
                return token
            else:
                raise SyntaxError(f"Expected token value {token_value} for type {token_type}, got {tval} at pos {self.pos}")
        raise SyntaxError(f"Expected token type {token_type}, got {ttype} at pos {self.pos} (current token: {self.current_token})")

    def parse_atom(self):
        ttype, tval = self.current_token
        if ttype == 'NUMBER':
            self._advance()
            return NumberLiteralNode(float(tval)) # Or int(tval) if no decimals
        elif ttype == 'STRING':
            self._advance()
            return StringLiteralNode(tval[1:-1]) # Remove quotes
        elif ttype == 'TRUE':
            self._advance()
            return BooleanLiteralNode(True)
        elif ttype == 'FALSE':
            self._advance()
            return BooleanLiteralNode(False)
        elif ttype == 'IDENTIFIER':
            name = tval
            self._advance()
            if self.current_token[0] == 'LPAREN': # Function call
                self._advance() # Consume LPAREN
                args = []
                if self.current_token[0] != 'RPAREN': # Check if there are arguments
                    while True:
                        args.append(self.parse_expression())
                        if self.current_token[0] == 'COMMA':
                            self._advance()
                        elif self.current_token[0] == 'RPAREN':
                            break
                        else:
                            raise SyntaxError(f"Expected COMMA or RPAREN in argument list, got {self.current_token[0]}")
                self._expect('RPAREN')
                return CallNode(callee_name=name, arguments=args)
            return IdentifierNode(name)
        elif ttype == 'LPAREN':
            self._advance()
            expr = self.parse_expression()
            self._expect('RPAREN')
            return expr
        elif ttype == 'VECTOR': # VECTOR [el1, el2]
            self._advance()
            self._expect('LBRACKET')
            elements = []
            if self.current_token[0] != 'RBRACKET': # Check if there are elements
                while True:
                    elements.append(self.parse_expression())
                    if self.current_token[0] == 'COMMA':
                        self._advance()
                    elif self.current_token[0] == 'RBRACKET':
                        break
                    else:
                        raise SyntaxError(f"Expected COMMA or RBRACKET in vector literal, got {self.current_token[0]}")
            self._expect('RBRACKET')
            return VectorLiteralNode(elements)
        elif ttype in ['ATTEND', 'SAMPLE', 'PROPAGATE', 'MERGE', 'EVOLVE']: # Neural Ops
            op_name = tval
            self._advance()
            args = []
            # Neural ops take comma-separated expressions as args
            if self.current_token[0] not in ['SEMICOLON', None, 'RPAREN', 'RBRACE', 'LBRACE']:
                while True:
                    args.append(self.parse_expression()) # Use parse_expression for args
                    if self.current_token[0] == 'COMMA':
                        self._advance()
                    else:
                        break # No more arguments if no comma
            return NeuralOperationNode(operation_name=op_name, arguments=args)

        raise SyntaxError(f"Unexpected token {self.current_token} in parse_atom at pos {self.pos}")

    def parse_term(self): # Handles * and /
        node = self.parse_atom()
        while self.current_token[0] in ('MULTIPLY', 'DIVIDE'):
            op_ttype, op_tval = self.current_token
            self._advance()
            right = self.parse_atom()
            node = BinaryOperationNode(node, op_tval, right)
        return node

    def parse_expression(self): # Handles + and - (and comparisons for simplicity here)
        node = self.parse_term()
        while self.current_token[0] in ('PLUS', 'MINUS', 'EQ', 'NEQ', 'LT', 'GT', 'LTE', 'GTE'):
            op_ttype, op_tval = self.current_token
            self._advance()
            right = self.parse_term()
            node = BinaryOperationNode(node, op_tval, right)
        return node

    def parse_statement(self):
        ttype, tval = self.current_token
        if ttype == 'DEF':
            return self.parse_function_definition()
        elif ttype == 'RETURN':
            self._advance()
            expr = None
            if self.current_token[0] != 'SEMICOLON': # RETURN value;
                expr = self.parse_expression()
            self._expect('SEMICOLON')
            return ReturnNode(expr)
        elif ttype == 'IF':
            return self.parse_if_statement()
        # Assignment or lone expression (e.g. function call or neural op)
        elif ttype == 'IDENTIFIER' and self._peek()[0] == 'ASSIGN':
            identifier_name = self._expect('IDENTIFIER')[1]
            self._expect('ASSIGN')
            value_expr = self.parse_expression()
            self._expect('SEMICOLON')
            return AssignmentNode(target=identifier_name, value_expr=value_expr)
        else: 
            # Could be a function call statement, neural op statement, etc.
            expr = self.parse_expression() 
            if self.current_token[0] == 'SEMICOLON':
                 self._advance() # Consume semicolon after expression statement
            return expr


    def parse_function_definition(self):
        self._expect('DEF')
        name = self._expect('IDENTIFIER')[1]
        self._expect('LPAREN')
        params = []
        if self.current_token[0] != 'RPAREN':
            while True:
                param_name = self._expect('IDENTIFIER')[1]
                param_node = IdentifierNode(param_name) # Store as IdentifierNode
                # Optional type hint: param_name : TYPE
                if self.current_token[0] == 'COLON':
                    self._advance() # Consume COLON
                    type_name = self._expect('IDENTIFIER')[1] 
                    param_node.type_hint = type_name # Add type_hint to IdentifierNode
                params.append(param_node)
                if self.current_token[0] == 'COMMA':
                    self._advance()
                elif self.current_token[0] == 'RPAREN':
                    break
                else:
                    raise SyntaxError(f"Expected COMMA or RPAREN in parameter list, got {self.current_token[0]}")
        self._expect('RPAREN')
        
        return_type_name = None
        if self.current_token[0] == 'ARROW':
            self._advance() # Consume ARROW
            return_type_name = self._expect('IDENTIFIER')[1] # Store return type name

        self._expect('LBRACE')
        body_statements = []
        while self.current_token[0] != 'RBRACE' and self.current_token[0] is not None:
            body_statements.append(self.parse_statement())
        self._expect('RBRACE')
        return FunctionDefNode(name, params, ProgramNode(body_statements), return_type=return_type_name) # Body is a ProgramNode


    def parse_if_statement(self):
        self._expect('IF')
        condition = self.parse_expression()
        self._expect('LBRACE')
        then_branch_statements = []
        while self.current_token[0] != 'RBRACE' and self.current_token[0] is not None:
            then_branch_statements.append(self.parse_statement())
        self._expect('RBRACE')
        
        else_branch_statements = None
        if self.current_token[0] == 'ELSE':
            self._advance()
            self._expect('LBRACE')
            else_branch_statements = []
            while self.current_token[0] != 'RBRACE' and self.current_token[0] is not None:
                else_branch_statements.append(self.parse_statement())
            self._expect('RBRACE')
        
        then_program_node = ProgramNode(then_branch_statements)
        else_program_node = ProgramNode(else_branch_statements) if else_branch_statements is not None else None
        return IfNode(condition, then_program_node, else_program_node)


    def parse(self):
        statements = []
        while self.current_token[0] is not None: # While not EOF
            try:
                statement = self.parse_statement()
                if statement: # Ensure statement is not None (e.g. from error recovery)
                    statements.append(statement)
                # If parse_statement consumed all tokens (e.g. single expression without semicolon)
                if self.current_token[0] is None and not statements:
                    # If it was a single expression program that parse_statement returned
                    return ProgramNode([statement]) if statement else ProgramNode([])


            except SyntaxError as e:
                print(f"Syntax Error during parsing: {e}") 
                # Simple error recovery: skip the problematic token and try to continue.
                # This might lead to cascaded errors or incomplete ASTs.
                # A more robust recovery would try to find a synchronization point (e.g., next SEMICOLON or RBRACE).
                # For now, we break on error to avoid potential infinite loops with naive advance.
                # self._advance() 
                break 
        return ProgramNode(statements)

if __name__ == '__main__':
    # Assuming lexer.py is in the same directory or accessible
    # from ..lexer.lexer import tokenize # If lexer is in neuralscript-core/lexer/
    
    # Fallback: Re-define tokenize for standalone testing if import fails
    def tokenize_placeholder(code):
        import re
        temp_tokens_spec = [
            ("DEF", r"DEF\b"), ("RETURN", r"RETURN\b"), ("IF", r"IF\b"), ("ELSE", r"ELSE\b"),
            ("TRUE", r"TRUE\b"), ("FALSE", r"FALSE\b"),
            ("ATTEND", r"ATTEND\b"), ("MERGE", r"MERGE\b"), ("SAMPLE", r"SAMPLE\b"),
            ("PROPAGATE", r"PROPAGATE\b"),("EVOLVE", r"EVOLVE\b"),
            ("VECTOR", r"VECTOR\b"), ("MATRIX", r"MATRIX\b"),("SCALAR", r"SCALAR\b"),
            ("IDENTIFIER", r"[a-zA-Z_][a-zA-Z0-9_]*"),
            ("NUMBER", r"[0-9]+(?:\.[0-9]+)?"), ("STRING", r'"[^"]*"'),
            ("ARROW", r"->"), ("EQ", r"=="),("NEQ", r"!="), ("LTE", r"<="), ("GTE", r">="),
            ("LT", r"<"), ("GT", r">"),
            ("PLUS", r"\+"),("MINUS", r"-"),("MULTIPLY", r"\*"),("DIVIDE", r"/"),
            ("ASSIGN", r"="),
            ("LPAREN", r"\("), ("RPAREN", r"\)"), ("LBRACE", r"\{"), ("RBRACE", r"\}"),
            ("LBRACKET", r"\["), ("RBRACKET", r"\]"), ("COMMA", r","),
            ("SEMICOLON", r";"), ("COLON", r":"),
            ("WHITESPACE", r"[ \t\n]+"), ("COMMENT", r"//[^\n]*")
        ]
        tokens = []
        idx = 0
        code_len = len(code)
        while idx < code_len:
            matched_this_round = False
            for token_name, token_regex in temp_tokens_spec:
                match = re.match(token_regex, code[idx:])
                if match:
                    value = match.group(0)
                    if token_name not in ["WHITESPACE", "COMMENT"]:
                        tokens.append((token_name, value))
                    idx += len(value)
                    matched_this_round = True
                    break
            if not matched_this_round:
                # print(f"Unknown token start: {code[idx]}")
                tokens.append(("UNKNOWN", code[idx]))
                idx += 1
        return tokens

    sample_code_complex = """
    DEF process_vectors(q_vec: VECTOR, k_vec: VECTOR) -> VECTOR {
        // A neural operation as part of an assignment
        attention_output = ATTEND q_vec, k_vec, k_vec; 
        
        // An assignment with an expression
        intermediate_result = attention_output + VECTOR [1.0, 0.5];
        
        IF TRUE == FALSE {
            final_result = MERGE intermediate_result, q_vec;
        } ELSE {
            final_result = intermediate_result;
        }
        RETURN final_result;
    }

    main_query = VECTOR [0.1, 0.2, 0.3];
    main_keys = VECTOR [0.4, 0.5, 0.6];
    
    // Function call
    output_vector = process_vectors(main_query, main_keys); 
    // A standalone neural op
    SAMPLE output_vector;
    """

    print("--- Parsing Complex Sample ---")
    tokens_complex = tokenize_placeholder(sample_code_complex)
    # print("\nTokens for complex sample:")
    # for token_tup in tokens_complex: print(token_tup)
    
    parser_complex = Parser(tokens_complex)
    ast_complex = parser_complex.parse()
    print("\nAST for complex sample:")
    print(ast_complex)

    print("\n--- Parsing Simpler ATTEND statement (direct expression) ---")
    tokens_attend_direct = tokenize_placeholder("ATTEND query, key, value;")
    parser_attend_direct = Parser(tokens_attend_direct)
    ast_attend_direct = parser_attend_direct.parse()
    print(ast_attend_direct)

    print("\n--- Parsing Assignment with ATTEND ---")
    tokens_attend_assign = tokenize_placeholder("processed = ATTEND query, key, value;")
    parser_attend_assign = Parser(tokens_attend_assign)
    ast_attend_assign = parser_attend_assign.parse()
    print(ast_attend_assign)
    
    print("\n--- Parsing simple IF statement ---")
    # Note: The previous example was missing semicolons, which the current parser expects.
    tokens_if = tokenize_placeholder("IF x == y { x = x + 1; } ELSE { y = y + 1; }")
    parser_if = Parser(tokens_if)
    ast_if = parser_if.parse()
    print(ast_if)

    print("\n--- Parsing function call with no args ---")
    tokens_call_no_args = tokenize_placeholder("do_something();")
    parser_call_no_args = Parser(tokens_call_no_args)
    ast_call_no_args = parser_call_no_args.parse()
    print(ast_call_no_args)

    print("\n--- Parsing empty vector literal ---")
    tokens_empty_vec = tokenize_placeholder("empty_v = VECTOR [];")
    parser_empty_vec = Parser(tokens_empty_vec)
    ast_empty_vec = parser_empty_vec.parse()
    print(ast_empty_vec)

    print("\n--- Parsing vector with one element ---")
    tokens_single_el_vec = tokenize_placeholder("single_v = VECTOR [1.0];")
    parser_single_el_vec = Parser(tokens_single_el_vec)
    ast_single_el_vec = parser_single_el_vec.parse()
    print(ast_single_el_vec)

    print("\n--- Parsing function def with no params and no return type annotation ---")
    tokens_simple_func = tokenize_placeholder("DEF simple_func() { RETURN 1; }")
    parser_simple_func = Parser(tokens_simple_func)
    ast_simple_func = parser_simple_func.parse()
    print(ast_simple_func)

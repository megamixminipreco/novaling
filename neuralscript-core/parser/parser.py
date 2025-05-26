# neuralscript-core/parser/parser.py

class ASTNode:
    def __init__(self, node_type, children=None, value=None, **kwargs):
        self.type = node_type
        self.value = value 
        self.children = children if children is not None else []
        self.__dict__.update(kwargs) 
        # Store line/col from token if provided
        if 'token' in kwargs and kwargs['token'] is not None and isinstance(kwargs['token'], dict):
             self.line = kwargs['token'].get('line')
             self.col = kwargs['token'].get('col')
        else: # Default if no token provided
            self.line = None
            self.col = None

    def __repr__(self, level=0): # Updated to include line/col if present
        ret = "\t" * level + f"ASTNode(type='{self.type}'"
        if self.value is not None:
            ret += f", value='{self.value}'"
        
        attrs_to_print = ['name', 'operator', 'callee', 'params', 'return_type', 'type_hint', 'has_else']
        for attr in attrs_to_print:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                val = getattr(self, attr)
                if attr == 'params' and isinstance(val, list): 
                    param_repr = "[" + ", ".join([p.__repr__(level+1) if isinstance(p, ASTNode) else str(p) for p in val]) + "]"
                    ret += f", {attr}={param_repr}"
                elif isinstance(val, ASTNode): 
                     ret += f", {attr}=\n" + val.__repr__(level + 1)
                else:
                    ret += f", {attr}='{val}'"
        
        if self.line is not None: ret += f", line={self.line}"
        if self.col is not None: ret += f", col={self.col}"

        if self.children:
            # Avoid double printing children if a named attribute (like 'body' for FunctionDefNode) is the same list.
            is_named_child_list = any(hasattr(self, attr_name) and getattr(self, attr_name) is self.children for attr_name in ['body']) 
            if not is_named_child_list and self.children: 
                ret += ", children=[\n"
                for child in self.children:
                    if child: ret += child.__repr__(level + 1) + ",\n"
                    else: ret += "\t" * (level + 1) + "None,\n" 
                ret += "\t" * level + "]"
        ret += ")"
        return ret

# --- Node factory functions updated to pass token for positional info ---
def ProgramNode(statements, token=None):
    if not token and statements:
        first_stmt_token = getattr(statements[0], 'token', None)
        if first_stmt_token:
            token = first_stmt_token
        elif hasattr(statements[0], 'line') and statements[0].line is not None :
            token = {'line': statements[0].line, 'col': statements[0].col, 'value': 'ProgramStart', 'type': 'PROGRAM_START'}
    if not token: 
        token = {'line': 1, 'col': 1, 'value': 'ProgramStart', 'type': 'PROGRAM_START'}
    return ASTNode(node_type='Program', children=statements, token=token)

def AssignmentNode(target_token, value_expr, assign_op_token):
    return ASTNode(node_type='Assignment', name=target_token['value'], children=[value_expr], token=assign_op_token)

def IdentifierNode(token):
    return ASTNode(node_type='Identifier', name=token['value'], token=token)

def NumberLiteralNode(token):
    return ASTNode(node_type='NumberLiteral', value=float(token['value']), token=token)

def StringLiteralNode(token):
    return ASTNode(node_type='StringLiteral', value=token['value'][1:-1], token=token)

def BooleanLiteralNode(token):
    return ASTNode(node_type='BooleanLiteral', value=(token['value'] == 'TRUE'), token=token)

def VectorLiteralNode(element_nodes, vector_keyword_token):
    return ASTNode(node_type='VectorLiteral', children=element_nodes, token=vector_keyword_token)
    
def BinaryOperationNode(left, op_token, right):
    return ASTNode(node_type='BinaryOperation', operator=op_token['value'], children=[left, right], token=op_token)

def NeuralOperationNode(op_token, arg_nodes):
    return ASTNode(node_type='NeuralOperation', name=op_token['value'], children=arg_nodes, token=op_token)

def FunctionDefNode(def_token, name_token, params_list, return_type_node_or_val, body_program_node):
    if not (isinstance(body_program_node, ASTNode) and body_program_node.type == 'Program'):
        body_program_node = ProgramNode(body_program_node.children if hasattr(body_program_node, 'children') else [], 
                                        token=getattr(body_program_node, 'token', def_token))
    return ASTNode(node_type='FunctionDef', 
                   name=name_token['value'], 
                   params=params_list, 
                   return_type=return_type_node_or_val, 
                   children=[body_program_node], # Body is a ProgramNode
                   token=def_token)

def ReturnNode(return_token, expression_node=None):
    # Ensure expression_node is not None before putting it in a list if it's the only child
    children_list = [expression_node] if expression_node is not None else []
    return ASTNode(node_type='Return', children=children_list, token=return_token)


def IfNode(if_token, condition_node, then_branch_program_node, else_branch_program_node=None):
    children = [condition_node, then_branch_program_node]
    if else_branch_program_node:
        children.append(else_branch_program_node)
    return ASTNode(node_type='If', children=children, has_else=else_branch_program_node is not None, token=if_token)

def CallNode(callee_token, arg_nodes): 
     return ASTNode(node_type='Call', callee=callee_token['value'], children=arg_nodes, token=callee_token)


class Parser:
    def __init__(self, tokens):
        self.tokens = [t for t in tokens if t['type'] not in ['WHITESPACE', 'COMMENT', 'NEWLINE']]
        self.pos = 0
        
        eof_line, eof_col = 1, 1 
        if self.tokens:
            last_real_token = self.tokens[-1]
            eof_line = last_real_token['line']
            eof_col = last_real_token['col'] + len(str(last_real_token['value'])) 
        else: 
            eof_line = 1 
            eof_col = 1

        self.eof_sentinel = {'type': None, 'value': None, 'line': eof_line, 'col': eof_col, 'error_type': 'EOF'}
        self.current_token = self.tokens[self.pos] if self.pos < len(self.tokens) else self.eof_sentinel

    def _advance(self):
        self.pos += 1
        self.current_token = self.tokens[self.pos] if self.pos < len(self.tokens) else self.eof_sentinel

    def _peek(self):
        return self.tokens[self.pos + 1] if self.pos + 1 < len(self.tokens) else self.eof_sentinel 

    def _expect(self, token_type, token_value=None):
        tok = self.current_token
        if tok['type'] == token_type:
            if token_value is None or tok['value'] == token_value:
                consumed_token = tok
                self._advance()
                return consumed_token
            else:
                err = SyntaxError(
                    f"Line {tok['line']}:{tok['col']}: Expected token value '{token_value}' for type '{token_type}', "
                    f"got value '{tok['value']}'"
                )
                raise err
        elif tok.get('error_type') == 'EOF': 
             err = SyntaxError(
                f"Line {tok['line']}:{tok['col']}: Unexpected end of input. Expected '{token_type}'."
            )
             raise err
        else: 
            err = SyntaxError(
                f"Line {tok['line']}:{tok['col']}: Expected token type '{token_type}', "
                f"got type '{tok['type']}' (value: '{tok['value']}')"
            )
            raise err

    def parse_atom(self):
        tok = self.current_token
        if tok['type'] == 'NUMBER':
            self._advance()
            return NumberLiteralNode(tok)
        elif tok['type'] == 'STRING':
            self._advance()
            return StringLiteralNode(tok)
        elif tok['type'] == 'TRUE' or tok['type'] == 'FALSE':
            self._advance()
            return BooleanLiteralNode(tok)
        elif tok['type'] == 'IDENTIFIER':
            identifier_token = tok 
            self._advance() 
            if self.current_token['type'] == 'LPAREN': 
                self._expect('LPAREN') 
                args = []
                if self.current_token['type'] != 'RPAREN':
                    while True:
                        if self.current_token.get('error_type') == 'EOF':
                             raise SyntaxError(f"Line {identifier_token['line']}:{identifier_token['col']}: Unexpected EOF in function call argument list for '{identifier_token['value']}'. Expected expression or ')'.")
                        args.append(self.parse_expression())
                        if self.current_token['type'] == 'COMMA':
                            self._advance()
                        elif self.current_token['type'] == 'RPAREN':
                            break
                        else:
                            err_tok_arg = self.current_token
                            raise SyntaxError(f"Line {err_tok_arg['line']}:{err_tok_arg['col']}: Expected ',' or ')', got '{err_tok_arg['value']}' (type: {err_tok_arg['type']})")
                self._expect('RPAREN')
                return CallNode(callee_token=identifier_token, arg_nodes=args)
            return IdentifierNode(identifier_token) 
        elif tok['type'] == 'LPAREN':
            start_paren_tok = tok # Keep for potential ParenExpressionNode position
            self._advance()
            expr = self.parse_expression()
            self._expect('RPAREN')
            # If creating a specific ParenExpressionNode:
            # return ASTNode(node_type='ParenExpression', children=[expr], token=start_paren_tok)
            return expr 
        elif tok['type'] == 'VECTOR':
            vector_tok = tok
            self._advance() 
            self._expect('LBRACKET')
            elements = []
            if self.current_token['type'] != 'RBRACKET':
                while True:
                    if self.current_token.get('error_type') == 'EOF':
                        raise SyntaxError(f"Line {vector_tok['line']}:{vector_tok['col']}: Unexpected EOF in vector literal. Expected ']' or vector elements.")
                    elements.append(self.parse_expression())
                    if self.current_token['type'] == 'COMMA':
                        self._advance()
                    elif self.current_token['type'] == 'RBRACKET':
                        break
                    else:
                        err_tok_vec = self.current_token
                        raise SyntaxError(f"Line {err_tok_vec['line']}:{err_tok_vec['col']}: Expected ',' or ']' in vector literal, got '{err_tok_vec['value']}' (type: {err_tok_vec['type']})")
            self._expect('RBRACKET')
            return VectorLiteralNode(elements, vector_keyword_token=vector_tok)
        elif tok['type'] in ['ATTEND', 'SAMPLE', 'PROPAGATE', 'MERGE', 'EVOLVE']:
            op_tok = tok
            self._advance() 
            args = []
            # Neural ops argument parsing. Arguments are expressions.
            # Stop before a semicolon or other statement-starting/block-ending tokens.
            while self.current_token['type'] not in ['SEMICOLON', None, 'RPAREN', 'RBRACE', 'LBRACE', 'IF', 'DEF', 'RETURN', 'WHILE']:
                args.append(self.parse_expression()) # Arguments are full expressions
                if self.current_token['type'] == 'COMMA':
                    self._advance()
                else:
                    break 
            return NeuralOperationNode(op_token=op_tok, arg_nodes=args)

        err_tok = self.current_token
        if err_tok.get('error_type') == 'EOF': 
            raise SyntaxError(f"Line {err_tok['line']}:{err_tok['col']}: Unexpected end of input. Expected an expression atom.")
        else:
            raise SyntaxError(f"Line {err_tok['line']}:{err_tok['col']}: Unexpected token '{err_tok['value']}' (type: {err_tok['type']}) when expecting an expression atom.")

    def parse_term(self): 
        node = self.parse_atom()
        while self.current_token['type'] in ('MULTIPLY', 'DIVIDE'):
            op_token = self.current_token
            self._advance()
            right = self.parse_atom()
            node = BinaryOperationNode(left=node, op_token=op_token, right=right)
        return node

    def parse_expression(self): 
        node = self.parse_term()
        while self.current_token['type'] in ('PLUS', 'MINUS', 'EQ', 'NEQ', 'LT', 'GT', 'LTE', 'GTE'):
            op_token = self.current_token
            self._advance()
            right = self.parse_term()
            node = BinaryOperationNode(left=node, op_token=op_token, right=right)
        return node

    def parse_statement(self):
        stmt_start_token = self.current_token 

        if stmt_start_token['type'] == 'DEF':
            return self.parse_function_definition()
        elif stmt_start_token['type'] == 'RETURN':
            self._advance() 
            expr = None
            # Check if there's an expression to return or just a semicolon
            if self.current_token['type'] != 'SEMICOLON': 
                if self.current_token.get('error_type') == 'EOF': # Check for EOF before parsing expression
                    raise SyntaxError(f"Line {stmt_start_token['line']}:{stmt_start_token['col']}: Unexpected EOF after RETURN. Expected expression or ';'.")
                expr = self.parse_expression()
            self._expect('SEMICOLON')
            return ReturnNode(return_token=stmt_start_token, expression_node=expr)
        elif stmt_start_token['type'] == 'IF':
            return self.parse_if_statement()
        elif stmt_start_token['type'] == 'IDENTIFIER' and self._peek()['type'] == 'ASSIGN':
            identifier_token = self._expect('IDENTIFIER') 
            assign_op_token = self._expect('ASSIGN')
            # Check for missing expression after '='
            if self.current_token['type'] == 'SEMICOLON':
                raise SyntaxError(f"Line {assign_op_token['line']}:{assign_op_token['col'] + len(assign_op_token['value'])}: Unexpected ';'. Expected expression after '=' for assignment to '{identifier_token['value']}'.")
            if self.current_token.get('error_type') == 'EOF': # Check for EOF before parsing expression
                 raise SyntaxError(f"Line {assign_op_token['line']}:{assign_op_token['col'] + len(assign_op_token['value'])}: Unexpected EOF. Expected expression after '=' for assignment to '{identifier_token['value']}'.")
            value_expr = self.parse_expression()
            self._expect('SEMICOLON') 
            return AssignmentNode(target_token=identifier_token, value_expr=value_expr, assign_op_token=assign_op_token)
        else: 
            # This handles expression statements (e.g., function calls, standalone neural ops)
            expr_node = self.parse_expression() 
            # Expect a semicolon after an expression statement
            self._expect('SEMICOLON')
            return expr_node

    def parse_block(self):
        block_start_token = self._expect('LBRACE')
        statements = []
        # Allow empty blocks
        while self.current_token['type'] != 'RBRACE' and self.current_token['type'] is not None:
            statements.append(self.parse_statement())
        self._expect('RBRACE')
        return ProgramNode(statements, token=block_start_token) 

    def parse_function_definition(self):
        def_token = self._expect('DEF')
        name_token = self._expect('IDENTIFIER')
        self._expect('LPAREN')
        params = []
        if self.current_token['type'] != 'RPAREN':
            while True:
                param_name_token = self._expect('IDENTIFIER')
                param_node = IdentifierNode(param_name_token)
                if self.current_token['type'] == 'COLON':
                    self._advance() 
                    type_name_token = self._expect('IDENTIFIER') 
                    param_node.type_hint = type_name_token['value'] 
                params.append(param_node)
                if self.current_token['type'] == 'COMMA':
                    self._advance()
                elif self.current_token['type'] == 'RPAREN':
                    break
                else:
                    raise SyntaxError(f"Line {self.current_token['line']}:{self.current_token['col']}: Expected COMMA or RPAREN in parameter list, got '{self.current_token['value']}'")
        self._expect('RPAREN')
        
        return_type_val = None 
        if self.current_token['type'] == 'ARROW':
            self._advance() 
            return_type_token = self._expect('IDENTIFIER') 
            return_type_val = return_type_token['value']

        body_program_node = self.parse_block() 
        return FunctionDefNode(def_token, name_token, params, return_type_val, body_program_node)

    def parse_if_statement(self):
        if_token = self._expect('IF')
        condition = self.parse_expression() 
        then_branch = self.parse_block()
        else_branch = None
        if self.current_token['type'] == 'ELSE':
            self._advance()
            else_branch = self.parse_block()
        return IfNode(if_token, condition, then_branch, else_branch)

    def parse(self): 
        statements = []
        program_node_token_ref = self.tokens[0] if self.tokens else {'line': 1, 'col': 1, 'value': 'ProgramStart', 'type':'PROGRAM_START'}

        while self.current_token['type'] is not None: 
            try:
                statement = self.parse_statement()
                if statement: 
                    statements.append(statement)
                if self.current_token['type'] is None and not statements and statement: # Single expression program
                     return ProgramNode([statement], token=program_node_token_ref)
            except SyntaxError as e:
                # Ensure lineno and offset are set on the exception for consistent error reporting
                # Python's SyntaxError might not have these if raised manually without them.
                final_line = getattr(e, 'lineno', self.current_token['line'])
                final_col = getattr(e, 'offset', self.current_token['col'])
                if final_line is None or final_line == -1 : final_line = self.eof_sentinel['line']
                if final_col is None or final_col == -1 : final_col = self.eof_sentinel['col']
                
                # Reconstruct message if needed to ensure line/col are prominent
                msg_prefix = f"Line {final_line}:{final_col}: "
                original_msg = str(e)
                # Avoid double-prefixing if the error message from _expect already has it
                msg = original_msg if original_msg.lower().startswith("line ") else msg_prefix + original_msg
                
                print(f"Parser Error: {msg}") 
                
                error_node = ASTNode(node_type="ErrorNode", value=msg, 
                                     token={'line': final_line, 'col': final_col, 'type': 'ERROR'})
                statements.append(error_node)
                # For this subtask, return partial AST upon first error.
                return ProgramNode(statements, token=program_node_token_ref) 
        
        final_program_token = program_node_token_ref
        if statements:
            first_stmt = statements[0]
            if hasattr(first_stmt, 'token') and first_stmt.token:
                final_program_token = first_stmt.token
            elif hasattr(first_stmt, 'line') and first_stmt.line is not None:
                 final_program_token = {'line': first_stmt.line, 'col': first_stmt.col, 'value':'ProgramStart', 'type':'PROGRAM_START'}
        
        return ProgramNode(statements, token=final_program_token)

if __name__ == '__main__':
    import re 
    _LEXER_TOKENS_SPEC = [ 
        ("DEF", r"DEF\b"), ("RETURN", r"RETURN\b"), ("IF", r"IF\b"), ("ELSE", r"ELSE\b"),
        ("TRUE", r"TRUE\b"), ("FALSE", r"FALSE\b"),
        ("ATTEND", r"ATTEND\b"), ("MERGE", r"MERGE\b"), ("VECTOR", r"VECTOR\b"),("SCALAR", r"SCALAR\b"),
        ("IDENTIFIER", r"[a-zA-Z_][a-zA-Z0-9_]*"),
        ("NUMBER", r"[0-9]+(?:\.[0-9]+)?"), ("STRING", r'"[^"]*"'),
        ("ARROW", r"->"), ("EQ", r"=="), ("ASSIGN", r"="), ("PLUS", r"\+"),
        ("LPAREN", r"\("), ("RPAREN", r"\)"), ("LBRACE", r"\{"), ("RBRACE", r"\}"),
        ("LBRACKET", r"\["), ("RBRACKET", r"\]"), ("COMMA", r","),
        ("SEMICOLON", r";"), ("COLON", r":"),
        ("NEWLINE", r"\n"), ("WHITESPACE", r"[ \t]+"), ("COMMENT", r"//[^\n]*")
    ]
    def tokenize_for_parser_test(code): 
        tokens = []
        line, col = 1, 1
        remaining = code
        while remaining:
            matched_any = False
            for name, regex in _LEXER_TOKENS_SPEC:
                match = re.match(regex, remaining)
                if match:
                    val = match.group(0)
                    tok_data = {'type': name, 'value': val, 'line': line, 'col': col}
                    if name == 'NEWLINE': line += 1; col = 1
                    elif name not in ['WHITESPACE', 'COMMENT', 'NEWLINE']: 
                        tokens.append(tok_data)
                        col += len(val) 
                    elif name in ['WHITESPACE', 'COMMENT']: 
                        col += len(val)
                    remaining = remaining[len(val):]
                    matched_any = True
                    break
            if not matched_any: 
                tokens.append({'type': 'UNKNOWN', 'value': remaining[0], 'line': line, 'col': col})
                remaining = remaining[1:]; col += 1
        return tokens

    print("--- Testing Parser with Valid Code ---")
    valid_code = "x = 10 + y; DEF my_func() -> SCALAR { RETURN x; }"
    tokens_valid = tokenize_for_parser_test(valid_code)
    parser_valid = Parser(tokens_valid)
    ast_valid = parser_valid.parse()
    is_error_present_valid = any(child.type == "ErrorNode" for child in ast_valid.children) if hasattr(ast_valid, 'children') and ast_valid.children else False
    if not is_error_present_valid:
        print("Valid code parsed successfully.")
    else:
        print("Valid code parsing reported an error unexpectedly.")
        if hasattr(ast_valid, 'children') and ast_valid.children:
            for node_item in ast_valid.children: 
                if node_item.type == "ErrorNode": print(f"  ErrorNode content: {node_item.value}")


    print("\n--- Testing Parser with Syntax Error: Missing Semicolon ---")
    error_code_semicolon = "x = 10\ny = 20;" # Error should be on 'y' as unexpected or missing ';' after '10'
    tokens_err_semi = tokenize_for_parser_test(error_code_semicolon)
    parser_err_semi = Parser(tokens_err_semi)
    ast_err_semi = parser_err_semi.parse() 
    is_error_present_semi = any(child.type == "ErrorNode" for child in ast_err_semi.children) if hasattr(ast_err_semi, 'children') and ast_err_semi.children else False
    if is_error_present_semi:
        print("Parsed with error as expected (Missing Semicolon / Unexpected token 'y').")
    else:
        print("FAIL: Expected error for missing semicolon, but none found.")


    print("\n--- Testing Parser with Syntax Error: Unexpected Token ---")
    error_code_unexpected = "x = @ + 1;" 
    tokens_err_unexp = tokenize_for_parser_test(error_code_unexpected) 
    parser_err_unexp = Parser(tokens_err_unexp)
    ast_err_unexp = parser_err_unexp.parse()
    is_error_present_unexp = any(child.type == "ErrorNode" for child in ast_err_unexp.children) if hasattr(ast_err_unexp, 'children') and ast_err_unexp.children else False
    if is_error_present_unexp:
        print("Parsed with error as expected (Unexpected Token).")
    else:
        print("FAIL: Expected error for unexpected token, but none found.")
        
    print("\n--- Testing Parser with Syntax Error: Missing RPAREN ---")
    error_code_rparen = "CALL my_func(arg1, arg2 ;" 
    tokens_err_rparen = tokenize_for_parser_test(error_code_rparen)
    parser_err_rparen = Parser(tokens_err_rparen)
    ast_err_rparen = parser_err_rparen.parse()
    is_error_present_rparen = any(child.type == "ErrorNode" for child in ast_err_rparen.children) if hasattr(ast_err_rparen, 'children') and ast_err_rparen.children else False
    if is_error_present_rparen:
        print("Parsed with error as expected (Missing RPAREN).")
    else:
        print("FAIL: Expected error for missing RPAREN, but none found.")

    print("\n--- Testing Parser with Syntax Error: Incomplete Assignment ---")
    error_code_incomplete_assign = "my_var = ;"
    tokens_err_incomplete = tokenize_for_parser_test(error_code_incomplete_assign)
    parser_err_incomplete = Parser(tokens_err_incomplete)
    ast_incomplete = parser_err_incomplete.parse()
    is_error_present_incomplete = any(child.type == "ErrorNode" for child in ast_incomplete.children) if hasattr(ast_incomplete, 'children') and ast_incomplete.children else False
    if is_error_present_incomplete:
        print("Parsed with error as expected (Incomplete Assignment).")
    else:
        print("FAIL: Expected error for incomplete assignment, but none found.")

    print("\n--- Testing Parser with Syntax Error: EOF in vector ---")
    error_code_eof_vector = "v = VECTOR [1, 2" 
    tokens_eof_vector = tokenize_for_parser_test(error_code_eof_vector)
    parser_eof_vector = Parser(tokens_eof_vector)
    ast_eof_vector = parser_eof_vector.parse()
    is_error_present_eof_vector = any(child.type == "ErrorNode" for child in ast_eof_vector.children) if hasattr(ast_eof_vector, 'children') and ast_eof_vector.children else False
    if is_error_present_eof_vector:
        print("Parsed with error as expected (EOF in vector).")
    else:
        print("FAIL: Expected error for EOF in vector, but none found.")

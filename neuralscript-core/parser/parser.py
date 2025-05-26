# neuralscript-core/parser/parser.py

class ASTNode:
    def __init__(self, type, children=None, value=None):
        self.type = type
        self.value = value
        self.children = children if children is not None else []

    def __repr__(self):
        return f"ASTNode(type='{self.type}', value='{self.value}', children={self.children})"

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.current_token = self.tokens[self.pos] if self.tokens else None

    def _advance(self):
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current_token = self.tokens[self.pos]
        else:
            self.current_token = None

    def _expect(self, token_type, token_value=None):
        if self.current_token and self.current_token[0] == token_type:
            if token_value is None or self.current_token[1] == token_value:
                token = self.current_token
                self._advance()
                return token
            else:
                raise SyntaxError(f"Expected token value {token_value} for type {token_type}, got {self.current_token[1]}")
        raise SyntaxError(f"Expected token type {token_type}, got {self.current_token[0] if self.current_token else 'None'}")

    def parse_vector_literal(self):
        self._expect("LBRACKET", "[")
        elements = []
        while self.current_token and self.current_token[0] == "NUMBER":
            elements.append(ASTNode(type="NumberLiteral", value=self.current_token[1]))
            self._advance()
            if self.current_token and self.current_token[0] == "COMMA":
                self._advance() # Consume comma
            elif self.current_token and self.current_token[0] == "RBRACKET":
                break # End of vector before a comma
            elif self.current_token:
                 raise SyntaxError(f"Expected COMMA or RBRACKET in vector, got {self.current_token}")
            else:
                raise SyntaxError("Unexpected end of input in vector literal")
        self._expect("RBRACKET", "]")
        return ASTNode(type="VectorLiteral", children=elements)

    def parse_assignment(self):
        # Example: my_vector = VECTOR [1, 2, 3]
        identifier_token = self._expect("IDENTIFIER")
        variable_name = identifier_token[1]
        self._expect("ASSIGN", "=")

        if self.current_token and self.current_token[0] == "VECTOR":
            self._advance() # Consume VECTOR keyword
            vector_node = self.parse_vector_literal()
            return ASTNode(type="Assignment", value=variable_name, children=[vector_node])
        else:
            raise SyntaxError("Expected VECTOR keyword after assignment")

    def parse(self):
        # For MVP, assume the program is a single assignment statement
        # A more robust parser would handle multiple statements, EOF, etc.
        if not self.tokens:
            return ASTNode(type="EmptyProgram")
        
        # Try parsing an assignment
        # This is a very simplified approach
        if self.tokens[0][0] == "IDENTIFIER" and len(self.tokens) > 1 and self.tokens[1][0] == "ASSIGN":
            return self.parse_assignment()
        
        # Placeholder for other statement types
        # For now, if it's not an assignment, return a generic Program node
        # or raise an error for unhandled structures.
        
        # Simple fallback for other potential structures (highly simplified for MVP)
        # This part would need significant expansion for a real language
        nodes = []
        while self.current_token:
            # Crude way to consume tokens for MVP if not assignment
            nodes.append(ASTNode(type=self.current_token[0], value=self.current_token[1]))
            self._advance()
        return ASTNode(type="Program", children=nodes)


if __name__ == '__main__':
    # Assuming lexer.py is in the parent directory or accessible via PYTHONPATH
    # For direct execution, we might need to adjust imports or provide tokens directly
    
    # Sample tokens for: my_vector = VECTOR [1, 2, 3]
    sample_tokens_1 = [
        ('IDENTIFIER', 'my_vector'), ('ASSIGN', '='), ('VECTOR', 'VECTOR'), 
        ('LBRACKET', '['), ('NUMBER', '1'), ('COMMA', ','), ('NUMBER', '2'), 
        ('COMMA', ','), ('NUMBER', '3'), ('RBRACKET', ']')
    ]
    parser1 = Parser(sample_tokens_1)
    ast1 = parser1.parse()
    print("AST for 'my_vector = VECTOR [1, 2, 3]':")
    print(ast1)

    # Sample tokens for: ATTEND query
    # This will be parsed by the fallback mechanism in the current MVP parser
    sample_tokens_2 = [
        ('ATTEND', 'ATTEND'), ('IDENTIFIER', 'query')
    ]
    parser2 = Parser(sample_tokens_2)
    ast2 = parser2.parse()
    print("\nAST for 'ATTEND query':")
    print(ast2)
    
    # Sample tokens for an empty input
    parser3 = Parser([])
    ast3 = parser3.parse()
    print("\nAST for empty input:")
    print(ast3)

# neuralscript-core/lexer/lexer.py
import re

TOKENS = [
    # Keywords for neural operations
    ("ATTEND", r"ATTEND\b"),
    ("SAMPLE", r"SAMPLE\b"),
    ("PROPAGATE", r"PROPAGATE\b"),
    ("MERGE", r"MERGE\b"),
    ("EVOLVE", r"EVOLVE\b"),

    # Data types/constructors
    ("VECTOR", r"VECTOR\b"),
    ("MATRIX", r"MATRIX\b"),
    ("SCALAR", r"SCALAR\b"), # Added \b for consistency

    # Boolean Literals
    ("TRUE", r"TRUE\b"),
    ("FALSE", r"FALSE\b"),

    # Control Flow
    ("IF", r"IF\b"),
    ("ELSE", r"ELSE\b"),
    ("WHILE", r"WHILE\b"),
    
    # Function Definition (basic)
    ("DEF", r"DEF\b"),
    ("RETURN", r"RETURN\b"),

    # Operators
    ("ARROW", r"->"),
    ("EQ", r"=="),
    ("NEQ", r"!="),
    ("LTE", r"<="),
    ("GTE", r">="),
    ("LT", r"<"),
    ("GT", r">"),
    ("PLUS", r"\+"),
    ("MINUS", r"-"), # Minus can also be part of an identifier if not careful, but regex order helps
    ("MULTIPLY", r"\*"),
    ("DIVIDE", r"/"),
    ("ASSIGN", r"="), # Already exists

    # Delimiters
    ("LPAREN", r"\("),       # Already exists
    ("RPAREN", r"\)"),       # Already exists
    ("LBRACE", r"\{"),       # Already exists
    ("RBRACE", r"\}"),       # Already exists
    ("LBRACKET", r"\["),     # Already exists
    ("RBRACKET", r"\]"),     # Already exists
    ("COMMA", r","),          # Already exists
    ("COLON", r":"),
    ("SEMICOLON", r";"), # Optional: for statement termination

    # Literals
    ("IDENTIFIER", r"[a-zA-Z_][a-zA-Z0-9_]*"), # Already exists
    ("NUMBER", r"[0-9]+(?:\.[0-9]+)?"), # Support for floats, was r"[0-9]+"
    ("STRING", r'"[^"]*"'), # Basic string literal

    # Comments (to be ignored)
    ("COMMENT", r"//[^\n]*"),

    # Whitespace (to be ignored by the lexer logic)
    ("WHITESPACE", r"[ \t\n]+"), # Already exists
]

# Basic lexer function (placeholder)
def tokenize(code):
    tokens = []
    remaining_code = code
    while remaining_code:
        matched = False
        for token_name, token_regex in TOKENS:
            match = re.match(token_regex, remaining_code)
            if match:
                value = match.group(0)
                if token_name not in ["WHITESPACE", "COMMENT"]: # Skip whitespace and comments
                    tokens.append((token_name, value))
                remaining_code = remaining_code[len(value):]
                matched = True
                break
        if not matched:
            # If no token matches, it's an error or unrecognized character
            # For now, just consume the character and mark as UNKNOWN
            # A real lexer would handle this more robustly with error reporting (line/col)
            tokens.append(("UNKNOWN", remaining_code[0]))
            remaining_code = remaining_code[1:]
    return tokens

if __name__ == '__main__':
    sample_code = """
    // This is a NeuralScript example
    DEF my_attention_fn(query_vec, key_vec, value_vec) -> VECTOR {
        result = ATTEND query_vec, key_vec, value_vec;
        RETURN result;
    }

    v1 = VECTOR [1.0, 2.5, 3.0];
    v2 = my_attention_fn(v1, v1, v1); // Simplified call
    
    IF TRUE == TRUE {
        v3 = MERGE v1, v2;
        // PRINT v3; // Assuming a PRINT token might be added later
    } ELSE {
        // Do nothing
    }
    """
    tokens = tokenize(sample_code)
    print(f"Tokens for sample code:")
    for token in tokens:
        print(token)
    
    print("\nTokens for 'EVOLVE complex_structure':")
    tokens_evolve = tokenize("EVOLVE complex_structure")
    for token in tokens_evolve:
        print(token)

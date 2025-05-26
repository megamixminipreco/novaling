# neuralscript-core/lexer/lexer.py

TOKENS = [
    # Keywords for neural operations
    ("ATTEND", "ATTEND"),
    ("SAMPLE", "SAMPLE"),
    ("PROPAGATE", "PROPAGATE"),
    ("MERGE", "MERGE"),
    ("EVOLVE", "EVOLVE"),

    # Data types
    ("VECTOR", "VECTOR"),
    ("MATRIX", "MATRIX"),
    ("SCALAR", "SCALAR"),

    # Literals (very basic for now)
    ("IDENTIFIER", r"[a-zA-Z_][a-zA-Z0-9_]*"),
    ("NUMBER", r"[0-9]+"),

    # Delimiters and Operators (simplified)
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("LBRACE", r"\{"),
    ("RBRACE", r"\}"),
    ("LBRACKET", r"\["),
    ("RBRACKET", r"\]"),
    ("COMMA", r","),
    ("ASSIGN", r"="),

    # Whitespace (to be ignored by the lexer)
    ("WHITESPACE", r"[ \t\n]+"),
]

# Basic lexer function (placeholder)
def tokenize(code):
    # This will be a very simple regex-based tokenizer for now
    # It won't handle complex cases or errors gracefully yet
    tokens = []
    remaining_code = code
    while remaining_code:
        matched = False
        for token_name, token_regex in TOKENS:
            # Basic regex matching
            import re
            match = re.match(token_regex, remaining_code)
            if match:
                value = match.group(0)
                if token_name != "WHITESPACE": # Skip whitespace
                    tokens.append((token_name, value))
                remaining_code = remaining_code[len(value):]
                matched = True
                break
        if not matched:
            # If no token matches, it's an error or unrecognized character
            # For MVP, we'll just consume the character and mark as UNKNOWN
            # A real lexer would handle this more robustly
            tokens.append(("UNKNOWN", remaining_code[0]))
            remaining_code = remaining_code[1:]
    return tokens

if __name__ == '__main__':
    sample_code = "ATTEND query = VECTOR [1, 2, 3]"
    tokens = tokenize(sample_code)
    print(f"Tokens for '{sample_code}':")
    for token in tokens:
        print(token)

    sample_code_2 = "MERGE dist1, dist2"
    tokens_2 = tokenize(sample_code_2)
    print(f"\nTokens for '{sample_code_2}':")
    for token in tokens_2:
        print(token)

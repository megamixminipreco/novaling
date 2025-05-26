# neuralscript-core/lexer/lexer.py
import re

# Keep the TOKENS list as is (from previous step)
TOKENS = [
    ("ATTEND", r"ATTEND\b"), ("SAMPLE", r"SAMPLE\b"), ("PROPAGATE", r"PROPAGATE\b"),
    ("MERGE", r"MERGE\b"), ("EVOLVE", r"EVOLVE\b"),
    ("VECTOR", r"VECTOR\b"), ("MATRIX", r"MATRIX\b"), ("SCALAR", r"SCALAR\b"),
    ("TRUE", r"TRUE\b"), ("FALSE", r"FALSE\b"),
    ("IF", r"IF\b"), ("ELSE", r"ELSE\b"), ("WHILE", r"WHILE\b"),
    ("DEF", r"DEF\b"), ("RETURN", r"RETURN\b"),
    ("ARROW", r"->"), ("EQ", r"=="), ("NEQ", r"!="), ("LTE", r"<="), ("GTE", r">="),
    ("LT", r"<"), ("GT", r">"), ("PLUS", r"\+"), ("MINUS", r"-"),
    ("MULTIPLY", r"\*"), ("DIVIDE", r"/"), ("ASSIGN", r"="),
    ("LPAREN", r"\("), ("RPAREN", r"\)"), ("LBRACE", r"\{"), ("RBRACE", r"\}"),
    ("LBRACKET", r"\["), ("RBRACKET", r"\]"), ("COMMA", r","),
    ("COLON", r":"), ("SEMICOLON", r";"),
    ("IDENTIFIER", r"[a-zA-Z_][a-zA-Z0-9_]*"),
    ("NUMBER", r"[0-9]+(?:\.[0-9]+)?"),
    ("STRING", r'"[^"]*"'),
    ("COMMENT", r"//[^\n]*"),
    ("NEWLINE", r"\n"), # Explicit NEWLINE token for line tracking
    ("WHITESPACE", r"[ \t]+"), # Whitespace excluding newline
]


def tokenize(code):
    tokens = []
    line_num = 1
    col_num = 1
    remaining_code = code

    while remaining_code:
        matched = False
        for token_name, token_regex in TOKENS:
            match = re.match(token_regex, remaining_code)
            if match:
                value = match.group(0)
                
                # Create token data before potentially changing line_num and col_num for NEWLINE
                current_col = col_num

                if token_name == "NEWLINE":
                    line_num += 1
                    col_num = 1
                    # Do not add NEWLINE tokens to the output stream for the parser
                elif token_name == "WHITESPACE":
                    col_num += len(value)
                    # Do not add WHITESPACE tokens
                elif token_name == "COMMENT":
                    # Comments are ignored. Column advances by length of comment.
                    # Current regex `//[^\n]*` ensures comment doesn't include newline itself.
                    col_num += len(value)
                else: # Actual token to be added
                    token_data = {
                        'type': token_name,
                        'value': value,
                        'line': line_num,
                        'col': current_col # Use column at the start of the token
                    }
                    tokens.append(token_data)
                    col_num += len(value) # Advance column by length of the token

                remaining_code = remaining_code[len(value):]
                matched = True
                break
        
        if not matched:
            # Unrecognized character
            unknown_char = remaining_code[0]
            tokens.append({
                'type': "UNKNOWN",
                'value': unknown_char,
                'line': line_num,
                'col': col_num # Column where the unknown char is found
            })
            # Decide how to advance col_num for unknown chars.
            # Simple assumption: advance by 1. If it was a tab, this might be off.
            col_num += 1
            remaining_code = remaining_code[1:]
            
    return tokens

if __name__ == '__main__':
    sample_code_ok = """
    // Test code
    DEF my_func(val: NUMBER) -> NUMBER {
        x = val + 10; // Indented
        RETURN x;
    }
    """
    print("--- Tokens for OK code ---")
    tokens_ok = tokenize(sample_code_ok)
    for token in tokens_ok:
        print(token)

    sample_code_error = """
    y = 20;
    z = $ # Invalid character
    a = "another line";
    """
    print("\n--- Tokens for code with an error ---")
    tokens_err = tokenize(sample_code_error)
    for token in tokens_err:
        if token['type'] == 'UNKNOWN':
            print(f"LEXER ERROR: Unknown token '{token['value']}' at Line {token['line']}, Col {token['col']}")
        else:
            print(token)

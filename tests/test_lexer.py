# tests/test_lexer.py
import unittest
import sys
import os

# Add neuralscript-core to sys.path to allow direct import of lexer
# This assumes 'tests' is at the same level as 'neuralscript-core' directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../neuralscript-core')))
from lexer.lexer import tokenize # Adjusted import path

class TestLexer(unittest.TestCase):

    def test_neural_keywords(self):
        code = "ATTEND SAMPLE PROPAGATE MERGE EVOLVE"
        expected_tokens = [
            ('ATTEND', 'ATTEND'), ('SAMPLE', 'SAMPLE'), ('PROPAGATE', 'PROPAGATE'),
            ('MERGE', 'MERGE'), ('EVOLVE', 'EVOLVE')
        ]
        self.assertEqual(tokenize(code), expected_tokens)

    def test_identifiers_and_numbers(self):
        code = "var1 = 123 + 45.67"
        expected_tokens = [
            ('IDENTIFIER', 'var1'), ('ASSIGN', '='), ('NUMBER', '123'),
            ('PLUS', '+'), ('NUMBER', '45.67')
        ]
        self.assertEqual(tokenize(code), expected_tokens)

    def test_string_literal(self):
        code = 'msg = "hello world"'
        expected_tokens = [
            ('IDENTIFIER', 'msg'), ('ASSIGN', '='), ('STRING', '"hello world"')
        ]
        self.assertEqual(tokenize(code), expected_tokens)

    def test_control_flow_and_delimiters(self):
        code = "IF x > 0 { RETURN y; }"
        expected_tokens = [
            ('IF', 'IF'), ('IDENTIFIER', 'x'), ('GT', '>'), ('NUMBER', '0'),
            ('LBRACE', '{'), ('RETURN', 'RETURN'), ('IDENTIFIER', 'y'),
            ('SEMICOLON', ';'), ('RBRACE', '}')
        ]
        self.assertEqual(tokenize(code), expected_tokens)
        
    def test_comments_and_whitespace(self):
        code = """
        // This is a comment
        x = 10; // Another comment
        y = 20;
        """
        expected_tokens = [
            ('IDENTIFIER', 'x'), ('ASSIGN', '='), ('NUMBER', '10'), ('SEMICOLON', ';'),
            ('IDENTIFIER', 'y'), ('ASSIGN', '='), ('NUMBER', '20'), ('SEMICOLON', ';')
        ]
        self.assertEqual(tokenize(code), expected_tokens)

    def test_unknown_token_reporting(self):
        code = "a = 10; \n$ সমস্যা" # $ is unknown, সমস্যা are unicode chars
        tokens = tokenize(code)
        # Expected:
        # {'type': 'IDENTIFIER', 'value': 'a', 'line': 1, 'col': 1}, 
        # {'type': 'ASSIGN', 'value': '=', 'line': 1, 'col': 3}, 
        # {'type': 'NUMBER', 'value': '10', 'line': 1, 'col': 5}, 
        # {'type': 'SEMICOLON', 'value': ';', 'line': 1, 'col': 7},
        # -- NEWLINE processed by lexer for line count --
        # {'type': 'UNKNOWN', 'value': '$', 'line': 2, 'col': 1},
        # -- WHITESPACE after $ --
        # {'type': 'UNKNOWN', 'value': 'স', 'line': 2, 'col': 3},
        # {'type': 'UNKNOWN', 'value': 'ম', 'line': 2, 'col': 4},
        # ...and so on for other unicode chars if they are treated individually.
        # The current lexer's UNKNOWN token consumes one character at a time.
        
        unknown_token_1 = next((t for t in tokens if t['type'] == 'UNKNOWN' and t['value'] == '$'), None)
        self.assertIsNotNone(unknown_token_1, "Expected UNKNOWN token for '$'")
        self.assertEqual(unknown_token_1['line'], 2)
        self.assertEqual(unknown_token_1['col'], 1)
        
        # Check for the first unicode character 'স'
        unknown_token_2 = next((t for t in tokens if t['type'] == 'UNKNOWN' and t['value'] == 'স'), None)
        self.assertIsNotNone(unknown_token_2, "Expected UNKNOWN token for 'স'")
        self.assertEqual(unknown_token_2['line'], 2)
        # The column for 'স' depends on how whitespace after '$' is handled.
        # Lexer's WHITESPACE token: r"[ \t]+" (excludes newline)
        # Code: "$ সমস্যা" -> '$' (col 1), ' ' (col 2, skipped), 'স' (col 3)
        self.assertEqual(unknown_token_2['col'], 3)

if __name__ == '__main__':
    unittest.main()

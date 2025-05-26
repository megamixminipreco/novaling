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

if __name__ == '__main__':
    unittest.main()

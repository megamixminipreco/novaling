# tests/test_parser.py
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../neuralscript-core')))
from lexer.lexer import tokenize
from parser.parser import Parser, ASTNode # Ensure ASTNode is importable or defined

# Helper to compare ASTs (simplified version from prompt)
def compare_ast(node1, node2):
    if node1 is None and node2 is None: return True
    if node1 is None or node2 is None: return False
    if node1.type != node2.type: return False
    if node1.value != node2.value: return False # For literals/identifiers
    
    # Compare specific attributes relevant to nodes
    for attr in ['name', 'operator', 'callee']: # Attributes from prompt's helper
        if hasattr(node1, attr) or hasattr(node2, attr): # Check if attr exists on either
            if getattr(node1, attr, None) != getattr(node2, attr, None):
                return False
                
    if len(node1.children) != len(node2.children): return False
    for c1, c2 in zip(node1.children, node2.children):
        if not compare_ast(c1, c2): return False
    return True

class TestParser(unittest.TestCase):

    def test_simple_assignment(self):
        code = "x = 10;"
        tokens = tokenize(code)
        ast = Parser(tokens).parse()
        
        self.assertEqual(ast.type, 'Program')
        self.assertEqual(len(ast.children), 1)
        assign_node = ast.children[0]
        self.assertEqual(assign_node.type, 'Assignment')
        self.assertEqual(assign_node.name, 'x') # Parser uses 'name' for target
        self.assertEqual(assign_node.children[0].type, 'NumberLiteral')
        self.assertEqual(assign_node.children[0].value, 10.0)

    def test_neural_operation_statement(self):
        code = "ATTEND q, k, v;"
        tokens = tokenize(code)
        ast = Parser(tokens).parse() 
        
        self.assertEqual(ast.type, 'Program')
        self.assertEqual(len(ast.children), 1) # ProgramNode([NeuralOpNode(...)])
        op_node = ast.children[0]
        self.assertEqual(op_node.type, 'NeuralOperation')
        self.assertEqual(op_node.name, 'ATTEND') # name attribute for op name
        self.assertEqual(len(op_node.children), 3)
        self.assertEqual(op_node.children[0].name, 'q') # children are IdentifierNodes

    def test_if_statement(self):
        code = "IF x == 1 { y = 2; } ELSE { y = 3; }"
        tokens = tokenize(code)
        ast = Parser(tokens).parse()
        
        self.assertEqual(ast.type, 'Program')
        self.assertEqual(len(ast.children), 1)
        if_node = ast.children[0]
        self.assertEqual(if_node.type, 'If')
        self.assertTrue(hasattr(if_node, 'has_else') and if_node.has_else)
        self.assertEqual(if_node.children[0].type, 'BinaryOperation') # Condition
        self.assertEqual(if_node.children[1].type, 'Program') # Then branch
        self.assertEqual(if_node.children[2].type, 'Program') # Else branch

    def test_function_definition(self):
        code = "DEF my_func(a: VECTOR) -> SCALAR { RETURN a; }"
        tokens = tokenize(code)
        ast = Parser(tokens).parse()

        self.assertEqual(ast.type, 'Program')
        self.assertEqual(len(ast.children), 1)
        func_def_node = ast.children[0]
        self.assertEqual(func_def_node.type, 'FunctionDef')
        self.assertEqual(func_def_node.name, 'my_func')
        self.assertEqual(len(func_def_node.params), 1)
        param_node = func_def_node.params[0] # params is a list of IdentifierNodes
        self.assertEqual(param_node.name, 'a')
        self.assertEqual(param_node.type_hint, 'VECTOR') # type_hint stored on IdentifierNode
        self.assertEqual(func_def_node.return_type, 'SCALAR') # return_type stored on FunctionDefNode
        
        # Body of FunctionDefNode is a ProgramNode which contains the ReturnNode
        body_program_node = func_def_node.children[0]
        self.assertEqual(body_program_node.type, 'Program')
        self.assertEqual(len(body_program_node.children), 1)
        return_node = body_program_node.children[0]
        self.assertEqual(return_node.type, 'Return') 
        self.assertEqual(return_node.children[0].name, 'a') # ReturnNode's child is IdentifierNode 'a'

    def test_parser_error_missing_semicolon(self):
        # Expected error: "Line 1:Col 7: Expected token type SEMICOLON, got type IDENTIFIER (value: 'y')"
        # The parse_statement for "x = 10" will consume "x", "=", "10".
        # Then it will expect a SEMICOLON. current_token will be 'y'.
        code = "x = 10 y = 20;"
        tokens = tokenize(code) # Lexer output is fine
        parser = Parser(tokens)
        with self.assertRaisesRegex(SyntaxError, r"Line 1:Col 8: Expected token type SEMICOLON, got type IDENTIFIER"):
            parser.parse()

    def test_parser_error_unexpected_token_in_expression(self):
        # Expected error: "Line 1:Col 8: Unexpected token '%' (type: UNKNOWN) when expecting an expression atom."
        code = "x = 10 + % 5;" 
        tokens = tokenize(code) # Lexer will make % an UNKNOWN token
        parser = Parser(tokens)
        with self.assertRaisesRegex(SyntaxError, r"Line 1:Col 8: Unexpected token '%' \(type: UNKNOWN\) when expecting an expression atom."):
            parser.parse()
            
    def test_parser_error_unexpected_eof(self):
        # Expected: "Line 1:Col 18: Unexpected end of input. Expected 'RBRACE'."
        code = "DEF my_func() {" # Missing closing RBRACE for block and function
        tokens = tokenize(code)
        parser = Parser(tokens)
        with self.assertRaisesRegex(SyntaxError, r"Line 1:Col \d+: Unexpected end of input. Expected 'RBRACE'"):
            parser.parse()

    def test_parser_error_mismatched_delimiter(self):
        # Expected: "Line 1:Col 24: Expected COMMA or RBRACKET in vector literal, got RPAREN"
        code = "my_vector = VECTOR [1, 2, 3);" 
        tokens = tokenize(code)
        parser = Parser(tokens)
        with self.assertRaisesRegex(SyntaxError, r"Line 1:Col \d+: Expected COMMA or RBRACKET in vector literal, got RPAREN"):
            parser.parse()

if __name__ == '__main__':
    unittest.main()

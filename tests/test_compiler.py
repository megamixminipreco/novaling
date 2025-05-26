# tests/test_compiler.py
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../neuralscript-core')))
from lexer.lexer import tokenize
from parser.parser import Parser 
from compiler.compiler import Compiler 

class TestCompiler(unittest.TestCase):

    def test_compile_assignment(self):
        code = "x = 10 + 5;"
        tokens = tokenize(code)
        ast = Parser(tokens).parse()
        bytecode = Compiler().compile(ast)
        
        expected_bytecode = [
            ('LOAD_CONST', 10.0),
            ('LOAD_CONST', 5.0),
            ('CALL_OP', '+', 2),
            ('STORE_VAR', 'x')
        ]
        self.assertEqual(bytecode, expected_bytecode)

    def test_compile_neural_op_assignment(self):
        # Assuming q, k, v are defined elsewhere or compiler makes them symbolic
        code = "result = ATTEND q_var, k_var, v_var;"
        tokens = tokenize(code)
        ast = Parser(tokens).parse()
        bytecode = Compiler().compile(ast)
        
        # Note: The compiler's NEURAL_OP bytecode includes resolved arg names/values
        expected_bytecode = [
            ('LOAD_VAR', 'q_var'),
            ('LOAD_VAR', 'k_var'),
            ('LOAD_VAR', 'v_var'),
            ('NEURAL_OP', 'ATTEND', ['q_var', 'k_var', 'v_var']), # Or actual values if literals
            ('STORE_VAR', 'result')
        ]
        self.assertEqual(bytecode, expected_bytecode)

    def test_compile_if_statement(self):
        code = "IF x == 10 { y = 1; } ELSE { y = 0; }"
        # Compiler generates labels like L1, L2, etc. These are hard to predict exactly
        # So, check structure rather than exact label names.
        tokens = tokenize(code)
        ast = Parser(tokens).parse()
        compiler = Compiler() # Instantiate to access labels if needed, though not strictly for this test structure
        bytecode = compiler.compile(ast)
        
        # Expected structure:
        # LOAD_VAR x, LOAD_CONST 10, CALL_OP ==
        # JUMP_IF_FALSE to else_label
        # then_branch_bytecode (LOAD_CONST 1, STORE_VAR y)
        # JUMP to end_label
        # LABEL else_label
        # else_branch_bytecode (LOAD_CONST 0, STORE_VAR y)
        # LABEL end_label
        
        self.assertIn(('LOAD_VAR', 'x'), bytecode)
        # The label name is dynamic, so we check for the instruction type and its purpose.
        # Find the JUMP_IF_FALSE instruction and its target label.
        jump_if_false_instr = next((instr for instr in bytecode if instr[0] == 'JUMP_IF_FALSE'), None)
        self.assertIsNotNone(jump_if_false_instr, "JUMP_IF_FALSE instruction not found")
        
        self.assertIn(('STORE_VAR', 'y'), bytecode) # y is stored in both branches
        # More detailed structural asserts would be needed for full validation
        # For example, checking the sequence of operations or presence of JUMP over ELSE.
        jump_over_else_instr = next((instr for instr in bytecode if instr[0] == 'JUMP'), None)
        self.assertIsNotNone(jump_over_else_instr, "JUMP instruction (to skip else) not found")

        # Check for labels (at least two for if/else/endif)
        label_count = sum(1 for instr in bytecode if instr[0] == 'LABEL')
        self.assertGreaterEqual(label_count, 2, "Not enough LABEL instructions for if/else structure")


if __name__ == '__main__':
    unittest.main()

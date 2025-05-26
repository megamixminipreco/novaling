# tests/test_runtime.py
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../neuralscript-core')))
from runtime.runtime import Runtime
# For runtime tests, we often need bytecode. We can use the compiler or pre-cooked bytecode.
from compiler.compiler import Compiler
from lexer.lexer import tokenize
from parser.parser import Parser


class TestRuntime(unittest.TestCase):

    def setUp(self):
        self.compiler = Compiler()
        self.runtime = Runtime()

    def _compile_and_run(self, code_string, clear_vars=True):
        if clear_vars:
            self.runtime.variables = {} # Clear global scope for each test run
            self.runtime.stack = []     # Clear stack
            self.runtime.call_stack = [] # Clear call stack
            self.runtime.pc = 0
            self.runtime.bytecode = []
            self.runtime.functions = {}
            self.runtime.labels = {}
            # Re-register stdlib functions if they were cleared or if it's a fresh runtime for test
            # This ensures stdlib is available if runtime was cleared or is new for each test method
            if hasattr(self.runtime, 'stdlib_functions'):
                 try:
                    from stdlib.vector_operations import STD_LIB_VECTOR_FUNCTIONS
                    if STD_LIB_VECTOR_FUNCTIONS: # Check if it was imported successfully
                        self.runtime.register_stdlib_module(STD_LIB_VECTOR_FUNCTIONS)
                 except ImportError:
                    # This case should ideally not happen if runtime.py handles its own import fallbacks
                    # print("TestRuntime: Failed to re-import STD_LIB_VECTOR_FUNCTIONS for setup.")
                    pass 
        
        tokens = tokenize(code_string)
        ast = Parser(tokens).parse()
        bytecode = self.compiler.compile(ast)
        # print(f"\nTesting code: '''{code_string}'''\nBytecode: {bytecode}")
        return self.runtime.execute(bytecode)

    def test_runtime_assignment_and_expression(self):
        code = "x = (10 + 5) * 2;" # x = 30
        self._compile_and_run(code)
        self.assertEqual(self.runtime.variables.get('x'), 30)

    def test_runtime_vector_ops_and_stdlib_call(self):
        code = """
        v1 = VECTOR [1.0, 2.0, 3.0];
        v2 = VECTOR [4.0, 5.0, 6.0];
        dot_p = vector_dot(v1, v2); // Stdlib call
        """
        self._compile_and_run(code)
        self.assertTrue(np.array_equal(self.runtime.variables.get('v1'), np.array([1.,2.,3.])))
        self.assertTrue(np.array_equal(self.runtime.variables.get('v2'), np.array([4.,5.,6.])))
        self.assertEqual(self.runtime.variables.get('dot_p'), 32.0)

    def test_runtime_if_statement(self):
        code_true = "cond = TRUE; IF cond == TRUE { res = 10; } ELSE { res = 20; }"
        self._compile_and_run(code_true)
        self.assertEqual(self.runtime.variables.get('res'), 10)
        
        code_false = "cond = FALSE; IF cond == TRUE { res = 10; } ELSE { res = 20; }"
        self._compile_and_run(code_false)
        self.assertEqual(self.runtime.variables.get('res'), 20)
        
    def test_runtime_function_call(self):
        # Current compiler does not add implicit RETURN_VOID for functions not ending in RETURN.
        # Runtime's visit_Return handles RETURN_VOID if no expr, RETURN_VALUE if expr.
        # The compiler's FunctionDef does not add implicit return.
        # The test case expects a value, so function must RETURN.
        code = """
        DEF my_add(a, b) -> NUMBER {
            RETURN a + b;
        }
        c = my_add(15, 25); // c = 40
        """
        self._compile_and_run(code)
        self.assertEqual(self.runtime.variables.get('c'), 40)

    def test_runtime_neural_op_placeholder(self):
        code = """
        q = VECTOR [1]; k = VECTOR [2]; v = VECTOR [3];
        att_res = ATTEND q, k, v; 
        """
        self._compile_and_run(code)
        # Check if the placeholder result is stored
        self.assertIn('att_res', self.runtime.variables)
        # The exact mock result depends on runtime's NEURAL_OP placeholder
        # The compiler provides ['q', 'k', 'v'] as arg_names_or_values
        # The runtime resolves these to np.array([1]), np.array([2]), np.array([3])
        # The placeholder in runtime.py is: result = np.array([np.sum(arg) for arg in resolved_args])
        # So, result = np.array([1.0, 2.0, 3.0])
        expected_mock_result = np.array([1.0, 2.0, 3.0]) # Based on current runtime placeholder
        actual_result = self.runtime.variables.get('att_res')

        self.assertTrue(isinstance(actual_result, np.ndarray), f"Expected ndarray, got {type(actual_result)}")
        self.assertTrue(np.array_equal(actual_result, expected_mock_result), f"Expected {expected_mock_result}, got {actual_result}")


if __name__ == '__main__':
    unittest.main()

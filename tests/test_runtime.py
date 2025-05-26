# tests/test_runtime.py
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../neuralscript-core')))
from runtime.runtime import Runtime, RuntimeErrorNS # Ensure RuntimeErrorNS is imported if used in this file
# For runtime tests, we often need bytecode. We can use the compiler or pre-cooked bytecode.
# The minimal CompilerStub is defined in runtime.py __main__ for its tests,
# but for these tests, we will define a similar one or use pre-defined bytecode.

# Assuming ASTNode and CompilerStub are defined as in runtime.py's __main__ for testing ease
# If not, they need to be imported or defined here.
# For this subtask, we'll use manually crafted bytecode more than full compilation for stdlib tests.

# --- ASTNode definition for __main__ (should match one used by Parser if importing ASTs) ---
class ASTNode: # Minimal ASTNode for test setup
    def __init__(self, node_type, children=None, value=None, **kwargs):
        self.type = node_type; self.value = value
        self.children = children if children is not None else []
        self.__dict__.update(kwargs)
        self.line = kwargs.get('line', -1) 
        self.col = kwargs.get('col', -1)   
        if node_type == 'NeuralOperation' and 'arguments' in kwargs: 
            self.children = kwargs['arguments'] or []


# --- Minimal Compiler for generating test bytecode ---
class CompilerStub: # Simplified compiler for test cases
    def __init__(self): self.bytecode = []; self.label_count = 0
    def _generate_label(self, prefix="L"): self.label_count += 1; return f"{prefix}{self.label_count}"
    def compile(self, node): # node is ProgramNode
        self.bytecode = []
        if node and node.children:
            for stmt_node in node.children: self.visit(stmt_node)
        return self.bytecode
    def visit(self, node): # Simplified dispatcher
        method_name = f'visit_{node.type}'; getattr(self, method_name, self.generic_visit)(node)
    def generic_visit(self, node): raise NotImplementedError(f"MiniCompiler for test cannot visit {node.type}")
    def visit_Assignment(self, node): self.visit(node.children[0]); self.bytecode.append(('STORE_VAR', node.name))
    def visit_NumberLiteral(self, node): self.bytecode.append(('LOAD_CONST', node.value))
    def visit_VectorLiteral(self, node): 
        for el in node.children: self.visit(el)
        self.bytecode.append(('MAKE_VECTOR', len(node.children)))
    def visit_Identifier(self, node): self.bytecode.append(('LOAD_VAR', node.name))
    def visit_Call(self, node):
        for arg in node.children: self.visit(arg)
        self.bytecode.append(('CALL_FUNC', node.callee, len(node.children)))
    def visit_StringLiteral(self, node): self.bytecode.append(('LOAD_CONST', node.value))


# AST Node factories for __main__ tests
def ProgramNode(stmts): return ASTNode('Program', children=stmts)
def AssignmentNode(name, expr): return ASTNode('Assignment', children=[expr], name=name)
def IdentifierNode(name): return ASTNode('Identifier', name=name)
def NumberLiteralNode(val): return ASTNode('NumberLiteral', value=val)
def VectorLiteralNode(elements): return ASTNode('VectorLiteral', children=elements)
def StringLiteralNode(val): return ASTNode('StringLiteral', value=val)
def CallNode(callee, args): return ASTNode('Call', children=args, callee=callee)


class TestRuntime(unittest.TestCase):

    def setUp(self):
        self.compiler = CompilerStub() # Use the local stub for tests
        self.runtime = Runtime()
        # Ensure stdlib functions are registered for each test run if runtime is reset
        # This is now handled by Runtime.__init__ itself.

    def _compile_and_run(self, ast_program_node, clear_vars=True):
        if clear_vars: # Reset runtime state for isolated tests
            self.runtime.variables = {} 
            self.runtime.stack = []     
            self.runtime.call_stack = [] 
            self.runtime.pc = 0
            # self.runtime.bytecode = [] # execute will set this
            # self.runtime.functions = {} # _preprocess will set this
            # self.runtime.labels = {}    # _preprocess will set this
        
        bytecode = self.compiler.compile(ast_program_node)
        # print(f"\nExecuting Bytecode: {bytecode}")
        return self.runtime.execute(bytecode)

    def test_runtime_stdlib_vector_ops(self):
        # This test combines setup and call for vector_dot
        self.runtime.variables = {} # Explicitly clear for this test line
        ast_vector_dot = ProgramNode([
            AssignmentNode(name='v1', value_expr=VectorLiteralNode([
                NumberLiteralNode(1.0), NumberLiteralNode(2.0), NumberLiteralNode(3.0)
            ])),
            AssignmentNode(name='v2', value_expr=VectorLiteralNode([
                NumberLiteralNode(4.0), NumberLiteralNode(5.0), NumberLiteralNode(6.0)
            ])),
            AssignmentNode(name='dot_p', value_expr=CallNode(
                callee="vector_dot", 
                args=[IdentifierNode('v1'), IdentifierNode('v2')]
            ))
        ])
        self._compile_and_run(ast_vector_dot, clear_vars=False) # Keep v1,v2 in self.runtime.variables
        self.assertEqual(self.runtime.variables.get('dot_p'), 32.0)
        print("Test 'test_runtime_stdlib_vector_ops' passed.")


    def test_runtime_stdlib_attention_placeholder(self):
        self.runtime.variables = {} # Clear for test
        self.runtime.stack = []
        
        # Setup vectors
        ast_attn_setup = ProgramNode([
            AssignmentNode(name='q_vec', value_expr=VectorLiteralNode([NumberLiteralNode(1.0), NumberLiteralNode(0.5)])),
            AssignmentNode(name='k_vec', value_expr=VectorLiteralNode([NumberLiteralNode(0.7), NumberLiteralNode(1.2)])),
            AssignmentNode(name='v_vec', value_expr=VectorLiteralNode([NumberLiteralNode(1.0), NumberLiteralNode(1.0)]))
        ])
        self._compile_and_run(ast_attn_setup, clear_vars=False)

        # Call scaled_dot_product_attention
        ast_sdpa_call = ProgramNode([
             AssignmentNode(name='sdpa_out_tuple', value_expr=CallNode(
                callee="scaled_dot_product_attention",
                args=[IdentifierNode('q_vec'), IdentifierNode('k_vec'), IdentifierNode('v_vec')]
            ))
        ])
        self._compile_and_run(ast_sdpa_call, clear_vars=False)
        
        self.assertIn('sdpa_out_tuple', self.runtime.variables)
        sdpa_output, sdpa_weights = self.runtime.variables['sdpa_out_tuple']
        self.assertIsInstance(sdpa_output, np.ndarray)
        self.assertIsInstance(sdpa_weights, np.ndarray)
        self.assertEqual(sdpa_output.shape, (2,)) # Based on 1D mock behavior
        self.assertEqual(sdpa_weights.shape, (2,2))

        # Test multi_head_attention placeholder
        self.runtime.variables['q_mha'] = np.array([[1.0,2.0,3.0,4.0],[1.1,2.1,3.1,4.1]]) 
        self.runtime.variables['k_mha'] = np.array([[0.1,0.2,0.3,0.4],[0.5,0.6,0.7,0.8]])
        self.runtime.variables['v_mha'] = np.array([[1.0,1.0,1.0,1.0],[2.0,2.0,2.0,2.0]])

        ast_mha_call = ProgramNode([
            AssignmentNode(name='mha_res', value_expr=CallNode(
                callee="multi_head_attention",
                args=[
                    IdentifierNode('q_mha'), IdentifierNode('k_mha'), IdentifierNode('v_mha'),
                    NumberLiteralNode(2), # num_heads
                    NumberLiteralNode(4)  # d_model
                ]
            ))
        ])
        self._compile_and_run(ast_mha_call, clear_vars=False)
        self.assertIn('mha_res', self.runtime.variables)
        mha_output = self.runtime.variables['mha_res']
        self.assertIsInstance(mha_output, np.ndarray)
        self.assertEqual(mha_output.shape, (2,4)) 
        print("Test 'test_runtime_stdlib_attention_placeholder' passed.")

    def test_runtime_stdlib_distributions_placeholder(self):
        self.runtime.variables = {} # Clear for test
        self.runtime.stack = []

        # 1. Create Normal Distribution
        # norm_dist = create_normal(VECTOR [0.0, 0.0], VECTOR [1.0, 1.0]);
        ast_create_normal = ProgramNode([
            AssignmentNode(name='mean_vec', value_expr=VectorLiteralNode([NumberLiteralNode(0.0), NumberLiteralNode(0.0)])),
            AssignmentNode(name='std_vec', value_expr=VectorLiteralNode([NumberLiteralNode(1.0), NumberLiteralNode(1.0)])),
            AssignmentNode(name='norm_dist', value_expr=CallNode(
                callee="create_normal",
                args=[IdentifierNode('mean_vec'), IdentifierNode('std_vec')]
            ))
        ])
        self._compile_and_run(ast_create_normal, clear_vars=False)
        self.assertIn('norm_dist', self.runtime.variables)
        self.assertTrue(hasattr(self.runtime.variables['norm_dist'], 'sample'))

        # 2. Sample from Normal Distribution
        # normal_sample = sample(norm_dist);
        ast_sample_normal = ProgramNode([
            AssignmentNode(name='normal_sample', value_expr=CallNode(
                callee="sample", args=[IdentifierNode('norm_dist')]
            ))
        ])
        self._compile_and_run(ast_sample_normal, clear_vars=False)
        self.assertIn('normal_sample', self.runtime.variables)
        self.assertIsInstance(self.runtime.variables['normal_sample'], np.ndarray)
        self.assertEqual(self.runtime.variables['normal_sample'].shape, (2,))

        # 3. Log probability for Normal Distribution
        # normal_logp = log_prob(norm_dist, normal_sample);
        ast_logp_normal = ProgramNode([
            AssignmentNode(name='normal_logp', value_expr=CallNode(
                callee="log_prob", args=[IdentifierNode('norm_dist'), IdentifierNode('normal_sample')]
            ))
        ])
        self._compile_and_run(ast_logp_normal, clear_vars=False)
        self.assertIn('normal_logp', self.runtime.variables)
        self.assertIsInstance(self.runtime.variables['normal_logp'], float) # sum of log_probs

        # 4. Create Categorical Distribution
        # cat_dist = create_categorical(VECTOR [0.1, 0.8, 0.1]);
        ast_create_cat = ProgramNode([
            AssignmentNode(name='logits_vec', value_expr=VectorLiteralNode([
                NumberLiteralNode(0.1), NumberLiteralNode(0.8), NumberLiteralNode(0.1)
            ])),
            AssignmentNode(name='cat_dist', value_expr=CallNode(
                callee="create_categorical", args=[IdentifierNode('logits_vec')]
            ))
        ])
        self._compile_and_run(ast_create_cat, clear_vars=False)
        self.assertIn('cat_dist', self.runtime.variables)
        self.assertTrue(hasattr(self.runtime.variables['cat_dist'], 'sample'))

        # 5. Sample from Categorical Distribution
        # cat_sample = sample(cat_dist);
        ast_sample_cat = ProgramNode([
            AssignmentNode(name='cat_sample', value_expr=CallNode(
                callee="sample", args=[IdentifierNode('cat_dist')]
            ))
        ])
        self._compile_and_run(ast_sample_cat, clear_vars=False)
        self.assertIn('cat_sample', self.runtime.variables)
        self.assertIsInstance(self.runtime.variables['cat_sample'], (int, np.integer)) # Sampled category index

        # 6. Log probability for Categorical Distribution
        # cat_logp = log_prob(cat_dist, cat_sample);
        ast_logp_cat = ProgramNode([
             AssignmentNode(name='cat_logp', value_expr=CallNode(
                callee="log_prob", args=[IdentifierNode('cat_dist'), IdentifierNode('cat_sample')]
            ))
        ])
        self._compile_and_run(ast_logp_cat, clear_vars=False)
        self.assertIn('cat_logp', self.runtime.variables)
        self.assertIsInstance(self.runtime.variables['cat_logp'], float)
        print("Test 'test_runtime_stdlib_distributions_placeholder' passed.")

    def test_runtime_stdlib_gradient_flow_placeholders(self):
        self.runtime.variables = {} # Clear for test
        self.runtime.stack = []

        # 1. tape1 = start_tape();
        ast_start_tape = ProgramNode([AssignmentNode(name='tape1', value_expr=CallNode(callee="start_tape", args=[]))])
        self._compile_and_run(ast_start_tape, clear_vars=False)
        self.assertIn('tape1', self.runtime.variables)
        self.assertTrue(hasattr(self.runtime.variables['tape1'], 'watch'))

        # 2. my_param = VECTOR [1.0, 2.0];
        ast_param_setup = ProgramNode([
            AssignmentNode(name='my_param', value_expr=VectorLiteralNode([NumberLiteralNode(1.0), NumberLiteralNode(2.0)]))
        ])
        self._compile_and_run(ast_param_setup, clear_vars=False)
        self.assertIn('my_param', self.runtime.variables)

        # 3. watch_variable(tape1, "my_param");
        ast_watch = ProgramNode([
            CallNode(callee="watch_variable", args=[IdentifierNode('tape1'), StringLiteralNode("my_param")])
        ])
        self._compile_and_run(ast_watch, clear_vars=False)
        self.assertIn('my_param', self.runtime.variables['tape1'].watched_variables)

        # 4. grads = compute_gradients(tape1, "loss_var", ["my_param"]);
        self.runtime.variables['loss_var'] = 0.5 # Mock target variable
        
        # Manually create bytecode for the list argument to compute_gradients
        # This is a hack because the CompilerStub doesn't support list literals directly
        compute_grads_bc = [
            ('LOAD_VAR', 'tape1'),
            ('LOAD_CONST', 'loss_var'), # target_variable_name (string)
            ('LOAD_CONST', ['my_param']), # source_variable_names (Python list)
            ('CALL_FUNC', 'compute_gradients', 3),
            ('STORE_VAR', 'grads')
        ]
        self.runtime.execute(compute_grads_bc) # Execute manually crafted bytecode
        self.assertIn('grads', self.runtime.variables)
        self.assertIn('my_param', self.runtime.variables['grads'])
        self.assertIsInstance(self.runtime.variables['grads']['my_param'], np.ndarray)

        # 5. updated_params = apply_gradients({"my_param": my_param}, grads);
        params_dict_val = {'my_param': self.runtime.variables['my_param']}
        self.runtime.variables['params_for_apply'] = params_dict_val # Store Python dict

        apply_grads_bc = [
            ('LOAD_VAR', 'params_for_apply'),
            ('LOAD_VAR', 'grads'),
            ('CALL_FUNC', 'apply_gradients', 2),
            ('STORE_VAR', 'updated_p')
        ]
        self.runtime.execute(apply_grads_bc)
        self.assertIn('updated_p', self.runtime.variables)
        self.assertIn('my_param', self.runtime.variables['updated_p'])
        self.assertTrue(np.all(self.runtime.variables['updated_p']['my_param'] < params_dict_val['my_param']))

        # 6. stop_tape(tape1);
        ast_stop_tape = ProgramNode([CallNode(callee="stop_tape", args=[IdentifierNode('tape1')])])
        self._compile_and_run(ast_stop_tape, clear_vars=False) # clear_vars=False to check tape1 status
        self.assertFalse(self.runtime.variables['tape1'].active)
        print("Test 'test_runtime_stdlib_gradient_flow_placeholders' passed.")


if __name__ == '__main__':
    unittest.main()

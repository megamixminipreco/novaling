# neuralscript-core/runtime/runtime.py
import numpy as np # Using numpy for vector operations

class Runtime:
    def __init__(self):
        self.variables = {} # To store variables like vectors

    def visit(self, node):
        method_name = f'visit_{node.type}'
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise Exception(f'No visit_{node.type} method defined')

    def visit_NumberLiteral(self, node):
        return float(node.value) # Or int, depending on desired precision

    def visit_VectorLiteral(self, node):
        # Assuming children of VectorLiteral are NumberLiteral nodes
        return np.array([self.visit(child) for child in node.children])

    def visit_Assignment(self, node):
        variable_name = node.value
        # The first child of an Assignment node is the value to be assigned
        # (e.g., a VectorLiteral node)
        if node.children:
            value_node = node.children[0]
            self.variables[variable_name] = self.visit(value_node)
            print(f"Assigned {variable_name} = {self.variables[variable_name]}")
            return self.variables[variable_name]
        else:
            raise ValueError("Assignment node has no children to assign")

    def visit_Program(self, node):
        results = []
        for child_node in node.children:
            results.append(self.visit(child_node))
        return results

    # Placeholder for operations - this is where vector_add would go
    # For a full system, operations would likely be AST nodes themselves
    # e.g., AddOperation(left_operand, right_operand)

    def execute_operation(self, operation_name, *args):
        if operation_name == "vector_add":
            if len(args) != 2:
                raise ValueError("vector_add requires two arguments")
            # Basic type checking (can be expanded)
            if not (isinstance(args[0], np.ndarray) and isinstance(args[1], np.ndarray)):
                raise TypeError("Arguments for vector_add must be numpy arrays")
            return np.add(args[0], args[1])
        else:
            raise NotImplementedError(f"Operation '{operation_name}' not implemented.")

if __name__ == '__main__':
    # This is a manual AST construction for testing the runtime
    # In a real scenario, this AST would come from the parser

    # Simulating AST for: vec1 = VECTOR [1, 2, 3]
    ast_vec1_assignment = {
        "type": "Assignment",
        "value": "vec1",
        "children": [{
            "type": "VectorLiteral",
            "children": [
                {"type": "NumberLiteral", "value": "1"},
                {"type": "NumberLiteral", "value": "2"},
                {"type": "NumberLiteral", "value": "3"}
            ]
        }]
    }
    
    # Simulating AST for: vec2 = VECTOR [4, 5, 6]
    ast_vec2_assignment = {
        "type": "Assignment",
        "value": "vec2",
        "children": [{
            "type": "VectorLiteral",
            "children": [
                {"type": "NumberLiteral", "value": "4"},
                {"type": "NumberLiteral", "value": "5"},
                {"type": "NumberLiteral", "value": "6"}
            ]
        }]
    }

    # A simple way to use the Runtime with manually created "AST-like" dicts
    # We need to convert these dicts to ASTNode instances for the current Runtime
    
    # Re-using ASTNode definition (ideally imported or defined in a shared place)
    class ASTNode:
        def __init__(self, type, children=None, value=None):
            self.type = type
            self.value = value
            self.children = children if children is not None else []

        def __repr__(self):
            return f"ASTNode(type='{self.type}', value='{self.value}', children={self.children})"

    def dict_to_astnode(node_dict):
        children = []
        if "children" in node_dict and node_dict["children"] is not None:
            for child_dict in node_dict["children"]:
                children.append(dict_to_astnode(child_dict))
        return ASTNode(type=node_dict["type"], value=node_dict.get("value"), children=children)

    runtime = Runtime()

    # Execute assignments
    vec1_node = dict_to_astnode(ast_vec1_assignment)
    runtime.visit(vec1_node) 
    # Expected output: Assigned vec1 = [1. 2. 3.]
    
    vec2_node = dict_to_astnode(ast_vec2_assignment)
    runtime.visit(vec2_node)
    # Expected output: Assigned vec2 = [4. 5. 6.]

    print(f"Variables in runtime: {runtime.variables}")
    # Expected: {'vec1': array([1., 2., 3.]), 'vec2': array([4., 5., 6.])}

    # Demonstrate vector addition
    # This part is conceptual for now, as the parser doesn't yet produce operation nodes.
    # We'll call `execute_operation` directly for this demonstration.
    if 'vec1' in runtime.variables and 'vec2' in runtime.variables:
        result_add = runtime.execute_operation("vector_add", runtime.variables['vec1'], runtime.variables['vec2'])
        print(f"Result of vec1 + vec2: {result_add}")
        # Expected: Result of vec1 + vec2: [5. 7. 9.]
    else:
        print("Error: vec1 or vec2 not defined in runtime for addition test.")

    # Test with a non-implemented operation
    try:
        runtime.execute_operation("vector_multiply", runtime.variables.get('vec1'), runtime.variables.get('vec2'))
    except NotImplementedError as e:
        print(e)
        # Expected: Operation 'vector_multiply' not implemented.

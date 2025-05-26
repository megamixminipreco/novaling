# neuralscript-core/stdlib/attention_mechanisms.py
import numpy as np

def scaled_dot_product_attention(query: np.ndarray, key: np.ndarray, value: np.ndarray, mask: np.ndarray = None):
    """
    Computes scaled dot-product attention.
    Args:
        query: Query tensor (e.g., shape [batch_size, seq_len_q, depth_q])
        key: Key tensor (e.g., shape [batch_size, seq_len_k, depth_k])
        value: Value tensor (e.g., shape [batch_size, seq_len_v, depth_v]) (seq_len_k typically == seq_len_v)
        mask: Optional mask tensor.
    Returns:
        Output tensor and attention weights.
    """
    print(f"StdLib: Called scaled_dot_product_attention with Q:{query.shape}, K:{key.shape}, V:{value.shape}")
    # Placeholder: Assume depth_q == depth_k for dot product, and output has depth_v
    # Output shape: (batch_size, seq_len_q, depth_v)
    # Attention weights shape: (batch_size, seq_len_q, seq_len_k)
    
    # For mock, let's assume query, key, value are 2D [seq_len, depth] for simplicity if batch_size=1
    if query.ndim == 2 and key.ndim == 2 and value.ndim == 2:
        seq_len_q, depth_q = query.shape
        seq_len_k, depth_k = key.shape
        seq_len_v, depth_v = value.shape # seq_len_k == seq_len_v
        
        if depth_q != depth_k:
            raise ValueError(f"Depth of query ({depth_q}) and key ({depth_k}) must match for dot product.")

        # Mock output
        mock_output = np.random.rand(seq_len_q, depth_v)
        mock_weights = np.random.rand(seq_len_q, seq_len_k) # Simplified weights
        print(f"StdLib: scaled_dot_product_attention returning mock output: {mock_output.shape}, mock weights: {mock_weights.shape}")
        return mock_output, mock_weights
    else: # More general case, or just return fixed size mock for now
        print("StdLib: scaled_dot_product_attention using generic mock for complex shapes.")
        # Simplified mock for any shape, assuming batch dimension if present
        # Determine batch_size, seq_len_q, seq_len_k, depth_v from inputs
        # This generic mock needs to be robust to different ndims
        
        # Query: (B, S_q, D_q) or (S_q, D_q)
        # Key:   (B, S_k, D_k) or (S_k, D_k)
        # Value: (B, S_v, D_v) or (S_v, D_v) (assume S_k == S_v)
        # Output:(B, S_q, D_v) or (S_q, D_v)
        # AttW:  (B, S_q, S_k) or (S_q, S_k)

        q_is_batched = query.ndim == 3
        k_is_batched = key.ndim == 3
        # v_is_batched = value.ndim == 3 # Value batching usually follows key/query

        if q_is_batched != k_is_batched : # Mismatched batching for Q and K
            # This check might be too simple for broadcasting scenarios but fine for a mock.
            # Allow 1D vectors to pass here, as they are handled by the test case for 1D.
            if not (query.ndim == 1 and key.ndim == 1 and value.ndim ==1): 
                 raise ValueError(f"Query (ndim={query.ndim}) and Key (ndim={key.ndim}) batching must be consistent for non-1D inputs.")

        if query.ndim == 1 and key.ndim == 1 and value.ndim ==1: # Handle simple 1D vectors
            seq_len_q = query.shape[0]
            seq_len_k = key.shape[0]
            mock_output = np.random.rand(seq_len_q) 
            mock_weights = np.random.rand(seq_len_q, seq_len_k)
            print(f"StdLib: scaled_dot_product_attention (1D inputs) returning mock output: {mock_output.shape}, mock weights: {mock_weights.shape}")
            return mock_output, mock_weights


        batch_size = query.shape[0] if q_is_batched else 1
        seq_len_q = query.shape[1] if q_is_batched else query.shape[0]
        depth_q = query.shape[2] if q_is_batched else query.shape[1]

        seq_len_k = key.shape[1] if k_is_batched else key.shape[0]
        depth_k = key.shape[2] if k_is_batched else key.shape[1]
        
        depth_v = value.shape[-1] # Depth of value tensor is its last dimension

        if depth_q != depth_k:
             raise ValueError(f"Depth of query ({depth_q}) and key ({depth_k}) must match for dot product in generic mock.")

        # Construct mock output shapes
        if q_is_batched:
            mock_output_shape = (batch_size, seq_len_q, depth_v)
            mock_weights_shape = (batch_size, seq_len_q, seq_len_k)
        else: # Not batched
            mock_output_shape = (seq_len_q, depth_v)
            mock_weights_shape = (seq_len_q, seq_len_k)
            
        mock_output = np.random.rand(*mock_output_shape)
        mock_weights = np.random.rand(*mock_weights_shape)
        
        print(f"StdLib: scaled_dot_product_attention (generic) returning mock output: {mock_output.shape}, mock weights: {mock_weights.shape}")
        return mock_output, mock_weights


def multi_head_attention_forward(query: np.ndarray, key: np.ndarray, value: np.ndarray, 
                                 num_heads: int, d_model: int, mask: np.ndarray = None):
    """
    Computes multi-head attention.
    Args:
        query, key, value: Input tensors.
        num_heads: Number of attention heads.
        d_model: Dimensionality of the model (must be divisible by num_heads).
        mask: Optional mask.
    Returns:
        Output tensor of shape (batch_size, seq_len_q, d_model) or (seq_len_q, d_model).
    """
    print(f"StdLib: Called multi_head_attention_forward with Q:{query.shape}, K:{key.shape}, V:{value.shape}, heads:{num_heads}, d_model:{d_model}")
    if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads.")
    
    # Placeholder: Output shape (batch_size, seq_len_q, d_model) or (seq_len_q, d_model)
    if query.ndim == 2: # seq_len, depth (d_model_in)
        seq_len_q, d_model_in = query.shape
        # if d_model_in != d_model: # Input projection would handle this in a real model
        #     print(f"Warning: MHA input query depth {d_model_in} != d_model {d_model}. Mock proceeds.")
        mock_output = np.random.rand(seq_len_q, d_model)
    elif query.ndim == 3: # batch, seq_len, depth (d_model_in)
        batch_size, seq_len_q, d_model_in = query.shape
        # if d_model_in != d_model:
        #     print(f"Warning: MHA input query depth {d_model_in} != d_model {d_model}. Mock proceeds.")
        mock_output = np.random.rand(batch_size, seq_len_q, d_model)
    else:
        raise ValueError("Query input for multi_head_attention_forward must be 2D or 3D.")
        
    print(f"StdLib: multi_head_attention_forward returning mock output: {mock_output.shape}")
    return mock_output # Only output, not attention weights for this high-level func usually


STD_LIB_ATTENTION_FUNCTIONS = {
    "scaled_dot_product_attention": scaled_dot_product_attention,
    "multi_head_attention": multi_head_attention_forward,
}

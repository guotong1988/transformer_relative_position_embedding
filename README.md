# Relative Position Representations
## 调用方式
```python
# return [batch_size, sequence_len, hidden_size]
multihead_attention(
    query_antecedent=layer_input, # [batch_size, sequence_len, hidden_size]
    memory_antecedent=layer_input, # [batch_size, sequence_len, hidden_size]
    bias=attention_mask, # [batch_size, sequence_len, sequence_len]
    total_key_depth=768,
    total_value_depth=768,
    output_depth=768,
    num_heads=12,
    dropout_rate=0.1,
    max_relative_position=10)
```
## TensorFlow版本 1.14 or 1.15
## Reference
https://github.com/tensorflow/tensor2tensor

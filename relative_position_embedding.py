import tensorflow as tf
import six
from tensorflow.python.ops import inplace_ops


def _generate_relative_positions_matrix(length_q, length_k,
                                        max_relative_position,
                                        cache=False):
    """Generates matrix of relative positions between inputs."""
    if not cache:
        if length_q == length_k:
            range_vec_q = range_vec_k = tf.range(length_q)
        else:
            range_vec_k = tf.range(length_k)
            range_vec_q = range_vec_k[-length_q:]
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
    else:
        distance_mat = tf.expand_dims(tf.range(-length_k + 1, 1, 1), 0)
    distance_mat_clipped = tf.clip_by_value(distance_mat, -max_relative_position,
                                            max_relative_position)
    # Shift values to be >= 0. Each integer still uniquely identifies a relative
    # position difference.
    final_mat = distance_mat_clipped + max_relative_position
    return final_mat


def _generate_relative_positions_embeddings(length_q, length_k, depth,
                                            max_relative_position, name,
                                            cache=False):
    """Generates tensor of size [1 if cache else length_q, length_k, depth]."""
    with tf.variable_scope(name):
        relative_positions_matrix = _generate_relative_positions_matrix(
            length_q, length_k, max_relative_position, cache=cache)
        vocab_size = max_relative_position * 2 + 1
        # Generates embedding for each relative position of dimension depth.
        embeddings_table = tf.get_variable("embeddings", [vocab_size, depth])
        embeddings = tf.gather(embeddings_table, relative_positions_matrix)
        return embeddings


def dot_product_attention_relative(q,
                                   k,
                                   v,
                                   bias,
                                   max_relative_position,
                                   dropout_rate=0.0,
                                   image_shapes=None,
                                   save_weights_to=None,
                                   name=None,
                                   make_image_summary=True,
                                   cache=False,
                                   allow_memory=False,
                                   hard_attention_k=0,
                                   gumbel_noise_weight=0.0):
    """Calculate relative position-aware dot-product self-attention.
    The attention calculation is augmented with learned representations for the
    relative position between each element in q and each element in k and v.
    Args:
      q: a Tensor with shape [batch, heads, length, depth].
      k: a Tensor with shape [batch, heads, length, depth].
      v: a Tensor with shape [batch, heads, length, depth].
      bias: bias Tensor.
      max_relative_position: an integer specifying the maximum distance between
          inputs that unique position embeddings should be learned for.
      dropout_rate: a floating point number.
      image_shapes: optional tuple of integer scalars.
      save_weights_to: an optional dictionary to capture attention weights
        for visualization; the weights tensor will be appended there under
        a string key created from the variable scope (including name).
      name: an optional string.
      make_image_summary: Whether to make an attention image summary.
      cache: whether use cache mode
      allow_memory: whether to assume that recurrent memory is in use. If True,
        the length dimension of k/v/bias may be longer than the queries, and it is
        assumed that the extra memory entries precede the non-memory entries.
      hard_attention_k: integer, if > 0 triggers hard attention (picking top-k)
      gumbel_noise_weight: if > 0, apply Gumbel noise with weight
        `gumbel_noise_weight` before picking top-k. This is a no op if
        hard_attention_k <= 0.
    Returns:
      A Tensor.
    Raises:
      ValueError: if max_relative_position is not > 0.
    """
    if not max_relative_position:
        raise ValueError("Max relative position (%s) should be > 0 when using "
                         "relative self attention." % (max_relative_position))
    with tf.variable_scope(
            name, default_name="dot_product_attention_relative",
            values=[q, k, v]) as scope:

        # This calculation only works for self attention.
        # q, k and v must therefore have the same shape, unless memory is enabled.
        if not cache and not allow_memory:
            q.get_shape().assert_is_compatible_with(k.get_shape())
            q.get_shape().assert_is_compatible_with(v.get_shape())

        # Use separate embeddings suitable for keys and values.
        depth = k.get_shape().as_list()[3]
        length_k = get_shape_list(k)[2]
        length_q = get_shape_list(q)[2] if allow_memory else length_k
        relations_keys = _generate_relative_positions_embeddings(
            length_q, length_k, depth, max_relative_position,
            "relative_positions_keys", cache=cache)
        relations_values = _generate_relative_positions_embeddings(
            length_q, length_k, depth, max_relative_position,
            "relative_positions_values", cache=cache)

        # Compute self attention considering the relative position embeddings.
        logits = _relative_attention_inner(q, k, relations_keys, True)
        if bias is not None:
            logits = logits * tf.expand_dims(bias, 1)  # FIXME
        weights = tf.nn.softmax(logits, name="attention_weights")
        # if hard_attention_k > 0:
        #     weights = harden_attention_weights(weights, hard_attention_k,
        #                                        gumbel_noise_weight)
        if save_weights_to is not None:
            save_weights_to[scope.name] = weights
            save_weights_to[scope.name + "/logits"] = logits
        weights = tf.nn.dropout(weights, 1.0 - dropout_rate)
        return _relative_attention_inner(weights, v, relations_values, False)


def _relative_attention_inner(x, y, z, transpose):
    """Relative position-aware dot-product attention inner calculation.
    This batches matrix multiply calculations to avoid unnecessary broadcasting.
    Args:
      x: Tensor with shape [batch_size, heads, length or 1, length or depth].
      y: Tensor with shape [batch_size, heads, length or 1, depth].
      z: Tensor with shape [length or 1, length, depth].
      transpose: Whether to transpose inner matrices of y and z. Should be true if
          last dimension of x is depth, not length.
    Returns:
      A Tensor with shape [batch_size, heads, length, length or depth].
    """
    batch_size = tf.shape(x)[0]
    heads = x.get_shape().as_list()[1]
    length = tf.shape(x)[2]

    # xy_matmul is [batch_size, heads, length or 1, length or depth]
    xy_matmul = tf.matmul(x, y, transpose_b=transpose)
    # x_t is [length or 1, batch_size, heads, length or depth]
    x_t = tf.transpose(x, [2, 0, 1, 3])
    # x_t_r is [length or 1, batch_size * heads, length or depth]
    x_t_r = tf.reshape(x_t, [length, heads * batch_size, -1])
    # x_tz_matmul is [length or 1, batch_size * heads, length or depth]
    x_tz_matmul = tf.matmul(x_t_r, z, transpose_b=transpose)
    # x_tz_matmul_r is [length or 1, batch_size, heads, length or depth]
    x_tz_matmul_r = tf.reshape(x_tz_matmul, [length, batch_size, heads, -1])
    # x_tz_matmul_r_t is [batch_size, heads, length or 1, length or depth]
    x_tz_matmul_r_t = tf.transpose(x_tz_matmul_r, [1, 2, 0, 3])
    return xy_matmul + x_tz_matmul_r_t


def combine_last_two_dimensions(x):
    """Reshape x so that the last two dimension become one.
    Args:
      x: a Tensor with shape [..., a, b]
    Returns:
      a Tensor with shape [..., ab]
    """
    x_shape = get_shape_list(x)
    a, b = x_shape[-2:]
    return tf.reshape(x, x_shape[:-2] + [a * b])


def combine_heads(x):
    """Inverse of split_heads.
    Args:
      x: a Tensor with shape [batch, num_heads, length, channels / num_heads]
    Returns:
      a Tensor with shape [batch, length, channels]
    """
    return combine_last_two_dimensions(tf.transpose(x, [0, 2, 1, 3]))


def compute_qkv(query_antecedent,
                memory_antecedent,
                total_key_depth,
                total_value_depth,
                q_filter_width=1,
                kv_filter_width=1,
                q_padding="VALID",
                kv_padding="VALID",
                vars_3d_num_heads=0,
                layer_collection=None):
    """Computes query, key and value.
    Args:
      query_antecedent: a Tensor with shape [batch, length_q, channels]
      memory_antecedent: a Tensor with shape [batch, length_m, channels]
      total_key_depth: an integer
      total_value_depth: an integer
      q_filter_width: An integer specifying how wide you want the query to be.
      kv_filter_width: An integer specifying how wide you want the keys and values
      to be.
      q_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
      kv_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
      vars_3d_num_heads: an optional (if we want to use 3d variables)
      layer_collection: A tensorflow_kfac.LayerCollection. Only used by the
        KFAC optimizer. Default is None.
    Returns:
      q, k, v : [batch, length, depth] tensors
    """
    if memory_antecedent is None:
        memory_antecedent = query_antecedent
    q = compute_attention_component(
        query_antecedent,
        total_key_depth,
        q_filter_width,
        q_padding,
        "q",
        vars_3d_num_heads=vars_3d_num_heads,
        layer_collection=layer_collection)
    k = compute_attention_component(
        memory_antecedent,
        total_key_depth,
        kv_filter_width,
        kv_padding,
        "k",
        vars_3d_num_heads=vars_3d_num_heads,
        layer_collection=layer_collection)
    v = compute_attention_component(
        memory_antecedent,
        total_value_depth,
        kv_filter_width,
        kv_padding,
        "v",
        vars_3d_num_heads=vars_3d_num_heads,
        layer_collection=layer_collection)
    return q, k, v


def compute_attention_component(antecedent,
                                total_depth,
                                filter_width=1,
                                padding="VALID",
                                name="c",
                                vars_3d_num_heads=0,
                                layer_collection=None):
    """Computes attention component (query, key or value).
    Args:
      antecedent: a Tensor with shape [batch, length, channels]
      total_depth: an integer
      filter_width: An integer specifying how wide you want the attention
        component to be.
      padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
      name: a string specifying scope name.
      vars_3d_num_heads: an optional integer (if we want to use 3d variables)
      layer_collection: A tensorflow_kfac.LayerCollection. Only used by the
        KFAC optimizer. Default is None.
    Returns:
      c : [batch, length, depth] tensor
    """
    if layer_collection is not None:
        if filter_width != 1 or vars_3d_num_heads != 0:
            raise ValueError(
                "KFAC implementation only supports filter_width=1 (actual: {}) and "
                "vars_3d_num_heads=0 (actual: {}).".format(
                    filter_width, vars_3d_num_heads))
    if vars_3d_num_heads is not None and vars_3d_num_heads > 0:
        assert filter_width == 1
        input_depth = antecedent.get_shape().as_list()[-1]
        depth_per_head = total_depth // vars_3d_num_heads
        initializer_stddev = input_depth ** -0.5
        if "q" in name:
            initializer_stddev *= depth_per_head ** -0.5
        var = tf.get_variable(
            name, [input_depth,
                   vars_3d_num_heads,
                   total_depth // vars_3d_num_heads],
            initializer=tf.random_normal_initializer(stddev=initializer_stddev))
        var = tf.cast(var, antecedent.dtype)
        var = tf.reshape(var, [input_depth, total_depth])
        return tf.tensordot(antecedent, var, axes=1)
    if filter_width == 1:
        return tf.layers.dense(
            antecedent, total_depth, use_bias=False, name=name)
    else:
        return tf.layers.conv1d(
            antecedent, total_depth, filter_width, padding=padding, name=name)


def split_heads(x, num_heads):
    """Split channels (dimension 2) into multiple heads (becomes dimension 1).
    Args:
      x: a Tensor with shape [batch, length, channels]
      num_heads: an integer
    Returns:
      a Tensor with shape [batch, num_heads, length, channels / num_heads]
    """
    return tf.transpose(split_last_dimension(x, num_heads), [0, 2, 1, 3])


def split_last_dimension(x, n):
    """Reshape x so that the last dimension becomes two dimensions.
    The first of these two dimensions is n.
    Args:
      x: a Tensor with shape [..., m]
      n: an integer.
    Returns:
      a Tensor with shape [..., n, m/n]
    """
    x_shape = get_shape_list(x)
    m = x_shape[-1]
    if isinstance(m, int) and isinstance(n, int):
        assert m % n == 0
    return tf.reshape(x, x_shape[:-1] + [n, m // n])


def multihead_attention(query_antecedent,
                        memory_antecedent,
                        bias,
                        total_key_depth,
                        total_value_depth,
                        output_depth,
                        num_heads,
                        dropout_rate,
                        attention_type="dot_product_relative",
                        max_relative_position=None,
                        heads_share_relative_embedding=False,
                        add_relative_to_values=False,
                        image_shapes=None,
                        block_length=128,
                        block_width=128,
                        q_filter_width=1,
                        kv_filter_width=1,
                        q_padding="VALID",
                        kv_padding="VALID",
                        cache=None,
                        gap_size=0,
                        num_memory_blocks=2,
                        name="multihead_attention",
                        save_weights_to=None,
                        make_image_summary=True,
                        dropout_broadcast_dims=None,
                        vars_3d=False,
                        layer_collection=None,
                        recurrent_memory=None,
                        chunk_number=None,
                        hard_attention_k=0,
                        gumbel_noise_weight=0.0,
                        max_area_width=1,
                        max_area_height=1,
                        memory_height=1,
                        area_key_mode="mean",
                        area_value_mode="sum",
                        training=True,
                        **kwargs):
    """Multihead scaled-dot-product attention with input/output transformations.
    Args:
      query_antecedent: a Tensor with shape [batch, length_q, channels]
      memory_antecedent: a Tensor with shape [batch, length_m, channels] or None
      bias: bias Tensor (see attention_bias())
      total_key_depth: an integer
      total_value_depth: an integer
      output_depth: an integer
      num_heads: an integer dividing total_key_depth and total_value_depth
      dropout_rate: a floating point number
      attention_type: a string, either "dot_product", "dot_product_relative",
                      "local_mask_right", "local_unmasked", "masked_dilated_1d",
                      "unmasked_dilated_1d", graph, or any attention function
                      with the signature (query, key, value, **kwargs)
      max_relative_position: Maximum distance between inputs to generate
                             unique relation embeddings for. Only relevant
                             when using "dot_product_relative" attention.
      heads_share_relative_embedding: boolean to share relative embeddings
      add_relative_to_values: a boolean for whether to add relative component to
                              values.
      image_shapes: optional tuple of integer scalars.
                    see comments for attention_image_summary()
      block_length: an integer - relevant for "local_mask_right"
      block_width: an integer - relevant for "local_unmasked"
      q_filter_width: An integer specifying how wide you want the query to be.
      kv_filter_width: An integer specifying how wide you want the keys and values
                       to be.
      q_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
                 kv_padding: One of "VALID", "SAME" or "LEFT". Default is "VALID":
                 no padding.
      cache: dict containing Tensors which are the results of previous
             attentions, used for fast decoding. Expects the dict to contrain two
             keys ('k' and 'v'), for the initial call the values for these keys
             should be empty Tensors of the appropriate shape.
                 'k' [batch_size, 0, key_channels]
                 'v' [batch_size, 0, value_channels]
      gap_size: Integer option for dilated attention to indicate spacing between
                memory blocks.
      num_memory_blocks: Integer option to indicate how many memory blocks to look
                         at.
      name: an optional string.
      save_weights_to: an optional dictionary to capture attention weights
        for vizualization; the weights tensor will be appended there under
        a string key created from the variable scope (including name).
      make_image_summary: Whether to make an attention image summary.
      dropout_broadcast_dims:  an optional list of integers less than 4
        specifying in which dimensions to broadcast the dropout decisions.
        saves memory.
      vars_3d: use 3-dimensional variables for input/output transformations
      layer_collection: A tensorflow_kfac.LayerCollection. Only used by the
        KFAC optimizer. Default is None.
      recurrent_memory: An optional transformer_memory.RecurrentMemory, which
        retains state across chunks. Default is None.
      chunk_number: an optional integer Tensor with shape [batch] used to operate
        the recurrent_memory.
      hard_attention_k: integer, if > 0 triggers hard attention (picking top-k).
      gumbel_noise_weight: if > 0, apply Gumbel noise with weight
        `gumbel_noise_weight` before picking top-k. This is a no op if
        hard_attention_k <= 0.
      max_area_width: the max width allowed for an area.
      max_area_height: the max height allowed for an area.
      memory_height: the height of the memory.
      area_key_mode: the mode for computing area keys, which can be "mean",
        "concat", "sum", "sample_concat", and "sample_sum".
      area_value_mode: the mode for computing area values, which can be either
        "mean", or "sum".
      training: indicating if it is in the training mode.
      **kwargs (dict): Parameters for the attention function.
    Caching:
      WARNING: For decoder self-attention, i.e. when memory_antecedent == None,
      the caching assumes that the bias contains future masking.
      The caching works by saving all the previous key and value values so that
      you are able to send just the last query location to this attention
      function. I.e. if the cache dict is provided it assumes the query is of the
      shape [batch_size, 1, hidden_dim] rather than the full memory.
    Returns:
      The result of the attention transformation. The output shape is
          [batch_size, length_q, hidden_dim]
      unless the cache dict is provided in which case only the last memory
      position is calculated and the output shape is [batch_size, 1, hidden_dim]
      Optionally returns an additional loss parameters (ex: load balance loss for
      the experts) returned by the attention_type function.
    Raises:
      ValueError: if the key depth or value depth are not divisible by the
        number of attention heads.
    """
    if total_key_depth % num_heads != 0:
        raise ValueError("Key depth (%d) must be divisible by the number of "
                         "attention heads (%d)." % (total_key_depth, num_heads))
    if total_value_depth % num_heads != 0:
        raise ValueError("Value depth (%d) must be divisible by the number of "
                         "attention heads (%d)." % (total_value_depth, num_heads))
    vars_3d_num_heads = num_heads if vars_3d else 0

    if layer_collection is not None:
        if cache is not None:
            raise ValueError("KFAC implementation only supports cache is None.")
        if vars_3d:
            raise ValueError("KFAC implementation does not support 3d vars.")

    if recurrent_memory is not None:
        if memory_antecedent is not None:
            raise ValueError("Recurrent memory requires memory_antecedent is None.")
        if cache is not None:
            raise ValueError("Cache is not supported when using recurrent memory.")
        if vars_3d:
            raise ValueError("3d vars are not supported when using recurrent memory.")
        if layer_collection is not None:
            raise ValueError("KFAC is not supported when using recurrent memory.")
        if chunk_number is None:
            raise ValueError("chunk_number is required when using recurrent memory.")

    with tf.variable_scope(name, default_name="multihead_attention",
                           values=[query_antecedent, memory_antecedent]):

        if recurrent_memory is not None:
            (
                recurrent_memory_transaction,
                query_antecedent, memory_antecedent, bias,
            ) = recurrent_memory.pre_attention(
                chunk_number,
                query_antecedent, memory_antecedent, bias,
            )

        if cache is None or memory_antecedent is None:
            q, k, v = compute_qkv(query_antecedent, memory_antecedent,
                                  total_key_depth, total_value_depth, q_filter_width,
                                  kv_filter_width, q_padding, kv_padding,
                                  vars_3d_num_heads=vars_3d_num_heads,
                                  layer_collection=layer_collection)
        if cache is not None:
            if attention_type not in ["dot_product", "dot_product_relative"]:
                # TODO(petershaw): Support caching when using relative position
                # representations, i.e. "dot_product_relative" attention.
                raise NotImplementedError(
                    "Caching is not guaranteed to work with attention types other than"
                    " dot_product.")
            if bias is None:
                raise ValueError("Bias required for caching. See function docstring "
                                 "for details.")

            if memory_antecedent is not None:
                # Encoder-Decoder Attention Cache
                q = compute_attention_component(query_antecedent, total_key_depth,
                                                q_filter_width, q_padding, "q",
                                                vars_3d_num_heads=vars_3d_num_heads)
                k = cache["k_encdec"]
                v = cache["v_encdec"]
            else:
                k = split_heads(k, num_heads)
                v = split_heads(v, num_heads)
                decode_loop_step = kwargs.get("decode_loop_step")
                if decode_loop_step is None:
                    k = cache["k"] = tf.concat([cache["k"], k], axis=2)
                    v = cache["v"] = tf.concat([cache["v"], v], axis=2)
                else:
                    # Inplace update is required for inference on TPU.
                    # Inplace_ops only supports inplace_update on the first dimension.
                    # The performance of current implementation is better than updating
                    # the tensor by adding the result of matmul(one_hot,
                    # update_in_current_step)
                    tmp_k = tf.transpose(cache["k"], perm=[2, 0, 1, 3])
                    tmp_k = inplace_ops.alias_inplace_update(
                        tmp_k, decode_loop_step, tf.squeeze(k, axis=2))
                    k = cache["k"] = tf.transpose(tmp_k, perm=[1, 2, 0, 3])
                    tmp_v = tf.transpose(cache["v"], perm=[2, 0, 1, 3])
                    tmp_v = inplace_ops.alias_inplace_update(
                        tmp_v, decode_loop_step, tf.squeeze(v, axis=2))
                    v = cache["v"] = tf.transpose(tmp_v, perm=[1, 2, 0, 3])

        q = split_heads(q, num_heads)
        if cache is None:
            k = split_heads(k, num_heads)
            v = split_heads(v, num_heads)

        key_depth_per_head = total_key_depth // num_heads
        if not vars_3d:
            q *= key_depth_per_head ** -0.5

        additional_returned_value = None
        if callable(attention_type):  # Generic way to extend multihead_attention
            x = attention_type(q, k, v, **kwargs)
            if isinstance(x, tuple):
                x, additional_returned_value = x  # Unpack
        elif attention_type == "dot_product":
            pass
            # if max_area_width > 1 or max_area_height > 1:
            #     x = area_attention.dot_product_area_attention(
            #         q, k, v, bias, dropout_rate, image_shapes,
            #         save_weights_to=save_weights_to,
            #         dropout_broadcast_dims=dropout_broadcast_dims,
            #         max_area_width=max_area_width,
            #         max_area_height=max_area_height,
            #         memory_height=memory_height,
            #         area_key_mode=area_key_mode,
            #         area_value_mode=area_value_mode,
            #         training=training)
            # else:
            #     x = dot_product_attention(
            #         q, k, v, bias, dropout_rate, image_shapes,
            #         save_weights_to=save_weights_to,
            #         make_image_summary=make_image_summary,
            #         dropout_broadcast_dims=dropout_broadcast_dims,
            #         activation_dtype=kwargs.get("activation_dtype"),
            #         hard_attention_k=hard_attention_k,
            #         gumbel_noise_weight=gumbel_noise_weight)
        elif attention_type == "dot_product_relative":
            x = dot_product_attention_relative(
                q,
                k,
                v,
                bias,
                max_relative_position,
                dropout_rate,
                image_shapes,
                save_weights_to=save_weights_to,
                make_image_summary=make_image_summary,
                cache=cache is not None,
                allow_memory=recurrent_memory is not None,
                hard_attention_k=hard_attention_k,
                gumbel_noise_weight=gumbel_noise_weight)
        elif attention_type == "dot_product_unmasked_relative_v2":
            pass
            # x = dot_product_unmasked_self_attention_relative_v2(
            #     q,
            #     k,
            #     v,
            #     bias,
            #     max_relative_position,
            #     dropout_rate,
            #     image_shapes,
            #     save_weights_to=save_weights_to,
            #     make_image_summary=make_image_summary,
            #     dropout_broadcast_dims=dropout_broadcast_dims,
            #     heads_share_relative_embedding=heads_share_relative_embedding,
            #     add_relative_to_values=add_relative_to_values)
        elif attention_type == "dot_product_relative_v2":
            pass
            # x = dot_product_self_attention_relative_v2(
            #     q,
            #     k,
            #     v,
            #     bias,
            #     max_relative_position,
            #     dropout_rate,
            #     image_shapes,
            #     save_weights_to=save_weights_to,
            #     make_image_summary=make_image_summary,
            #     dropout_broadcast_dims=dropout_broadcast_dims,
            #     heads_share_relative_embedding=heads_share_relative_embedding,
            #     add_relative_to_values=add_relative_to_values)
        elif attention_type == "local_within_block_mask_right":
            pass
            # x = masked_within_block_local_attention_1d(
            #     q, k, v, block_length=block_length)
        elif attention_type == "local_relative_mask_right":
            pass
            # x = masked_relative_local_attention_1d(
            #     q,
            #     k,
            #     v,
            #     block_length=block_length,
            #     make_image_summary=make_image_summary,
            #     dropout_rate=dropout_rate,
            #     heads_share_relative_embedding=heads_share_relative_embedding,
            #     add_relative_to_values=add_relative_to_values,
            #     name="masked_relative_local_attention_1d")
        elif attention_type == "local_mask_right":
            pass
            # x = masked_local_attention_1d(
            #     q,
            #     k,
            #     v,
            #     block_length=block_length,
            #     make_image_summary=make_image_summary)
        elif attention_type == "local_unmasked":
            pass
            # x = local_attention_1d(
            #     q, k, v, block_length=block_length, filter_width=block_width)
        elif attention_type == "masked_dilated_1d":
            pass
            # x = masked_dilated_self_attention_1d(q, k, v, block_length, block_width,
            #                                      gap_size, num_memory_blocks)
        else:
            assert attention_type == "unmasked_dilated_1d"
            pass
            # x = dilated_self_attention_1d(q, k, v, block_length, block_width,
            #                               gap_size, num_memory_blocks)
        x = combine_heads(x)

        # Set last dim specifically.
        x.set_shape(x.shape.as_list()[:-1] + [total_value_depth])

        if vars_3d:
            o_var = tf.get_variable(
                "o", [num_heads, total_value_depth // num_heads, output_depth])
            o_var = tf.cast(o_var, x.dtype)
            o_var = tf.reshape(o_var, [total_value_depth, output_depth])
            x = tf.tensordot(x, o_var, axes=1)
        else:
            x = tf.layers.dense(
                x, output_depth, use_bias=False, name="output_transform")

        if recurrent_memory is not None:
            x = recurrent_memory.post_attention(recurrent_memory_transaction, x)
        if additional_returned_value is not None:
            return x, additional_returned_value
        return x


def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.
    Args:
      tensor: A tf.Tensor object to find the shape of.
      expected_rank: (optional) int. The expected rank of `tensor`. If this is
        specified and the `tensor` has a different rank, and exception will be
        thrown.
      name: Optional name of the tensor for the error message.
    Returns:
      A list of dimensions of the shape of tensor. All static dimensions will
      be returned as python integers, and dynamic dimensions will be returned
      as tf.Tensor scalars.
    """
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.
    Args:
      tensor: A tf.Tensor to check the rank of.
      expected_rank: Python integer or list of integers, expected rank.
      name: Optional name of the tensor for the error message.
    Raises:
      ValueError: If the expected shape doesn't match the actual shape.
    """
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))


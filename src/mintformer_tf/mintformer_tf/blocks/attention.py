import tensorflow as tf
import tensorflow_addons as tfa
from typing import Optional
from mintformer_tf.blocks.rff import RFF

def _large_compatible_negative(tensor_type):
    if tensor_type == tf.float16:
        return tf.float16.min
    return -1e9

class _MaskedSparsemax(tfa.layers.Sparsemax):

    def call(self, inputs, mask = None):
        # code from tensorflow softmax implementation 
        # https://github.com/keras-team/keras/blob/v2.12.0/keras/layers/activation/softmax.py#L28
        if mask is not None:
            # Since mask is 1.0 for positions we want to keep and 0.0 for masked
            # positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -1e.9 for masked positions.
            adder = (1.0 - tf.cast(mask, inputs.dtype)) * (
                _large_compatible_negative(inputs.dtype)
            )

            # Since we are adding it to the raw scores before the softmax, this
            # is effectively the same as removing these entirely.
            inputs += adder
        return tfa.activations.sparsemax(inputs)



class _BetaAttention(tf.keras.layers.MultiHeadAttention):
    def __init__(
        self,
        num_heads,
        key_dim,
        value_dim=None,
        dropout=0.0,
        use_bias=True,
        output_shape=None,
        attention_axes=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        beta=1.0,
        activation='sparsemax',
        **kwargs
    ) -> None:
        self.beta = beta
        self.activation = activation
        super().__init__(
            num_heads,
            key_dim,
            value_dim,
            dropout,
            use_bias,
            output_shape,
            attention_axes,
            kernel_initializer,
            bias_initializer,
            kernel_regularizer,
            bias_regularizer,
            activity_regularizer,
            kernel_constraint,
            bias_constraint,
            **kwargs
        )

    def _compute_attention(self, query, key, value, attention_mask=None, training=None):
        query = tf.multiply(query, self.beta)

        return super()._compute_attention(query, key, value, attention_mask, training)
    
    def _build_attention(self, rank):
        super()._build_attention(rank)
        if self.activation == 'sparsemax':
            self._softmax = _MaskedSparsemax()


class _BaseAttentionBlock(tf.keras.Model):
    def __init__(
        self,
        n_heads: int,
        hidden_size: int,
        dropout_probability: float,
        use_linear: bool = True,
        output_size: Optional[int] = None,
        weighted_residual: bool = False,
        beta: float = 1.0,
    ):
        """
        Attention Block.

        Parameters
        ----------
        n_heads : int
            number of attention heads
        hidden_size : int
            size of the hidden state
        dropout_probability: float
            probability of dropout
        use_linear: bool
            defines whether linear block is to be used after the attention block
        output_size: int
            defines the size of the output, if None, projects back to hidden size, by default None
        weighted_residual: bool
            controls whether residual connection has trainable weights, by default False
        """
        super().__init__()
        if not output_size:
            output_size = hidden_size
        self.mha = _BetaAttention(
            num_heads=n_heads,
            key_dim=hidden_size,
            dropout=dropout_probability,
            output_shape=(output_size,),
            beta=beta,
        )
        self.weighted_residual = weighted_residual
        if weighted_residual:
            self.res = tf.keras.layers.Dense(output_size)
        self.ln0 = tf.keras.layers.LayerNormalization()
        self.use_linear = use_linear
        self.ln1 = tf.keras.layers.LayerNormalization()
        if use_linear:
            self.rff = RFF(output_size, 1)

class SampleAttentionBlock(_BaseAttentionBlock):

    """
    Cross attention between samples and memory.
    """

    def __call__(
        self,
        X: tf.Tensor,
        memory: tf.Tensor,
        memory_mask: Optional[tf.Tensor] = None,
        training: bool = True,
    ):
        """
        forward path

        Parameters
        ----------
        X : tf.Tensor
            input, should have a shape of [n_samples, n_features * embedding_size]
        memory : tf.Tensor
            memory, should have a shape of [1, n_samples_memory, n_features * embedding_size]
        memory_mask : Optional[tf.Tensor], optional
            mask to prevent certain lookups, should have a shape of [1, n_samples, n_samples_memory], by default None
        training : bool, optional
            operates in training mode if True, by default True
        """
        X_exp = tf.expand_dims(X, axis=0)
        X_exp = self.mha(
            self.ln0(X_exp, training=training),
            self.ln0(memory, training=training),
            attention_mask=memory_mask,
            training=training,
        )
        if self.weighted_residual:
            H = self.res(X) + tf.squeeze(X_exp, 0)
        else:
            H = X + tf.squeeze(X_exp, 0)
        if self.use_linear:
            H = H + self.rff(self.ln1(H, training=training))
        # else: 
        #     H = H + self.ln1(tf.keras.activations.gelu(H), training=training)
        return H

    def get_attention_maps(self, X, memory):
        X_exp = tf.expand_dims(X, axis=0)
        X_exp, att_map = self.mha(
            self.ln0(X_exp, training=False),
            self.ln0(memory, training=False),
            attention_mask=None,
            training=False,
            return_attention_scores=True,
        )
        if self.weighted_residual:
            H = self.res(X) + tf.squeeze(X_exp, 0)
        else:
            H = X + tf.squeeze(X_exp, 0)
        if self.use_linear:
            H = H + self.rff(self.ln1(H, training=False), training=False)
        # else: 
        #     H = H + self.ln1(tf.keras.activations.gelu(H), training=False)
        return H, att_map


class AttributeAttentionBlock(_BaseAttentionBlock):
    def __call__(self, X, training=True):
        X_norm = self.ln0(X)
        X_norm = self.mha(
            X_norm,
            X_norm,
            training=training,
        )
        if self.weighted_residual:
            H = self.res(X) + X_norm
        else:
            H = X + X_norm

        if self.use_linear:
            H = H + self.rff(self.ln1(H, training=training))
        # else: 
        #     H = H + self.ln1(tf.keras.activations.gelu(H), training=training)
        return H


class TransformerBlock(tf.keras.Model):
    def __init__(
        self,
        n_heads: int,
        n_features: int,
        hidden_size: int,
        dropout_probability: float,
        use_linear: bool = True,
        compression_factor_samples: Optional[int] = None,
        compression_factor_attributes: int = 2,
        weighted_residual_samples: bool = False,
        weigthed_residual_attributes: bool = False,
        samples_attention_beta: float = 1.0,
        attributes_attention_beta: float = 1.0,
    ):
        super().__init__()
        if not compression_factor_samples:
            compression_factor_samples = 2 * n_heads
        hidden_size_samples_samples_att = max(
            4, (hidden_size * n_features) // compression_factor_samples
        )
        hidden_size_samples_attributes_att = max(
            4, (hidden_size * n_features) // compression_factor_attributes
        )
        self.hidden_size = hidden_size  # E
        self.n_features = n_features  # D
        self.sample_att_block = SampleAttentionBlock(
            n_heads=n_heads,
            hidden_size=hidden_size_samples_samples_att,
            output_size=hidden_size * n_features,
            dropout_probability=dropout_probability,
            use_linear=use_linear,
            weighted_residual=weighted_residual_samples,
            beta=samples_attention_beta,
        )
        self.attribute_att_block = AttributeAttentionBlock(
            n_heads=n_heads,
            hidden_size=hidden_size_samples_attributes_att,
            output_size=hidden_size,
            dropout_probability=dropout_probability,
            use_linear=use_linear,
            weighted_residual=weigthed_residual_attributes,
            beta=attributes_attention_beta,
        )

    def __call__(self, X, memory, memory_mask=None, training=True):
        X = self.sample_att_block(
            X, memory=memory, memory_mask=memory_mask, training=training
        )
        X = tf.reshape(X, [-1, self.n_features, self.hidden_size])
        X = self.attribute_att_block(X, training=training)
        return X

    def get_attention_maps(self, X, memory):
        X, att_map = self.sample_att_block.get_attention_maps(X, memory=memory)
        X = tf.reshape(X, [-1, self.n_features, self.hidden_size])
        X = self.attribute_att_block(X, training=False)
        return X, att_map

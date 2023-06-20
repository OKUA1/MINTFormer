import tensorflow as tf
from typing import List

class ReconstructionBlock(tf.keras.Model):

    def __init__(self, feature_sizes: List[int], dropout_probability: float):
        super().__init__()
        self.n_features = len(feature_sizes)
        self.out_size = sum(feature_sizes) - self.n_features
        self.reconstruction_layer = tf.keras.layers.Dense(self.out_size)
        # self.reconstruction_layers = [tf.keras.layers.Dense(s-1) for s in feature_sizes]
        self.ln = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(dropout_probability)

    def __call__(self, X, training = True):
        
        X = tf.keras.activations.gelu(X)
        X = self.ln(X, training = training)
        X = self.dropout(X, training = training)
        reconstruction = self.reconstruction_layer(X, training = training)
        
        return reconstruction


    
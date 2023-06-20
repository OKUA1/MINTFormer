import tensorflow as tf 

class RFF(tf.keras.Model):
    def __init__(self, size: int, n_hidden_layers: int = 1):
        super().__init__()
        if n_hidden_layers < 1: 
            raise ValueError("n_hidden should be >= 1")

        layers = []
        for _ in range(n_hidden_layers):
            layers.append(tf.keras.layers.Dense(4*size, activation=tf.keras.activations.gelu))
        layers.append(tf.keras.layers.Dense(size, activation=tf.keras.activations.gelu))

        self.rff = tf.keras.Sequential(layers=layers)
    
    def __call__(self, X, training = True):
        return self.rff(X)
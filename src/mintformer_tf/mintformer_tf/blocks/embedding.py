import tensorflow as tf 
from typing import List, Any

class EmbeddingBlock(tf.keras.Model):

    def __init__(self, splits: List[int], feature_sizes: List[int], cat_ind: List[int], total_size: int, pos_encoding: bool, type_encoding: bool, dropout_probability: float, emb_size: int = 8) -> None:
        super().__init__()

        self.pos_encoding = pos_encoding
        self.type_encoding = type_encoding
        self.feature_sizes = feature_sizes
        self.cat_ind = cat_ind
        self.total_size = total_size
        self.n_features = len(feature_sizes)
        out_size = emb_size
        emb_size = emb_size * 4
        self.cat_mask = tf.constant([[1] if i in self.cat_ind else [0] for i in range(self.n_features)])
        self.positions = tf.constant([[i] for i in range(self.n_features)])
        self._gen_splits_tuples(splits, total_size)
        self.feature_emb_layers = [tf.keras.layers.Dense(emb_size) for _ in range(self.n_features)]
        if self.type_encoding:
            self.type_emb_layer = tf.keras.layers.Embedding(2, emb_size)
        if self.pos_encoding:
            self.pos_encoding_layer = tf.keras.layers.Embedding(self.n_features, emb_size)
        self.ln = tf.keras.layers.LayerNormalization()
        self.ln2 = tf.keras.layers.LayerNormalization()
        self.out_mapping = tf.keras.layers.Dense(out_size)
        self.dropout = tf.keras.layers.Dropout(dropout_probability)


    def _gen_splits_tuples(self, splits, total_size):
        ind = 0
        self.splits = []
        for s in splits:
            self.splits.append((ind, s))
            ind = s
        self.splits.append((ind, total_size))
    

    def __call__(self, X, training = True):
        features = [X[:, s[0]:s[1]] for s in self.splits]
        embeddings = []
        for i in range(self.n_features):
            feature = self.feature_emb_layers[i](features[i], training = training) 
            if self.type_encoding:
                feature += self.type_emb_layer(self.cat_mask[i])
            if self.pos_encoding:
                feature += self.pos_encoding_layer(self.positions[i])
            feature = tf.keras.activations.gelu(feature)
            feature = self.dropout(self.ln(feature, training = training), training = training)
            feature = self.out_mapping(feature, training = training)
            feature = self.ln2(feature, training = training)
            embeddings.append(feature)
        concatenated = tf.concat(embeddings, axis = 1)
        return concatenated
        
 
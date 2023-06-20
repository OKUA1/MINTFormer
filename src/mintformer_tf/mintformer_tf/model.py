import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
from copy import deepcopy
from typing import List, Callable, Optional
from mintformer_tf.blocks import EmbeddingBlock, TransformerBlock, ReconstructionBlock
from mintformer_tf.delayed_decay import DelayedCosineDecay

def _nonefn():
    return None


class MINTFormer(tf.keras.Model):
    def __init__(
        self,
        splits: List[int],
        cat_ind: List[int],
        target_ind: List[int],
        feature_sizes: List,
        input_size: int,
        emb_size: int = 32,
        pos_encoding: bool = True,
        type_encoding: bool = True,
        n_heads: int = 8,
        n_blocks: int = 8,
        dropout_probability_in=0.1,
        dropout_probability_att=0.1,
        dropout_probability_out=0.01,
        use_linear_after_att: bool = True,
        samples_attention_beta: float = 1.0,
        attributes_attention_beta: float = 1.0
    ):

        super().__init__()

        self.n_features = len(feature_sizes)
        self.n_blocks = n_blocks
        self._last_transformer_block_id = n_blocks - 1
        self.emb_size = emb_size
        self.target_ind = target_ind
        self.cat_ind = cat_ind
        self.embedding_block = EmbeddingBlock(
            splits=splits,
            feature_sizes=feature_sizes,
            cat_ind=cat_ind,
            total_size=input_size,
            pos_encoding=pos_encoding,
            type_encoding=type_encoding,
            emb_size=emb_size,
            dropout_probability=dropout_probability_in,
        )
        self.memory_embedding_block = EmbeddingBlock(
            splits=splits,
            feature_sizes=feature_sizes,
            cat_ind=cat_ind,
            total_size=input_size,
            pos_encoding=pos_encoding,
            type_encoding=type_encoding,
            emb_size=emb_size,
            dropout_probability=dropout_probability_in,
        )
        self.transformer_blocks = [
            TransformerBlock(
                n_heads=n_heads,
                n_features=self.n_features,
                hidden_size=emb_size,
                dropout_probability=dropout_probability_att,
                use_linear=use_linear_after_att,
                samples_attention_beta=samples_attention_beta,
                attributes_attention_beta=attributes_attention_beta
            )
            for _ in range(n_blocks)
        ]
        self.reconstrucion_block = ReconstructionBlock(
            feature_sizes=feature_sizes, dropout_probability=dropout_probability_out
        )

    def __call__(self, X, memory, memory_mask=None, training=True):
        # embed X
        X = self.embedding_block(X, training=training)
        # embed memory and add extra dim
        memory = self.memory_embedding_block(memory, training = training)
        memory = tf.expand_dims(memory, 0)
        # apply transformations
        for i in range(self.n_blocks):
            X = self.transformer_blocks[i](
                X, memory=memory, memory_mask=memory_mask, training=training
            )
            X = tf.reshape(X, [-1, self.n_features * self.emb_size])
        X = self.reconstrucion_block(X, training=training)
        return X

    def get_attention_maps(self, X, memory):
        X = self.embedding_block(X, training=False)
        memory = self.memory_embedding_block(memory, training = False)
        memory = tf.expand_dims(memory, 0)
        att_maps = []
        for i in range(self.n_blocks):
            X, att_map = self.transformer_blocks[i].get_attention_maps(X, memory=memory)
            att_maps.append(att_map)
            if i != self._last_transformer_block_id:
                X = tf.reshape(X, [-1, self.n_features * self.emb_size])
        return att_maps

    def loss_fn(
        self,
        y_h,
        y,
        mask,
        splits_in,
        splits_out,
        tradeoff: tf.Tensor,
        use_feature_loss: bool = True,
    ):
        feature_loss, target_loss = 0.0, 0.0
        normalization_f = 0.0
        normalization_t = 0.0
        for i in range(len(splits_in)):
            if not use_feature_loss and i not in self.target_ind:
                continue
            split_in = splits_in[i]
            split_out = splits_out[i]
            mask_i = mask[:, i]
            n_pred = tf.reduce_sum(mask_i)
            y_ = y[:, split_in[0] : split_in[1] - 1]
            y_h_ = y_h[:, split_out[0] : split_out[1]]
            if i in self.cat_ind:
                l = tf.keras.losses.CategoricalCrossentropy(
                    from_logits=True, reduction=tf.keras.losses.Reduction.NONE
                )(y_, y_h_)
            else:
                l = tf.keras.losses.MeanSquaredError(
                    reduction=tf.keras.losses.Reduction.NONE
                )(y_, y_h_)
            l = l * mask_i
            l = tf.reduce_sum(l)
            if i in self.target_ind:
                target_loss += l
                normalization_t += n_pred
            else:
                feature_loss += l
                normalization_f += n_pred
        if use_feature_loss:
            return (
                tradeoff * (feature_loss / normalization_f)
                + (1 - tradeoff) * (target_loss / normalization_t),
                target_loss / normalization_t,
            )
        else:
            return 0.0, target_loss / normalization_t

    @tf.function
    def training_step(
        self,
        X,
        y,
        memory,
        mask,
        memory_mask,
        splits_in,
        splits_out,
        tradeoff: tf.Tensor,
        use_feature_loss: bool = True,
    ):
        with tf.GradientTape() as tape:
            y_h = self(X, memory, memory_mask, training=True)
            loss = self.loss_fn(
                y_h,
                y,
                mask,
                splits_in=splits_in,
                splits_out=splits_out,
                tradeoff=tradeoff,
                use_feature_loss=use_feature_loss,
            )
        grads = tape.gradient(loss[0], self.trainable_weights)
        if not self.use_accum:
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        else:
            self.n_acum_step.assign_add(1)
            for i in range(len(self.gradient_accumulation)):
                self.gradient_accumulation[i].assign_add(grads[i])
            tf.cond(
                tf.equal(self.n_acum_step, self.grad_accum_steps),
                self._apply_accu_gradients,
                _nonefn,
            )
        return loss

    def _apply_accu_gradients(self):

        self.optimizer.apply_gradients(
            zip(self.gradient_accumulation, self.trainable_weights)
        )

        # reset
        self.n_acum_step.assign(0)
        # tf.print('applying grads')
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign(
                tf.zeros_like(self.trainable_weights[i], dtype=tf.float32)
            )

    @tf.function
    def validation_step(
        self,
        X,
        y,
        memory,
        mask,
        memory_mask,
        splits_in,
        splits_out,
        tradeoff: tf.Tensor,
        use_feature_loss: bool = False,
    ):  
        with tf.device("/cpu:0"):
            y_h = self(X, memory, memory_mask, training=False)
            loss = self.loss_fn(
                y_h,
                y,
                mask,
                splits_in=splits_in,
                splits_out=splits_out,
                tradeoff=tradeoff,
                use_feature_loss=use_feature_loss,
            )
            return loss[1]

    def build_from_data(self, X):
        X = X[:2]
        self(X, X, None)

    @tf.function
    def make_mem_batch(self, mem, mem_mask, batch_size):
        idxs = tf.range(tf.shape(mem)[0])
        ridxs = tf.random.shuffle(idxs)[:batch_size]
        rinput = tf.gather(mem, ridxs)
    
        rmask = tf.squeeze(mem_mask, axis = 0)
        rmask = tf.gather(rmask, ridxs, axis = 1)
        rmask = tf.expand_dims(rmask, axis = 0)
        return rinput, rmask
    
    def _sync_embeddings(self):
        self.memory_embedding_block.set_weights(self.embedding_block.get_weights())

    def fit_dataloader(
        self,
        dataloader: Callable,
        batch_size: int = 2048,
        learning_rate: float = 1e-3,
        verbose: int = 0,
        min_epochs: int = 3,
        n_update_steps: Optional[int] = 10000,
        restore_best_weights: Optional[bool] = True,
        n_steps_between_val: Optional[int] = 100,
        feature_mask_prob_train: float = 0.15,
        target_mask_prob_train: float = 1.0,
        feature_noise_prob_train: float = 0.1,
        target_noise_prob_train: float = 0.0,
        feature_mask_prob_val: float = 0.0,
        target_mask_prob_val: float = 1.0,
        feature_noise_prob_val: float = 0.0,
        target_noise_prob_val: float = 0.0,
        early_stopping_patience: float = 2000,
        mandatory_steps_fraction: Optional[float] = 0.75,
        grad_accum_steps: int = 1,
        memory_batch_size: int = 1024
    ):
        if n_update_steps is None:
            raise ValueError("`n_update_steps` must be set to a positive integer")
        self._sync_embeddings()
        n_update_steps = n_update_steps * grad_accum_steps
        batch_size = batch_size // grad_accum_steps
        self.use_accum = grad_accum_steps > 1
        self.grad_accum_steps = tf.constant(grad_accum_steps)
        if verbose > 0:
            print("Preparing the training loop")
            print(
                f"batch_size = {batch_size}, learning_rate = {learning_rate}, n_update_steps = {n_update_steps}"
            )
            if grad_accum_steps > 1:
                print(
                    f"Gradients accumulation is enabled with step_size = {grad_accum_steps}"
                )

        n_epochs = int(np.ceil(n_update_steps / dataloader.n_batches(batch_size)))
        if n_epochs < min_epochs:
            n_epochs = min_epochs
        # setting "real" n_update_steps
        n_update_steps = n_epochs * dataloader.n_batches(batch_size)
        self.optimizer = tfa.optimizers.Lookahead(
            tfa.optimizers.LAMB(
                learning_rate=DelayedCosineDecay(
                    learning_rate, alpha=0.0001, decay_steps=n_update_steps, delay_steps = int(n_update_steps * 0.7)
                ),
                clipnorm=1.0,
                weight_decay=1e-3,
            ),
            sync_period=6 * grad_accum_steps,
        )

        if self.use_accum:
            self.n_acum_step = tf.Variable(0, dtype=tf.int32, trainable=False)
            self.gradient_accumulation = [
                tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False)
                for v in self.trainable_weights
            ]

        # initial dummy metrics
        INF = float("inf")
        best_val_loss = INF
        current_val_loss = INF
        best_step = 0
        best_weights = None
        val_loss_ema = None

        if verbose > 0:
            metrics = ["train_loss", "train_loss_t"]
            if dataloader.is_val_available:
                metrics.append("val_loss_t")
                if restore_best_weights:
                    metrics.append("best_step")
            pb_i = tf.keras.utils.Progbar(n_update_steps, stateful_metrics=metrics)
        tradeoff_annealer = tf.keras.optimizers.schedules.CosineDecay(
            1.0, decay_steps=n_update_steps
        )
        memory = tf.constant(dataloader.memory)
        if memory_batch_size == -1 or memory_batch_size >= memory.shape[0]:
            memory_batch_size = memory.shape[0]
        splits_in = dataloader.splits
        splits_out = dataloader.splits_no_tokens

        break_ = False
        early_stopping_patience = early_stopping_patience * grad_accum_steps

        for epoch in range(n_epochs):
            if break_:
                break
            # Training
            train_losses = []
            train_losses_t = []
            for batch_id, batch in enumerate(
                dataloader.generate_train(
                    batch_size=batch_size,
                    feature_mask_prob=feature_mask_prob_train,
                    feature_noise_prob=feature_noise_prob_train,
                    target_mask_prob=target_mask_prob_train,
                    target_noise_prob=target_noise_prob_train,
                )
            ):
                step_id = epoch * dataloader.n_batches(batch_size) + batch_id
                tradeoff = tradeoff_annealer(step_id)
                unmasked, masked, mask, _, mem_mask, _, _ = batch
                unmasked, masked = (
                    tf.constant(unmasked, dtype=tf.float32),
                    tf.constant(masked, dtype=tf.float32),
                )
                mask, mem_mask = (
                    tf.constant(mask, dtype=tf.float32),
                    tf.constant(mem_mask, dtype=tf.float32),
                )
                memory_batch, memory_mask_batch = self.make_mem_batch(memory, mem_mask, memory_batch_size)
                loss = self.training_step(
                    masked,
                    unmasked,
                    memory_batch,
                    mask=mask,
                    memory_mask=memory_mask_batch,
                    splits_in=splits_in,
                    splits_out=splits_out,
                    tradeoff=tradeoff,
                    use_feature_loss=True,
                )
                train_losses.append(loss[0])
                train_losses_t.append(loss[1])
                # Validation
                val_losses = []
                if (
                    dataloader.is_val_available
                    and step_id % (n_steps_between_val * grad_accum_steps) == 0
                ):
                    for val_batch in dataloader.generate_val(
                        batch_size=batch_size,
                        feature_mask_prob=feature_mask_prob_val,
                        feature_noise_prob=feature_noise_prob_val,
                        target_mask_prob=target_mask_prob_val,
                        target_noise_prob=target_noise_prob_val,
                    ):
                        unmasked, masked, mask, _, _, _, _ = val_batch
                        unmasked, masked = (
                            tf.constant(unmasked, dtype=tf.float32),
                            tf.constant(masked, dtype=tf.float32),
                        )
                        mask = tf.constant(mask, dtype=tf.float32)
                        val_loss = self.validation_step(
                            masked,
                            unmasked,
                            memory,
                            mask=mask,
                            memory_mask=None,
                            splits_in=splits_in,
                            splits_out=splits_out,
                            tradeoff=tf.constant(0.0),
                            use_feature_loss=False,
                        )
                        val_losses.append(val_loss)
                    current_val_loss = tf.reduce_mean([val_losses])

                    if val_loss_ema is None:
                        val_loss_ema = current_val_loss
                    else:
                        val_loss_ema = current_val_loss * 0.5 + val_loss_ema * 0.5

                    if val_loss_ema < best_val_loss:
                        best_val_loss = val_loss_ema
                        best_step = step_id
                        if restore_best_weights:
                            best_weights = deepcopy(self.get_weights())

                    if (
                        early_stopping_patience > 0
                        and (step_id - best_step) > early_stopping_patience
                    ):
                        if not mandatory_steps_fraction or (
                            mandatory_steps_fraction
                            and step_id > mandatory_steps_fraction * n_update_steps
                        ):
                            if verbose > 0:
                                print(f"\nPerforming early stopping")
                            break_ = True
                            break

                if verbose > 0:
                    values = [
                        ("train_loss", loss[0]),
                        ("train_loss_t", loss[1]),
                    ]
                    if dataloader.is_val_available:
                        values.append(("val_loss_t", current_val_loss))
                        if restore_best_weights:
                            values.append(("best_step", best_step))
                    pb_i.add(1, values=values)

        if dataloader.is_val_available and restore_best_weights:
            print("restoring best weights")
            self.set_weights(best_weights)

    def summary(self):
        K = tf.keras.backend

        trainable_count = int(
            np.sum([K.count_params(p) for p in self.trainable_weights])
        )
        non_trainable_count = int(
            np.sum([K.count_params(p) for p in self.non_trainable_weights])
        )

        print("Total params: {:,}".format(trainable_count + non_trainable_count))
        print("Trainable params: {:,}".format(trainable_count))
        print("Non-trainable params: {:,}".format(non_trainable_count))

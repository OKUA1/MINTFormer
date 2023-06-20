import tensorflow as tf
import math

class DelayedCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, delay_steps, decay_steps, alpha=0.0):
        super(DelayedCosineDecay, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.delay_steps = delay_steps
        self.decay_steps = decay_steps - delay_steps
        self.alpha = alpha

    @tf.function
    def __call__(self, step):
        if step < self.delay_steps:
            return self.initial_learning_rate
        else:
            step = tf.cast(step - self.delay_steps, tf.float32)
            decay_steps = tf.cast(self.decay_steps, tf.float32)
            cosine_decay = 0.5 * (1 + tf.math.cos(math.pi * step / decay_steps))
            decayed_learning_rate = (1 - self.alpha) * cosine_decay + self.alpha
            return self.initial_learning_rate * decayed_learning_rate

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "delay_steps": self.delay_steps,
            "decay_steps": self.decay_steps,
            "alpha": self.alpha,
        }
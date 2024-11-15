import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from typing import Optional, Tuple

# Custom RMSLE metric
class RMSLE(tf.keras.metrics.Metric):
    def __init__(self, name='rmsle', **kwargs):
        super().__init__(name=name, **kwargs)
        self.squared_sum = self.add_weight(name='squared_sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

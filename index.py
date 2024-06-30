import tensorflow as tf
import numpy as np
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


celsius_q    = np.array([6, 7,  8,  8, 9, 10,  12],  dtype=float)
fahrenheit_a = np.array([6, 100, 1000, 2000, 1, 1, 0],  dtype=float)

for i,c in enumerate(celsius_q):
  print("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit_a[i]))
  
model = tf.keras.Sequential([
  tf.keras.layers.Dense(units=1, input_shape=[1])
])
model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1))
history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Finished training the model")

import matplotlib.pyplot as plt
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])

print(model.predict([13]))
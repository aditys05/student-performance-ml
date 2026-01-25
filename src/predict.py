import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("model/student_model.keras")

test_student = np.array([[6, 85, 66]])
prediction = model.predict(test_student)

print("Predicted score:", prediction[0][0])

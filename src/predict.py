import tensorflow as tf
import numpy as np
import joblib

model = tf.keras.models.load_model("model/student_model.keras")
scaler = joblib.load("model/scaler.pkl")

test_student = np.array([[2, 60, 40]])
test_student_scaled = scaler.transform(test_student)

prediction = model.predict(test_student)
print("Predicted score:", prediction[0][0])

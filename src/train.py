import tensorflow as tf
import pandas as pd 

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

#for loading data
data = pd.read_csv("data/data.csv")

x = data[["hours", "attendance", "prev_score"]]

y = data["final_score"]

#the model
model = tf.keras.Sequential([
   tf.keras.Input(shape=(3,)),
   tf.keras.layers.Dense(8, activation="relu"),
   tf.keras.layers.Dense(1)

])

model.compile(
    optimizer ="adam",
    loss      ="mean_squared_error"
)

#training the model
model.fit(x,y, epochs=200, verbose=0)

#saving the model
model.save("model/student_model.keras")

print("Model trained and saved successfully.")



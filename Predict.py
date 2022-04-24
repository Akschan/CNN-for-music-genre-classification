from Model import *

model = keras.models.load_model('CNN_Genre_Classifier.h5')

data = Audio_to_predict("Audio_Path")

predict(model,data)

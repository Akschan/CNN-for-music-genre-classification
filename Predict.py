from Model import *

model = keras.models.load_model('CNN_Genre_Classifier.h5')

predict(model,arr)

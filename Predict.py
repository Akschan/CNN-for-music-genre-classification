from Model import *

model = keras.models.load_model(r'C:\Users\Ju_Eun\PycharmProjects\pythonProject\CNN_Genre_Classifier.h5')

# please not that I predict on n segments of 30 second from audio, your audio must at least contain 30 seconds
# be careful to get out of boundaries using offset or by using a short audio file
# you can change the number of segments to use in the function

data = Audio_to_predict(r"C:\Users\Ju_Eun\Testing_Model\CNN\classic1.wav",offset=30,n_seg=3)

# the prediction uses 3 second segments of the 30 seconds segments (we have 3)
# so we predict 30 times and we take the most predicted genre as the result
predict(model,data)

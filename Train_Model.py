from Model import * # import our function

xtrain, xvalidation, xtest, ytrain, yvalidation, ytest, input_shape = Make_datasets(0.25, 0.2) #make train validation test sets and input shape
                                                                                               #(test is 25% of data and validation is 20%)
  
  
model = Model(input_shape)

optimizer = keras.optimizers.Adam(learning_rate = 0.0001) # useing adam optimizer with 0.0001 learning rate

model.compile(optimizer = optimizer, loss = "sparse_categorical_crossentropy", metrics = ['accuracy']) #compiling the model

history = model.fit(xtrain, ytrain, validation_data = (xvalidation,yvalidation), batch_size=32, epochs=50) # train the model

plot_history(history) #plot & save the plot as CNN.png


test_error, test_accuracy = model.evaluate(xtest, ytest, verbose=1) # evaluate the model on the test set

print("Error on test set is:{}, Accuracy on test set is:{}".format(test_error,test_accuracy))

model.save('CNN_Genre_Classifier.h5')

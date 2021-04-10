# NeuralNumpyNet

This is a little project for myself, to understand the inner workings of neural networks.

## features

* dense, conv2d, maxpool2d, flatten, dropout and batchnorm layers
* crossentropy and mse loss functions
* sgd with momentum and nadam optimizers
* gradient checking
* tanh, sigmoid, relu, lrelu and softmax activation functions
* L1, L2, L1_L2 regularizers
* normal and xavier initialization
* API is copied from keras

## example of the API

### program

```python

# setup code ...
xtrain, xtest, ytrain, ytest = load_mnist.load(fraction_of_data=1)

xtrain = np.reshape(xtrain, (28**2, -1))
xtest = np.reshape(xtest, (28**2, -1))

input_dim = xtrain.shape[0]
output_dim = ytrain.shape[0]

model = Sequential()

model.add(Dense(400, tanh, input_shape=input_dim, kernel_initializer=normal))
model.add(Dropout(0.5)),
model.add(Dense(100, tanh, kernel_initializer=normal, kernel_regularizer=L1_L2(1e-4, 1e-3)))
model.add(Dense(output_dim, softmax))


sgd = SGD(learning_rate=2*10**(-1), bias_correction=True, momentum=0.9)
nadam = Nadam(learning_rate=10**(-2), beta_1=0.9, beta_2=0.999, eps=10**(-8))

model.compile(loss = crossentropy, optimizer=nadam)
print(model.summary())

print('calculating loss for initial sanity check:')
model.loss(xtrain, ytrain, verbose=True)

history = model.fit(
    x=xtrain,
    y=ytrain,
    epochs=10,
    batch_size=500,
    validation_data=(xtest, ytest),
    gradients_to_check_each_epoch=5,
    verbose=True
)
```
### output

![example_output](gfx/fullyconnected_demo.gif)

## requirements

 * python 3.8
 * numpy

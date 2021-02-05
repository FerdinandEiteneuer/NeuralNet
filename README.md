# NeuralNumpyNet

This is a little project for myself, to understand the inner workings of neural networks.

## features

* fully connected layer
* commonly used activation functions and loss functions
* gradient checking
* sgd with momentum and nadam optimizers
* parts of the keras API are replicated

## example of the API

```python

# ....
# setupcode

input_dim = xtrain.shape[0]
output_dim = ytrain.shape[0]

kernel_init = normal
depth = 200

model = Network()

model.add(Dense(input_dim, depth, tanh, kernel_init, kernel_regularization=L2(1e-5)))
model.add(Dense(depth, depth, tanh, kernel_init, kernel_regularization=L1_L2(1e-4, 1e-4))
model.add(Dense(depth, output_dim, softmax, kernel_init))

nadam = Nadam(lr=1e-4, beta_1=0.9, beta_2=0.999, eps=10**(-8))
model.compile(loss=crossentropy, optimizer=nadam)

model.fit(
    x=xtrain,
    y=ytrain,
    epochs=500,
    batch_size=1000,
    validation_data=(xtest, ytest),
    gradients_to_check_each_epoch=5,
    verbose=True
)

```

## requirements

 * python 3.8
 * numpy

# NeuralNumpyNet

This is a little project for myself, to understand the inner workings of neural networks.

## features

* fully connected layer
* commonly used activation functions and loss functions
* gradient checking
* sgd with momentum and nadam optimizer
* parts of the keras API are replicated

## example of the API

### program

```python

# ....
# setupcode

input_dim = xtrain.shape[0]
output_dim = ytrain.shape[0]

kernel_init = normal
depth = 200

model = Network()

model.add(Dense(input_dim, depth, tanh, kernel_init, kernel_regularizer=L2(1e-5)))
model.add(Dense(depth, depth, tanh, kernel_init, kernel_regularizer=L1_L2(1e-4, 1e-4)))
model.add(Dense(depth, output_dim, softmax, kernel_init))

nadam = Nadam(lr=1e-4, beta_1=0.9, beta_2=0.999, eps=10**(-8))
model.compile(loss=crossentropy, optimizer=nadam)

model.fit(
    x=xtrain,
    y=ytrain,
    epochs=6,
    batch_size=500,
    validation_data=(xtest, ytest),
    gradients_to_check_each_epoch=5,
    verbose=True
)

```
### output

```
______________________________________________________________
Layer (type)                 Output Shape              Param #
==============================================================
dense_1 (Dense)              (200, 784)                157000
dense_2 (Dense)              (200, 200)                40200
dense_3 (Dense)              (10, 200)                 2010
Total params: 199210
Trainable params: 199210
Non-trainable params: 0

Optimizer: Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, eps=1e-08)

calculating loss for initial sanity check: loss=2.3028e+00, regularizer_loss=5.9825e-07
epoch=1, loss=0.723, train: 54990/60000, val_loss=0.286, test: 9143/10000, gradcheck: 7.049e-07
epoch=2, loss=0.243, train: 56655/60000, val_loss=0.194, test: 9441/10000, gradcheck: 2.316e-08
epoch=3, loss=0.171, train: 57554/60000, val_loss=0.150, test: 9536/10000, gradcheck: 4.718e-08
epoch=4, loss=0.130, train: 58167/60000, val_loss=0.122, test: 9642/10000, gradcheck: 1.883e-08
epoch=5, loss=0.103, train: 58512/60000, val_loss=0.107, test: 9674/10000, gradcheck: 1.590e-08
epoch=6, loss=0.084, train: 58836/60000, val_loss=0.090, test: 9718/10000, gradcheck: 2.357e-09

```

## requirements

 * python 3.8
 * numpy

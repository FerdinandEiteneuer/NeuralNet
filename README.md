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
    epochs=500,
    batch_size=1000,
    validation_data=(xtest, ytest),
    gradients_to_check_each_epoch=5,
    verbose=True
)

```
### output

```
Layer (type)                 Output Shape              Param #
==============================================================
dense_1 (Dense)              (100, 784)                78500
dense_2 (Dense)              (100, 100)                10100
dense_3 (Dense)              (10, 100)                 1010
Total params: 89610
Trainable params: 89610
Non-trainable params: 0

Optimizer: Nadam(lr=0.0007, beta_1=0.9, beta_2=0.999, eps=1e-08)

calculating loss for initial sanity check: loss=2.3028e+00, regularizer_loss=1.4963e-07
epoch=1, loss=1.232, train: 51253/60000, val_loss=0.536, test: 8579/10000, gradcheck: 1.378e-06
epoch=2, loss=0.399, train: 54829/60000, val_loss=0.308, test: 9113/10000, gradcheck: 8.165e-08
epoch=3, loss=0.271, train: 56163/60000, val_loss=0.234, test: 9326/10000, gradcheck: 2.900e-08
epoch=4, loss=0.210, train: 56952/60000, val_loss=0.189, test: 9459/10000, gradcheck: 3.230e-08
epoch=5, loss=0.170, train: 57504/60000, val_loss=0.160, test: 9549/10000, gradcheck: 9.404e-09
epoch=6, loss=0.142, train: 57881/60000, val_loss=0.140, test: 9577/10000, gradcheck: 8.605e-09
epoch=7, loss=0.121, train: 58205/60000, val_loss=0.126, test: 9626/10000, gradcheck: 4.520e-09
```

## requirements

 * python 3.8
 * numpy

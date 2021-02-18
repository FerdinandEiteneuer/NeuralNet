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
xtrain, xtest, ytrain, ytest = load_mnist.load()

xtrain = np.resize(xtrain, (28**2, 60000))
xtest = np.resize(xtest, (28**2, 10000))

input_dim = xtrain.shape[0]
output_dim = ytrain.shape[0]

model = Sequential()

model.add(Dense(200, tanh, input_dim=input_dim, kernel_initializer=normal))
model.add(Dense(100, tanh, kernel_initializer=normal, kernel_regularizer=L1_L2(1e-4, 1e-3)))
model.add(Dense(output_dim, softmax, kernel_init))


sgd = SGD(learning_rate=2*10**(-1), bias_correction=True, momentum=0.9)
nadam = Nadam(learning_rate=10**(-3), beta_1=0.9, beta_2=0.999, eps=10**(-8))

model.compile(loss = crossentropy, optimizer=nadam)
print(model.summary())

print('calculating loss for initial sanity check: ', end='')
model.get_loss(xtrain, ytrain, average_examples=True, verbose=True)

model.fit(
    x=xtrain,
    y=ytrain,
    epochs=8,
    batch_size=500,
    validation_data=(xtest, ytest),
    gradients_to_check_each_epoch=3,
    verbose=True
)

```
### output

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_0 (Dense)              (200, None)               157000
dense_1 (Dense)              (100, None)               20100
dense_2 (Dense)              (10, None)                1010
=================================================================
Total params: 178110
Trainable params: 178110
Non-trainable params: 0
_________________________________________________________________

Optimizer: Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, eps=1e-08)

calculating loss for initial sanity check: loss=2.3029e+00, regularizer_loss=2.9946e-07

epoch=1, loss=0.894, train: 54223/60000, val_loss=0.339, test: 9045/10000, gradcheck: 7.672e-07
epoch=2, loss=0.276, train: 56361/60000, val_loss=0.218, test: 9392/10000, gradcheck: 5.470e-09
epoch=3, loss=0.188, train: 57430/60000, val_loss=0.162, test: 9528/10000, gradcheck: 2.133e-08
epoch=4, loss=0.140, train: 58132/60000, val_loss=0.130, test: 9622/10000, gradcheck: 2.345e-08
epoch=5, loss=0.109, train: 58509/60000, val_loss=0.107, test: 9679/10000, gradcheck: 8.755e-09
epoch=6, loss=0.088, train: 58788/60000, val_loss=0.095, test: 9721/10000, gradcheck: 1.931e-09
epoch=7, loss=0.073, train: 59034/60000, val_loss=0.084, test: 9739/10000, gradcheck: 6.486e-09
epoch=8, loss=0.061, train: 59201/60000, val_loss=0.078, test: 9765/10000, gradcheck: 1.190e-08

```

## requirements

 * python 3.8
 * numpy

# NeuralNumpyNet

This is a little project for myself, to understand the inner workings of neural networks.


## features

* fully connected layer
* commonly used activation functions and loss functions
* SGD with momentum and Nadam optimizer
* gradient checking
* parts of the keras API are replicated

## usage

```python

xtrain, xtest, ytrain, ytest = load_mnist.load()

xtrain = np.resize(xtrain, (28**2, 60000))
xtest = np.resize(xtest, (28**2, 10000))

input_dim = xtrain.shape[0]
output_dim = ytrain.shape[0]

kernel_init= 'normal'

model = Sequential()

model.add(Dense(input_dim, 200, relu, kernel_init, kernel_regularizers=L2(1e-4))
model.add(Dense(200, 200, relu, kernel_init))
model.add(Dense(200, output_dim, softmax, kernel_init))

nadam = Nadam(learning_rate=10**(-3), beta_1=0.9, beta_2=0.999, eps=10**(-8))


model.compile(loss = crossentropy, optimizer=nadam)

model.fit(
    x=xtrain,
    y=ytrain,
    epochs=50,
    batch_size=256,
    validation_data=(xtest, ytest),
    gradients_to_check_each_epoch=5,
    verbose=True
)

```


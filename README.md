# NeuralNumpyNet

This is a little project for myself, to understand the inner workings of neural networks.


## features

* fully connected layer
* commonly used activation functions and loss functions
* gradient checking
* parts of the keras API are replicated

## example of the API

```python

# ....
# setupcode

input_dim = xtrain.shape[0]
output_dim = ytrain.shape[0]

kernel_init= 'normal'
depth = 200

model = Network()

model.add(Dense(input_dim, depth, relu, kernel_init))
model.add(Dense(depth, depth, relu, kernel_init))
model.add(Dense(depth, output_dim, softmax, kernel_init))

model.compile(loss = crossentropy, lr = 1*10**(-1))

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



**Note**: Following docs is prepared from the keras official docs/code, with the objective to prepared only info that is used daily in most cases.


This doc contains following contents:

- Imp things about Layer
- Get weights of layers
- Usuage of regularizer(#usuage-of-regularizer)
- Custom Loss function(#custom-loss-function) [Well-explained loss function](https://gombru.github.io/2018/05/23/cross_entropy_loss/)
- Usuage of initializers(#usuage-of-initializers)
- Keras Models(#keras-models)
- Usuage of optimizers(#usuage-of-optimizers)
    - SGD
    - RMSPRop
    - Adagrad
    - AdaDelta
    - Adam
    - Nadam


## What is inside the layer

Layer encapsulates the weights and the associated computations of the layer. The call method of a layer class contains the layer’s logic. The layer has inbound_nodes and outbound_nodes attributes. Each time a layer is connected to a new input, a node is added to inbound_nodes. Each time the output of a layer is used by another layer, a node is added to outbound_nodes. The layer also carries a list of trainable and non-trainable weights of the layer.



## get_weights of keras layers

```python
If you want to get weights and biases of all layers, you can simply use:

```python
for layer in model.layers: print(layer.get_config(), layer.get_weights())
```

This will print all information that's relevant.

If you want the weights directly returned as numpy arrays, you can use:

```python
first_layer_weights = model.layers[0].get_weights()[0]
first_layer_biases  = model.layers[0].get_weights()[1]
second_layer_weights = model.layers[1].get_weights()[0]
second_layer_biases  = model.layers[1].get_weights()[1]
```

## Usuage of regularizer

The penalties are applied on a per-layer basis. The exact API will depend on the layer, but the layers `Dense`, `Conv1D`, `Conv2D` and `Conv3D` have a unified API.

These layers expose 3 keyword arguments:

    - `kernel_regularizer`: instance of `keras.regularizers.Regularizer`
    - `bias_regularizer`: instance of `keras.regularizers.Regularizer`
    - `activity_regularizer`: instance of `keras.regularizers.Regularizer`


## Example

```python
from keras import regularizers
model.add(Dense(64, input_dim=64,
                kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
```

Available penalties

```python
keras.regularizers.l1(0.)
keras.regularizers.l2(0.)
keras.regularizers.l1_l2(l1=0.01, l2=0.01)
```

## Custom regularizers

Any function that takes in a weight matrix and returns a loss contribution tensor can be used as a regularizer, e.g.:

```python
from keras import backend as K

def l1_reg(weight_matrix):
    return 0.01 * K.sum(K.abs(weight_matrix))

model.add(Dense(64, input_dim=64,
                kernel_regularizer=l1_reg))
```




## Custom Loss function (possible with two keyword `y_true` and `y_pred`)


```python
def squared_hinge(y_true, y_pred):
    return K.mean(K.square(K.maximum(1. - y_true * y_pred, 0.)), axis=-1)


def hinge(y_true, y_pred):
return K.mean(K.maximum(1. - y_true * y_pred, 0.), axis=-1)
```



Another Nice Example on `stack-overflow`


There are two steps in implementing a parameterized custom loss function in Keras. First, writing a method for the coefficient/metric. Second, writing a wrapper function to format things the way Keras needs them to be.


```python
import keras.backend as K
def dice_coef(y_true, y_pred, smooth, thresh):
    y_pred = y_pred > thresh
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
```

Now for the tricky part. Keras loss functions must only take (y_true, y_pred) as parameters. So we need a separate function that returns another function.

```python
def dice_loss(smooth, thresh):
    def dice(y_true, y_pred)
        return -dice_coef(y_true, y_pred, smooth, thresh)
    return dice
```

Finally, you can use it as follows in Keras compile.

```python
# build model 
model = my_model()
# get the loss function
model_dice = dice_loss(smooth=1e-5, thresh=0.5)
# compile model
model.compile(loss=model_dice)
```


## Usuage of initializers:

An initializer may be passed as a string (must match one of the available initializers above), or as a callable:

```python
from keras import initializers

model.add(Dense(64, kernel_initializer=initializers.random_normal(stddev=0.01)))

# also works; will use the default parameters.
model.add(Dense(64, kernel_initializer='random_normal'))
```

### All initializers:
```python
#model.add(Dense(64, kernel_initializer=initializers.random_normal(stddev=0.01)))
keras.initializers.he_uniform(seed=None)
keras.initializers.he_normal(seed=None)
keras.initializers.glorot_normal(seed=None)
keras.initializers.glorot_uniform(seed=None)
keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
keras.initializers.Constant(value=0)
```


## Using custom initializers

If passing a custom callable, then it must take the argument `shape` (shape of the variable to initialize) and `dtype` (dtype of generated values):

```python
from keras import backend as K

def my_init(shape, dtype=None):
    return K.random_normal(shape, dtype=dtype)

model.add(Dense(64, kernel_initializer=my_init))
```


## Keras Models

These models have a number of methods and attributes in common:
"""
- `model.layers` is a flattened list of the layers comprising the model.
- `model.inputs` is the list of input tensors of the model.
- `model.outputs` is the list of output tensors of the model.
- `model.summary()` prints a summary representation of your model. Shortcut for [utils.print_summary](/utils/#print_summary)
- `model.get_config()` returns a dictionary containing the configuration of the model. The model can be reinstantiated from its config via:


- `model.get_weights()` returns a list of all weight tensors in the model, as Numpy arrays.
- `model.set_weights(weights)` sets the values of the weights of the model, from a list of Numpy arrays. The arrays in the list should have the same shape as those returned by `get_weights()`.


- `model.save_weights(filepath)` saves the weights of the model as a HDF5 file.
- `model.load_weights(filepath, by_name=False)` loads the weights of the model from a HDF5 file (created by `save_weights`). By default, the architecture is expected to be unchanged. To load weights into a different architecture (with some layers in common), use `by_name=True` to load only those layers with the same name.
"""





## Usage of optimizers

```python
from keras import optimizers

model = Sequential()
model.add(Dense(64, kernel_initializer='uniform', input_shape=(10,)))
model.add(Activation('softmax'))

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)
```

### List of optimizers
- SGD
- RMSPRop
- Adagrad
- AdaDelta
- Adam
- Nadam

## SGD
### SGD(lr=0.01, momentum=0., decay=0., nesterov=False):
    """Stochastic gradient descent optimizer.
    Includes support for momentum,
    learning rate decay, and Nesterov momentum.
    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter that accelerates SGD
            in the relevant direction and dampens oscillations.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
    """



## RMSProp

### RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.):
    """RMSProp optimizer.
    It is recommended to leave the parameters of this optimizer
    at their default values
    (except the learning rate, which can be freely tuned).
    This optimizer is usually a good choice for recurrent
    neural networks.
    # Arguments
        lr: float >= 0. Learning rate.
        rho: float >= 0.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
    # References
        - [rmsprop: Divide the gradient by a running average of its recent magnitude
           ](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
    """



## Adagrad

### Adagrad(lr=0.01, epsilon=None, decay=0.):
    """Adagrad optimizer.
    Adagrad is an optimizer with parameter-specific learning rates,
    which are adapted relative to how frequently a parameter gets
    updated during training. The more updates a parameter receives,
    the smaller the updates.
    It is recommended to leave the parameters of this optimizer
    at their default values.
    # Arguments
        lr: float >= 0. Initial learning rate.
        epsilon: float >= 0. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
    # References
        - [Adaptive Subgradient Methods for Online Learning and Stochastic
           Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
    """



## AdaDelta

### Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.,)
    """Adadelta optimizer.
    Adadelta is a more robust extension of Adagrad
    that adapts learning rates based on a moving window of gradient updates,
    instead of accumulating all past gradients. This way, Adadelta continues
    learning even when many updates have been done. Compared to Adagrad, in the
    original version of Adadelta you don't have to set an initial learning
    rate. In this version, initial learning rate and decay factor can
    be set, as in most other Keras optimizers.
    It is recommended to leave the parameters of this optimizer
    at their default values.
    # Arguments
        lr: float >= 0. Initial learning rate, defaults to 1.
            It is recommended to leave it at the default value.
        rho: float >= 0. Adadelta decay factor, corresponding to fraction of
            gradient to keep at each time step.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Initial learning rate decay.
    # References
        - [Adadelta - an adaptive learning rate method](
           https://arxiv.org/abs/1212.5701)
    """



## Adam


### Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0., amsgrad=False):
    """Adam optimizer.
    Default parameters follow those provided in the original paper.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and
            Beyond".
    # References
        - [Adam - A Method for Stochastic Optimization](
           https://arxiv.org/abs/1412.6980v8)
        - [On the Convergence of Adam and Beyond](
           https://openreview.net/forum?id=ryQu7f-RZ)
    """




## Nadam


### Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004):
    """Nesterov Adam optimizer.
    Much like Adam is essentially RMSprop with momentum,
    Nadam is Adam RMSprop with Nesterov momentum.
    Default parameters follow those provided in the paper.
    It is recommended to leave the parameters of this optimizer
    at their default values.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1/beta_2: floats, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
    # References
        - [Nadam report](http://cs229.stanford.edu/proj2015/054_report.pdf)
        - [On the importance of initialization and momentum in deep learning](
           http://www.cs.toronto.edu/~fritz/absps/momentum.pdf)
    """




### Best advice about the choice of optimizers


- For time series ==> prefereably RMSPRop
- Best to use ==> SGD+Nestrov or Adam
    Nestrov ==> First jump and then correct its step(or jump)
- sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
- Batch normalization additionally acts as a regularizer, reducing the need for Dropout.
- For sparse data (tf-idf) ==> AdaGrad, AdaDelta(improved over AdaGrad)
    Best, if we are using sparse data such as tf-idf features for words.



- AdaGrad: weakness: lr_rate decaying
Notice that the weights that receive high gradients will have their effective learning rate reduced, while weights that receive small or infrequent updates will have their effective learning rate increased. This agressive behaviour, stops Deep-NN to learn very early.

- RMSProp:
The RMSProp update adjusts the Adagrad method in a very simple way in an attempt to reduce its aggressive, monotonically decreasing learning rate. In particular, it uses a moving average of squared gradients instead. Hence, RMSProp still modulates the learning rate of each weight based on the magnitudes of its gradients, which has a beneficial equalizing effect, but unlike Adagrad the updates do not get monotonically smaller

- AdaDelta: Removed weakness of adadelta
Best, if we are using sparse data such as tf-idf features for words.


# Practical-Guide-Keras

## keras lr-rate scheduler as well as checkpoints with the epochs
```
def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180, 200 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 200:
        lr *= 1e-4
    elif epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr
```

```
# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler]
```


#### Note: Following docs is prepared from the keras official docs/code, with the objective to prepared only info that is used daily in most cases.

## verbose
verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.

## ModelCheckpoint
ModelCheckpoint((filepath, 
                 monitor='val_loss', 
                 verbose=0, # Best is to use 1, it shows the progress bar
                 save_best_only=False, 
                 save_weights_only=False, 
                 mode='auto', 
                 period=1)):
                 
    """Save the model after every epoch.
    
    # Arguments
        filepath: Best use is `filepath` ==> `weights.{epoch:02d}-{val_loss:.2f}.hdf5`.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            For `val_acc`, this should be `max`, 
            for `val_loss` this should be `min`. 
            In `auto` mode, the direction is automatically inferred
        period: Interval (number of epochs) between checkpoints.
    """

    
    
    
## EarlyStopping
    EarlyStopping(monitor='val_loss',
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None, # **Use any other model, like logistic regression first to get baseline**
                 restore_best_weights=False)
                 
    """Stop training when a monitored quantity has stopped improving.
    # Arguments
        monitor: quantity to be monitored.
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        baseline: Baseline value for the monitored quantity to reach.
            Training will stop if the model doesn't show improvement
            over the baseline.
        restore_best_weights: whether to restore model weights from
            the epoch with the best value of the monitored quantity.
            If False, the model weights obtained at the last step of
            training are used.
    """





## ReduceLROnPlateau

    # Example
```python
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
model.fit(X_train, Y_train, callbacks=[reduce_lr])
```

    ReduceLROnPlateau(monitor='val_loss', 
                     factor=0.1, 
                     patience=10,
                     verbose=0, 
                     mode='auto', 
                     min_lr=0,
                     **kwargs)
                     
        """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This callback monitors a
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    # Arguments
        factor: factor by which the learning rate will
            be reduced. new_lr = lr * factor
        patience: number of epochs with no improvement
            after which learning rate will be reduced.
        min_lr: lower bound on the learning rate.
    """




## fit(self,
        fit(self
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            steps_per_epoch=None,
            validation_split=0.,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0):

    Way to generate sample-weights(**Most important for the imbalanced data**)

    For simplicity, lets say you know the weights you need for each class and can pass it as a dictionary. In my example I have my y_train as a one hot encoded vector. I'm using that fact to reverse engineer which class each row is pointing to and adding a weight for it. You essentially need to pass an array of weights mapping to each label (so the same length as your training data) when you fit the model.

```python

**Test this for an example, if it works fine**
class_weight_dictionary = {'0':0.3, '1':0.7}

def generate_sample_weights(y_train, class_weight_dictionary): 
    sample_weights = [class_weight_dictionary[np.where(one_hot_row==1)[0][0]] for one_hot_row in y_train]
    return np.asarray(sample_weights)

model.fit(x=X_train, 
    y=y_train, 
    batch_size = 64,
    validation_data=(X_val, y_val),
    shuffle=True,
    epochs=20,
    sample_weight = generate_sample_weights(y_train, class_weights_dict)
)
```

        """
        # Arguments
            x:  single input    ==> numpy array                x 
                multiple input  ==> list of numpy array     [x1, x2]
            y: single input    ==> numpy array                 y
                multiple input  ==> list of numpy array     [y1, y2]
            batch_size: integer [default 32]
            epochs: Integer. 
            verbose: Integer. 0 = silent, 1 = progress bar, 2 = one line per epoch.
            callbacks: List of `keras.callbacks.Callback` instances.
            validation_split: Float between 0 and 1.
                Fraction of the training data to be used as validation data.
            validation_data: tuple `(x_val, y_val)` 
            shuffle: Boolean (whether to shuffle the training data
                before each epoch) or str (for 'batch').
                'batch' is a special option for dealing with the
                limitations of HDF5 data; it shuffles in batch-sized chunks.
                Has no effect when `steps_per_epoch` is not `None`.
            class_weight: Optional dictionary mapping class indices (integers)
                to a weight (float) value, used for weighting the loss function
                (during training only).
                This can be useful to tell the model to
                "pay more attention" to samples from
                an under-represented class.
            sample_weight: Optional Numpy array of weights for
                the training samples, used for weighting the loss function
                (during training only). You can either pass a flat (1D)
                Numpy array with the same length as the input samples
                (1:1 mapping between weights and samples),
                or in the case of temporal data,
                you can pass a 2D array with shape
                `(samples, sequence_length)`,
                to apply a different weight to every timestep of every sample.
                In this case you should make sure to specify
                `sample_weight_mode="temporal"` in `compile()`.
            initial_epoch: Integer (resuming a previous training run)
                Epoch at which to start training
            steps_per_epoch: Total number of steps (batches of samples)

        # Returns
            `History.history` attribute is a record of training loss values and metrics values.
        """
        
        
        
        
 ## fit_generator
```python
def generate_arrays_from_file(path):
    while True:
        with open(path) as f:
            for line in f:
                # create numpy arrays of input data
                # and labels, from each line in the file
                x1, x2, y = process_line(line)
                yield ({'input_1': x1, 'input_2': x2}, {'output': y})
model.fit_generator(generate_arrays_from_file('/my_file.txt'),
                    steps_per_epoch=10000, epochs=10)
```
    fit_generator(generator,
                  steps_per_epoch=None,
                  epochs=1,
                  verbose=1,
                  callbacks=None,
                  validation_data=None,
                  validation_steps=None,
                  class_weight=None,
                  max_queue_size=10,
                  workers=1,
                  use_multiprocessing=False,
                  shuffle=True,
                  initial_epoch=0)
        """Trains the model on data generated batch-by-batch by a Python generator
        (or an instance of `Sequence`).
        The generator is run in parallel to the model, for efficiency.
        For instance, this allows you to do real-time data augmentation
        on images on CPU in parallel to training your model on GPU.
        The use of `keras.utils.Sequence` guarantees the ordering
        and guarantees the single use of every input per epoch when
        using `use_multiprocessing=True`.
        # Arguments
            generator: A generator or an instance of `Sequence`
                (`keras.utils.Sequence`) object in order to avoid
                duplicate data when using multiprocessing.
                The output of the generator must be either
                - a tuple `(inputs, targets)`
                - a tuple `(inputs, targets, sample_weights)`.
                This tuple (a single output of the generator) makes a single
                batch. Therefore, all arrays in this tuple must have the same
                length (equal to the size of this batch). Different batches may
                have different sizes. For example, the last batch of the epoch
                is commonly smaller than the others, if the size of the dataset
                is not divisible by the batch size.
                The generator is expected to loop over its data
                indefinitely. An epoch finishes when `steps_per_epoch`
                batches have been seen by the model.
            steps_per_epoch: Total number of steps (batches of samples)
                
                Optional for `Sequence`: if unspecified, will use
                the `len(generator)` as a number of steps.
            epochs: Integer. Number of epochs to train the model.
            verbose: Integer. 0 = silent, 1 = progress bar, 2 = one line per epoch.
            callbacks: List of callbacks to apply during training.
            validation_data: This can be either
                - tuple `(x_val, y_val)`
                - tuple `(x_val, y_val, val_sample_weights)`
                
            class_weight: Optional dictionary mapping class indices (integers)
                to a weight (float) value, used for weighting the loss function
                (during training only). This can be useful to tell the model to
                "pay more attention" to samples
                from an under-represented class.
            max_queue_size: Integer. Maximum size for the generator queue.

            workers: Integer. Maximum number of processes to spin up
                when using process-based threading.
                If unspecified, `workers` will default to 1. If 0, will
                execute the generator on the main thread.
            use_multiprocessing: Boolean.
                If `True`, use process-based threading.
                If unspecified, `use_multiprocessing` will default to `False`.
               
            initial_epoch: Epoch at which to start training
                (useful for resuming a previous training run).
        # Returns
            `History.history` attribute has a record of training loss values and metrics values
       # Example

        """
        
        
        
        
        
## Custorm Training        

```python
from keras.callbacks import EarlyStopping, ModelCheckpoint
earlystopper = EarlyStopping(patience=2, verbose=1)
checkpointer = ModelCheckpoint('model.h5', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)

model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['mae','accuracy'])


def yield_data(dataset=data, batch_size=batch):
    while True:
        input1, input2, labels = generate_dataset(dataset=data, batch_size=batch_size)

        yield [input1, input2], labels


valid_users, valid_problems, valid_labels = generate_dataset(dataset=data, batch_size=4)


# # we want a constant validation group to have a frame of reference for model performance
# valid_a, valid_b, valid_sim = gen_random_batch(count_list, all_dir, 32)
batch=4
loss_history = rec_sys.fit_generator(yield_data(dataset=data, batch_size=batch), 
                                steps_per_epoch = 100,
                                validation_data=([valid_users, valid_problems], valid_labels),
                                epochs = 10,
                                verbose = True,
                                callbacks=[earlystopper,checkpointer])

```





## History object

```python
plt.figure(figsize=(12,8))
sns.lineplot(range(1, epochs+1), model.history['acc'], label='Train Accuracy')
sns.lineplot(range(1, epochs+1), model.history['val_acc'], label='Test Accuracy')
plt.show()
```





## LSTM 
    '''return_sequences: return output at each time step'''

### LSTM ==> forward
```python
LSTM(out_dims, return_sequences=True, go_backwards=False)(input_layer)
```
### LSTM ==> backward
```python
LSTM(out_dims, return_sequences=True, go_backwards=True)(input_layer)
```
### LSTM ==> Bidirection
        """ Bidirectional(layer, merge_mode='concat'):

        merge_mode: Mode by which outputs of the
                    forward and backward RNNs will be combined.
                    One of {'sum', 'mul', 'concat', 'ave'}
        """

```python
Bidirectional(LSTM(out_dims, return_sequences=True)))(input_layer)
```
## TimeDistributed ==> pass the output of each time-step from the same dense layer, with output of out_dims.
```python
TimeDistributed(Dense(out_dims, activation='sigmoid'))(input_layer)
```

## Bi-directional LSTM  ==>> using Keyword
```python
 
    def get_lstm_model(n_timesteps, backwards):
      model = Sequential()
      model.add(LSTM(20, input_shape=(n_timesteps, 1), return_sequences=True, go_backwards=backwards))
      model.add(TimeDistributed(Dense(1, activation='sigmoid')))
      model.compile(loss='binary_crossentropy', optimizer='adam')
      return model

    def get_bi_lstm_model(n_timesteps, mode):
      model = Sequential()
      model.add(Bidirectional(LSTM(20, return_sequences=True), input_shape=(n_timesteps, 1), merge_mode=mode))
      model.add(TimeDistributed(Dense(1, activation='sigmoid')))
      model.compile(loss='binary_crossentropy', optimizer='adam')
      return model

    def train_model(model, n_timesteps):
      loss = list()
      for _ in range(250):
        # generate new random sequence
        X,y = get_sequence(n_timesteps)
        # fit model for one epoch on this sequence
        hist = model.fit(X, y, epochs=1, batch_size=1, verbose=0)
        loss.append(hist.history['loss'][0])
      return loss


    n_timesteps = 10
    results = DataFrame()
    # lstm forwards
    model = get_lstm_model(n_timesteps, False)
    results['lstm_forw'] = train_model(model, n_timesteps)
    # lstm backwards
    model = get_lstm_model(n_timesteps, True)
    results['lstm_back'] = train_model(model, n_timesteps)
    # bidirectional concat
    model = get_bi_lstm_model(n_timesteps, 'concat')
    results['bilstm_con'] = train_model(model, n_timesteps)
    # line plot of results
    results.plot()
    pyplot.show()
```
    


    



## Stratified Sampling with Keras
Split the entire data into one training and test data-set
```python
    ## stratify sampling or splitting
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df.col1,df.target,
                                                        stratify=df.target, 
                                                        test_size=0.2)
```
## Stratify k-Fold training
Idea is to take one Fold and reset the model and train it again
```python
from sklearn.model_selection import StratifiedKFold

    # Instantiate the cross validator
    skf = StratifiedKFold(n_splits=kfold_splits, shuffle=True)

    # Loop through the indices the split() method returns
    for index, (train_indices, val_indices) in enumerate(skf.split(X, y)):
        print "Training on fold " + str(index+1) + "/10..."

        # Generate batches from indices
        xtrain, xval = X[train_indices], X[val_indices]
        ytrain, yval = y[train_indices], y[val_indices]

        # Clear model, and create it
        model = None
        model = create_model()#create_model is function to make new model

        # Debug message I guess
        # print "Training new iteration on " + str(xtrain.shape[0]) + " training samples, " + str(xval.shape[0]) + " validation samples, this may be a while..."

        history = train_model(model, xtrain, ytrain, xval, yval)
        accuracy_history = history.history['acc']
        val_accuracy_history = history.history['val_acc']
        print "Last training accuracy: " + str(accuracy_history[-1]) + ", last validation accuracy: " + str(val_accuracy_history[-1])
```
    
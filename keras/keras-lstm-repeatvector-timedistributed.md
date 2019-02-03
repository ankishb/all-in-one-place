

For first question, i am doing same thing, i didn't get any error, please share your error.

**Note**: I will give you example using functional API, which gives little more freedom(personal opinion)


    from keras.layers import Dense, Flatten, LSTM, Activation
    from keras.layers import Dropout, RepeatVector, TimeDistributed
    from keras import Input, Model
    
    seq_length = 15
    input_dims = 10
    output_dims = 8
    n_hidden = 10
    model1_inputs = Input(shape=(seq_length,input_dims,))
    model1_outputs = Input(shape=(output_dims,))
    
    net1 = LSTM(n_hidden, return_sequences=True)(model1_inputs)
    net1 = LSTM(n_hidden, return_sequences=False)(net1)
    net1 = Dense(output_dims, activation='relu')(net1)
    model1_outputs = net1
    
    model1 = Model(inputs=model1_inputs, outputs = model1_outputs, name='model1')
    
    ## Fit the model
    model1.summary()


    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_11 (InputLayer)        (None, 15, 10)            0         
    _________________________________________________________________
    lstm_8 (LSTM)                (None, 15, 10)            840       
    _________________________________________________________________
    lstm_9 (LSTM)                (None, 10)                840       
    _________________________________________________________________
    dense_9 (Dense)              (None, 8)                 88        
    _________________________________________________________________





To your second problem, there are two method:

1. If you are sending data without making sequence, which is of dims as `(batch, input_dims)`, then use can use this method [**RepeatVector**](https://keras.io/layers/core/), which repeat the same weights by `n_steps`, which is nothing but `rolling_steps` in LSTM.

{

    seq_length = 15
    input_dims = 16
    output_dims = 8
    n_hidden = 20
    lstm_dims = 10
    model1_inputs = Input(shape=(input_dims,))
    model1_outputs = Input(shape=(output_dims,))
    
    net1 = Dense(n_hidden)(model1_inputs)
    net1 = Dense(n_hidden)(net1)
    
    net1 = RepeatVector(3)(net1)
    net1 = LSTM(lstm_dims, return_sequences=True)(net1)
    net1 = LSTM(lstm_dims, return_sequences=False)(net1)
    net1 = Dense(output_dims, activation='relu')(net1)
    model1_outputs = net1
    
    model1 = Model(inputs=model1_inputs, outputs = model1_outputs, name='model1')
    
    ## Fit the model
    model1.summary()

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_13 (InputLayer)        (None, 16)                0         
    _________________________________________________________________
    dense_13 (Dense)             (None, 20)                340       
    _________________________________________________________________
    dense_14 (Dense)             (None, 20)                420       
    _________________________________________________________________
    repeat_vector_2 (RepeatVecto (None, 3, 20)             0         
    _________________________________________________________________
    lstm_14 (LSTM)               (None, 3, 10)             1240      
    _________________________________________________________________
    lstm_15 (LSTM)               (None, 10)                840       
    _________________________________________________________________
    dense_15 (Dense)             (None, 8)                 88        
    =================================================================



 2. If you are sending the sequence of dims `(seq_len, input_dims)`, then you can [**TimeDistributed**](https://keras.io/layers/wrappers/), which repeats the same weights of dense layer on the entire sequence.
 

{

    seq_length = 15
    input_dims = 10
    output_dims = 8
    n_hidden = 10
    lstm_dims = 6
    model1_inputs = Input(shape=(seq_length,input_dims,))
    model1_outputs = Input(shape=(output_dims,))
    
    net1 = TimeDistributed(Dense(n_hidden))(model1_inputs)
    net1 = LSTM(output_dims, return_sequences=True)(net1)
    net1 = LSTM(output_dims, return_sequences=False)(net1)
    net1 = Dense(output_dims, activation='relu')(net1)
    model1_outputs = net1
    
    model1 = Model(inputs=model1_inputs, outputs = model1_outputs, name='model1')
    
    ## Fit the model
    model1.summary()
    
    
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_17 (InputLayer)        (None, 15, 10)            0         
    _________________________________________________________________
    time_distributed_3 (TimeDist (None, 15, 10)            110       
    _________________________________________________________________
    lstm_18 (LSTM)               (None, 15, 8)             608       
    _________________________________________________________________
    lstm_19 (LSTM)               (None, 8)                 544       
    _________________________________________________________________
    dense_19 (Dense)             (None, 8)                 72        
    =================================================================



**Note**: I stacked two layer, on doing so, in the first layer i used `return_sequence`, which return the output at each time step, which is used by second layer, where it is return output only at last `time_step`.


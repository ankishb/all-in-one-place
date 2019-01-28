
## Set learning rate manullay (any method exp-decay, polynomial-vary, ....)

#### For ADAM-optimizer (in eager mode)
```python
lr = 1
optimizer = tf.train.AdamOptimizer(lr)#lr_rate)
for epoch in range(10):
    lr = lr/2
    optimizer._lr = tf.constant(lr)

    print(optimizer._lr.numpy(), optimizer._lr_t.numpy())
```

### Formula for poynomial decayed learning rate
```python
decayed_learning_rate = (starter_learning_rate - end_learning_rate) * \
                    (1 - epoch / decay_steps) ** (power) + \
                    end_learning_rate

```              

### lr_rate decay using Polynomial-function
```python
lr_rate = tfe.Variable(1e-3, trainable=False)
optimizer = tf.train.MomentumOptimizer(lr_rate, momentum=0.95)
starter_learning_rate = 1e-3
end_learning_rate = 1e-5
decay_steps = 100
power = 0.5

n_epochs = 100
n_episodes = 60

tr_losses, val_losses, accuracies = [], [], []
for epoch in range(1,n_epochs):
    decayed_learning_rate = (starter_learning_rate - end_learning_rate) * \
                        (1 - epoch / decay_steps) ** (power) + \
                        end_learning_rate
#     global_step.assign(global_step+1)
    lr_rate.assign(decayed_learning_rate)
```
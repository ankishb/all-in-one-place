---
layout: post
---



## Max-Pooling VS Avg-Pooling
	- Max pooling helps to extract imp information from the spaical region of path. For example if image patch is of size 3X3 then max pooling will give 1 pixel having max intenity that tell about edge.
	- Avg-Pooling mix up everything


> In imagine recognition, pooling also provides basic invariance to translating (shifting) and rotation. When you are pooling over a region, the output will stay approximately the same even if you shift/rotate the image by a few pixels, because the max operations will pick out the same value regardless.

## In sequence data-set, while using LSTM, the output at each time-step can be helpful in further prediction, which means the output at each roll-out of LSTM, can be imp then taking Max-Pool on all those value can be concatenate among the feature for further prediction




lstm1, state_h, state_c = LSTM(1, return_state = True)(inputs1) 


    - The LSTM hidden state output for the last time step.
    - The LSTM hidden state output for the last time step (again).
    - The LSTM cell state for the last time step.


 LSTM(1, return_sequences=True)(inputs1)

 	- return output at ecah roll-out, Helpful, when stacking another LSTM layer on top of first one.

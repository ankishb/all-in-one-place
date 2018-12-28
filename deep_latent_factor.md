
# Deep Matrix Factorization Model:

## Higher Level Idea

Keep two network, one for users having rating for each items, another for items having rating by each users. So we have one big vectors for ecah network, which will find the latent representation for user and item. After that, find the similarity(interaction) between user and item using cosine similarity.

Authr also presented new loss function which is called normalized cross entropy, which is same as binary cross entropy, except **Y** changed by **Y/max(R)**, where **max(R)** is the maximum rating. For example in 5 star rating system, it is 5.




Squred loss finction can not be used well with implicit feedback, because for implicit data, the target value Y ij is a binarized 1 or 0 denoting whether i has interacted with j or not. 
In what follows, a loss function which pays special attention to the binary property of implicit data was proposed by [He et al., 2017] as follows.	cross-entropy loss function


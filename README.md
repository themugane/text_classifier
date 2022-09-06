# Text Classification using Neural Networks

## How Neural Networks work


# Text Classification

Text classification comes with 3 Flavours:

#### 1. Pattern Matching

#### 2. Algorithms 
The algorithmic approach using Multinomial Naive bayes is surprising effective, however it suffers 3 fundermental flaws:

- the algorithm produces a score rather than a probability. We want a probability to ignore predictions below some threshold. This is akin to a 'squelch' dial on a VHF radio.

- the algorithm 'learns' from examples of what is in a class, but not what isn't. This learning of patterns of what does not belong to a class is often very important.

- Classes with disproportionately large training sets can create distorted classification scores, forcing the algorithm to adjust scores relative to class size. This is not ideal.

#### 3. Neural Networks

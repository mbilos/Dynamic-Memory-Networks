# Dynamic memory networks

Tensorflow implementation of DMN+ arhitecture described in [Dynamic Memory Networks for Visual and Textual Question Answering](https://arxiv.org/abs/1603.01417) by Xiong et al. Tested on bAbI-10k dataset.

## Dependencies and running

Dependencies:
```
python 3.6
tensorflow 1.4
numpy
```

Download bAbI tasks and unzip so that directory tree looks like this:
```
Dynamic-Memory-Networks
|-- data
|   |-- babi
|   |   |-- en
|   |   |-- en-10k
|   |-- glove
|   |   --- glove.6B.50d.txt (optional)
|-- babi_util.py
|-- dmn.py
```

Run:
```
    python dmy.py --task --batch --iterations
```

## bAbI dataset

bAbI is synthetic question answering dataset. It is divided into 20 tasks such as Single Supporting Fact, Yes/No Questions, Basic Coreference etc. It can be trained on two regimes:
* weak supervision - uses only question-answer pairs
* strong supervision - uses extra information, set of supporting facts

We will focus on weak supervision.

## DMN

Original [paper](https://arxiv.org/abs/1506.07285) (Kumar et al.) introduces dynamic memory network that processes input text and question and generates answer. It uses memory and attention mechanism and can be trained ent-to-end. Kumar et al. (2015) architecture yields state of the art results on question answering with strong supervision but doesn't perform well on bAbI-10k without supporting facts.

DMN+ is improvement on that work that yields state of the art without supporting facts. It consists of 4 modules: input, question, episodic memory and answer.

### Input

Input module takes text and outputs facts. Text is split into sentences and every sentence is encoded with positional encoding described in [Sukhbaatar et al. (2015)](https://arxiv.org/abs/1503.08895). Now we have sequence of sentence encodings, each one with size of word embedding.

Bidirectional GRU (gated recurrent unit) takes encodings and returns outputs from forward and backward pass and reduces them to sequence of facts. Those are inputs to next module.

### Question

Question module takes question represented as sequence of words (their embeddings) and feeds them to GRU and returns final state.

### Episodic memory

Episodic memory module takes facts and question states to calculate memory for episode. Size of memory is the same as the size of facts. It is updated by attention mechanism which allows focusing on important facts in one pass, and by allowing multiple passes recignizing more complex dependencies.

Attention encourages interactions between fact, question and memory representations:

```
z = [fact * question; fact * memory; |fact - question|; |fact - memory|]
Z = W2 tanh( W1 z + b1 ) + b2
g = softmax(Z)
```

g is then used in GRU to determine how much attention should be given to each fact:

```
state = g * output + (1 - g) * prev_state
```

Final hidden state of GRU is episodic memory state. All episodes are conected so that module output is:

```
m(t) = RELU( W [m(t-1); c; q] + b )
```

### Answer

Since most bAbI tasks have single word answers, simple softmax predicting word from vocabulary is used.

## Results

Tested on tasks with one word answer. Not tested yet on all tasks like Task 3 where
DMN+ training is not stable (as reported in the paper).

Task | Accuracy
---- | ---
1    | 99.6%
2    | 89.2%
3    | -
4    | 99.9%
5    | 98.2%
6    | -
7    | 91.6%
8    | -
9    | 99.7%
10   | 97.8%
11   | 100%
12   | 99.9%
13   | 100%
14   | 94.8%
15   | 99.1%
16   | 44.8%
17   | -
18   | 93.9%
19   | -
20   | 97.6%

## References

* Kumar, A., Irsoy, O., Ondruska, P., Iyyer, M., Bradbury, J., Gulrajani, I., ... & Socher, R. (2016, June). Ask me anything: Dynamic memory networks for natural language processing. In International Conference on Machine Learning (pp. 1378-1387).
* Xiong, C., Merity, S., & Socher, R. (2016, June). Dynamic memory networks for visual and textual question answering. In International Conference on Machine Learning (pp. 2397-2406).
* Sukhbaatar, S., Weston, J., & Fergus, R. (2015). End-to-end memory networks. In Advances in neural information processing systems (pp. 2440-2448).
* https://github.com/therne/dmn-tensorflow
* https://github.com/barronalex/Dynamic-Memory-Networks-in-TensorFlow
## **Transformer-XL: Attentive Language Models** **Beyond a Fixed-Length Context**

**Zihang Dai** _[∗]_ [12] **, Zhilin Yang** _[∗]_ [12] **, Yiming Yang** [1] **, Jaime Carbonell** [1] **,**
**Quoc V. Le** [2] **, Ruslan Salakhutdinov** [1]

1 Carnegie Mellon University, 2 Google Brain


{dzihang,zhiliny,yiming,jgc,rsalakhu}@cs.cmu.edu, qvl@google.com



**Abstract**


Transformers have a potential of learning
longer-term dependency, but are limited by a
fixed-length context in the setting of language
modeling. We propose a novel neural architecture _Transformer-XL_ that enables learning dependency beyond a fixed length without disrupting temporal coherence. It consists of a segment-level recurrence mechanism
and a novel positional encoding scheme. Our
method not only enables capturing longer-term
dependency, but also resolves the context fragmentation problem. As a result, TransformerXL learns dependency that is 80% longer than
RNNs and 450% longer than vanilla Transformers, achieves better performance on both
short and long sequences, and is up to 1,800+
times faster than vanilla Transformers during
evaluation. Notably, we improve the state-ofthe-art results of bpc/perplexity to 0.99 on enwiki8, 1.08 on text8, 18.3 on WikiText-103,
21.8 on One Billion Word, and 54.5 on Penn
Treebank (without finetuning). When trained
only on WikiText-103, Transformer-XL manages to generate reasonably coherent, novel
text articles with thousands of tokens. Our

code, pretrained models, and hyperparameters
are available in both Tensorflow and PyTorch [1] .


**1** **Introduction**


Language modeling is among the important problems that require modeling long-term dependency,
with successful applications such as unsupervised
pretraining (Dai and Le, 2015; Peters et al., 2018;
Radford et al., 2018; Devlin et al., 2018). However, it has been a challenge to equip neural
networks with the capability to model long-term
dependency in sequential data. Recurrent neural networks (RNNs), in particular Long Short

_∗_ Equal contribution. Order determined by swapping the
one in Yang et al. (2017).
1 [https://github.com/kimiyoung/](https://github.com/kimiyoung/transformer-xl)

[transformer-xl](https://github.com/kimiyoung/transformer-xl)



Term Memory (LSTM) networks (Hochreiter and
Schmidhuber, 1997), have been a standard solution to language modeling and obtained strong
results on multiple benchmarks. Despite the
wide adaption, RNNs are difficult to optimize
due to gradient vanishing and explosion (Hochreiter et al., 2001), and the introduction of gating in LSTMs and the gradient clipping technique (Graves, 2013) might not be sufficient to
fully address this issue. Empirically, previous
work has found that LSTM language models use
200 context words on average (Khandelwal et al.,
2018), indicating room for further improvement.
On the other hand, the direct connections between long-distance word pairs baked in attention mechanisms might ease optimization and enable the learning of long-term dependency (Bahdanau et al., 2014; Vaswani et al., 2017). Recently, Al-Rfou et al. (2018) designed a set of auxiliary losses to train deep Transformer networks
for character-level language modeling, which outperform LSTMs by a large margin. Despite the
success, the LM training in Al-Rfou et al. (2018)
is performed on separated fixed-length segments
of a few hundred characters, without any information flow across segments. As a consequence of
the fixed context length, the model cannot capture
any longer-term dependency beyond the predefined context length. In addition, the fixed-length
segments are created by selecting a consecutive
chunk of symbols without respecting the sentence
or any other semantic boundary. Hence, the model
lacks necessary contextual information needed to
well predict the first few symbols, leading to inefficient optimization and inferior performance. We
refer to this problem as _context fragmentation_ .

To address the aforementioned limitations of

fixed-length contexts, we propose a new architecture called Transformer-XL (meaning extra long).
We introduce the notion of recurrence into our


deep self-attention network. In particular, instead
of computing the hidden states from scratch for
each new segment, we reuse the hidden states obtained in previous segments. The reused hidden
states serve as memory for the current segment,
which builds up a recurrent connection between
the segments. As a result, modeling very longterm dependency becomes possible because information can be propagated through the recurrent connections. Meanwhile, passing information from the previous segment can also resolve
the problem of context fragmentation. More importantly, we show the necessity of using relative
positional encodings rather than absolute ones, in
order to enable state reuse without causing temporal confusion. Hence, as an additional technical contribution, we introduce a simple but more
effective relative positional encoding formulation
that generalizes to attention lengths longer than the
one observed during training.
Transformer-XL obtained strong results on five
datasets, varying from word-level to characterlevel language modeling. Transformer-XL is also
able to generate relatively coherent long text articles with _thousands of_ tokens (see Appendix E),
trained on only 100M tokens.

Our main technical contributions include intro
ducing the notion of recurrence in a purely selfattentive model and deriving a novel positional encoding scheme. These two techniques form a complete set of solutions, as any one of them alone
does not address the issue of fixed-length contexts. Transformer-XL is the first self-attention
model that achieves substantially better results
than RNNs on both character-level and word-level

language modeling.


**2** **Related Work**


In the last few years, the field of language modeling has witnessed many significant advances,
including but not limited to devising novel architectures to better encode the context (Bengio
et al., 2003; Mikolov et al., 2010; Merity et al.,
2016; Al-Rfou et al., 2018), improving regularization and optimization algorithms (Gal and Ghahramani, 2016), speeding up the Softmax computation (Grave et al., 2016a), and enriching the output
distribution family (Yang et al., 2017).
To capture the long-range context in language
modeling, a line of work directly feeds a representation of the wider context into the network



as an additional input. Existing works range
from ones where context representations are manually defined (Mikolov and Zweig, 2012; Ji et al.,
2015; Wang and Cho, 2015) to others that rely on
document-level topics learned from data (Dieng
et al., 2016; Wang et al., 2017).
More broadly, in generic sequence modeling,
how to capture long-term dependency has been a
long-standing research problem. From this perspective, since the ubiquitous adaption of LSTM,
many efforts have been spent on relieving the
vanishing gradient problem, including better initialization (Le et al., 2015), additional loss signal (Trinh et al., 2018), augmented memory structure (Ke et al., 2018) and others that modify the internal architecture of RNNs to ease the optimization (Wu et al., 2016; Li et al., 2018). Different
from them, our work is based on the Transformer
architecture and shows that language modeling as
a real-world task benefits from the ability to learn
longer-term dependency.


**3** **Model**


Given a corpus of tokens **x** = ( _x_ 1 _, . . ., x_ _T_ ), the
task of language modeling is to estimate the joint
probability _P_ ( **x** ), which is often auto-regressively
factorized as _P_ ( **x** ) = [�] _t_ _[P]_ [(] _[x]_ _[t]_ _[ |]_ **[ x]** _[<t]_ [)][. With the]
factorization, the problem reduces to estimating
each conditional factor. In this work, we stick to
the standard neural approach to modeling the conditional probability. Specifically, a trainable neural network is used to encode the context **x** _<t_ into
a fixed size hidden state, which is multiplied with
the word embeddings to obtain the logits. The logits are then fed into the Softmax function, yielding
a categorical probability distribution over the next
token.


**3.1** **Vanilla Transformer Language Models**


In order to apply Transformer or self-attention to
language modeling, the central problem is how to
train a Transformer to effectively encode an arbitrarily long context into a fixed size representation.
Given infinite memory and computation, a simple solution would be to process the entire context sequence using an unconditional Transformer
decoder, similar to a feed-forward neural network.
However, this is usually infeasible with the limited
resource in practice.
One feasible but crude approximation is to split
the entire corpus into shorter segments of man

x 1 x 2 x 3 x 4


Segment 1



x 5 x 6 x 7 x 8


Segment 2



x 1 x 2 x 3 x 4 x 5 x 6


Limited Context



x 1 x 2 x 3 x 4 x 5 x 6


Limited Context


(b) Evaluation phase.



x 1 x 2 x 3 x 4 x 5 x 6


Limited Context



(a) Train phase.



Figure 1: Illustration of the vanilla model with a segment length 4.



ageable sizes, and only train the model within
each segment, ignoring all contextual information
from previous segments. This is the idea adopted
by Al-Rfou et al. (2018). We call it the _vanilla_
_model_ and visualize it in Fig. 1a. Under this
training paradigm, information never flows across
segments in either the forward or backward pass.
There are two critical limitations of using a fixedlength context. First, the largest possible dependency length is upper bounded by the segment
length, which is a few hundred on character-level
language modeling (Al-Rfou et al., 2018). Therefore, although the self-attention mechanism is less
affected by the vanishing gradient problem compared to RNNs, the vanilla model is not able to
fully exploit this optimization advantage. Second,
though it is possible to use padding to respect the
sentence or other semantic boundaries, in practice
it has been standard practice to simply chunk long
text into fixed-length segments due to improved
efficiency (Peters et al., 2018; Devlin et al., 2018;
Al-Rfou et al., 2018). However, simply chunking
a sequence into fixed-length segments will lead to
the context fragmentation problem as discussed in
Section 1.


During evaluation, at each step, the vanilla
model also consumes a segment of the same length
as in training, but only makes one prediction at the
last position. Then, at the next step, the segment
is shifted to the right by only one position, and the
new segment has to be processed all from scratch.
As shown in Fig. 1b, this procedure ensures that
each prediction utilizes the longest possible context exposed during training, and also relieves context fragmentation issue encountered in training.
However, this evaluation procedure is extremely
expensive. We will show that our proposed architecture is able to substantially improve the evaluation speed.



**3.2** **Segment-Level Recurrence with State**

**Reuse**


To address the limitations of using a fixed-length
context, we propose to introduce a recurrence
mechanism to the Transformer architecture. Dur
ing training, the hidden state sequence computed
for the previous segment is _fixed_ and _cached_ to
be reused as an extended context when the model

processes the next new segment, as shown in Fig.
2a. Although the gradient still remains within a
segment, this additional input allows the network
to exploit information in the history, leading to an
ability of modeling longer-term dependency and
avoiding context fragmentation. Formally, let the
two consecutive segments of length _L_ be **s** _τ_ =

[ _x_ _τ,_ 1 _, · · ·, x_ _τ,L_ ] and **s** _τ_ +1 = [ _x_ _τ_ +1 _,_ 1 _, · · ·, x_ _τ_ +1 _,L_ ]
respectively. Denoting the _n_ -th layer hidden state
sequence produced for the _τ_ -th segment **s** _τ_ by
**h** _[n]_ _τ_ _[∈]_ [R] _[L][×][d]_ [, where] _[ d]_ [ is the hidden dimension.]
Then, the _n_ -th layer hidden state for segment **s** _τ_ +1
is produced (schematically) as follows,
**h** � _[n]_ _τ_ +1 _[−]_ [1] [=] �SG( **h** _[n]_ _τ_ _[−]_ [1] ) _◦_ **h** _[n]_ _τ_ +1 _[−]_ [1] � _,_

**q** _τ_ _[n]_ +1 _[,]_ **[ k]** _τ_ _[n]_ +1 _[,]_ **[ v]** _τ_ _[n]_ +1 [=] **[ h]** _[n]_ _τ_ +1 _[−]_ [1] **[W]** _q_ _[⊤]_ _[,]_ [ �] **[h]** _[n]_ _τ_ +1 _[−]_ [1] **[W]** _k_ _[⊤]_ _[,]_ [ �] **[h]** _[n]_ _τ_ +1 _[−]_ [1] **[W]** _v_ _[⊤]_ _[,]_
**h** _[n]_ _τ_ +1 [=][ Transformer-Layer][ (] **[q]** _[n]_ _τ_ +1 _[,]_ **[ k]** _τ_ _[n]_ +1 _[,]_ **[ v]** _τ_ _[n]_ +1 [)] _[ .]_


where the function SG( _·_ ) stands for stop-gradient,
the notation [ **h** _u_ _◦_ **h** _v_ ] indicates the concatenation
of two hidden sequences along the length dimension, and **W** _·_ denotes model parameters. Compared to the standard Transformer, the critical difference lies in that the key **k** _[n]_ _τ_ +1 [and value] **[ v]** _τ_ _[n]_ +1
are conditioned on the extended context **h** [�] _[n]_ _τ_ +1 _[−]_ [1] [and]
hence **h** _[n]_ _τ_ _[−]_ [1] cached from the previous segment.
We emphasize this particular design by the green
paths in Fig. 2a.
With this recurrence mechanism applied to every two consecutive segments of a corpus, it essentially creates a segment-level recurrence in the
hidden states. As a result, the effective context being utilized can go way beyond just two segments.
However, notice that the recurrent dependency between **h** _[n]_ _τ_ +1 [and] **[ h]** _τ_ _[n][−]_ [1] shifts one layer downwards


x 1 x 2 x 3 x 4 x 5 x 6 x 7 x 8


Fixed (No Grad) New Segment



x 1 x 2 x 3 x 4 x 5 x 6 x 7 x 8


Fixed (No Grad)


(a) Training phase.



x 9 x 10 x 11 x 12


New Segment



x 1 x 2 x 3 x 4 x 5 x 6 x 7 x 8 x 9 x 10 x 11 x 12


Extended Context


(b) Evaluation phase.



Figure 2: Illustration of the Transformer-XL model with a segment length 4.



per-segment, which differs from the same-layer
recurrence in conventional RNN-LMs. Conse
quently, the largest possible dependency length
grows linearly w.r.t. the number of layers as well
as the segment length, i.e., _O_ ( _N × L_ ), as visualized by the shaded area in Fig. 2b. This
is analogous to truncated BPTT (Mikolov et al.,
2010), a technique developed for training RNNLMs. However, different from truncated BPTT,
our method caches a sequence of hidden states instead of the last one, and should be applied together with the relative positional encoding technique described in Section 3.3.
Besides achieving extra long context and resolving fragmentation, another benefit that comes
with the recurrence scheme is significantly faster
evaluation. Specifically, during evaluation, the
representations from the previous segments can
be reused instead of being computed from scratch
as in the case of the vanilla model. In our ex
periments on enwiki8, Transformer-XL is up to
1,800+ times faster than the vanilla model during
evaluation (see Section 4).

Finally, notice that the recurrence scheme does
not need to be restricted to only the previous segment. In theory, we can cache as many previous
segments as the GPU memory allows, and reuse
all of them as the extra context when processing
the current segment. Thus, we can cache a predefined length- _M_ old hidden states spanning (possibly) multiple segments, and refer to them as the
memory **m** _[n]_ _τ_ _[∈]_ [R] _[M]_ _[×][d]_ [, due to a clear connection to]
the memory augmented neural networks (Graves
et al., 2014; Weston et al., 2014). In our experiments, we set _M_ equal to the segment length during training, and increase it by multiple times during evaluation.


**3.3** **Relative Positional Encodings**


While we found the idea presented in the previous subsection very appealing, there is a crucial technical challenge we haven’t solved in or


der to reuse the hidden states. That is, how can
we keep the positional information coherent when
we reuse the states? Recall that, in the standard
Transformer, the information of sequence order is
provided by a set of positional encodings, denoted
as **U** _∈_ R _[L]_ [max] _[×][d]_, where the _i_ -th row **U** _i_ corresponds to the _i_ -th _absolute_ position within a segment and _L_ max prescribes the maximum possible
length to be modeled. Then, the actual input to the
Transformer is the element-wise addition of the

word embeddings and the positional encodings. If
we simply adapt this positional encoding to our
recurrence mechanism, the hidden state sequence
would be computed schematically by


**h** _τ_ +1 = _f_ ( **h** _τ_ _,_ **E** **s** _τ_ +1 + **U** 1: _L_ )

**h** _τ_ = _f_ ( **h** _τ_ _−_ 1 _,_ **E** **s** _τ_ + **U** 1: _L_ ) _,_


where **E** **s** _τ_ _∈_ R _[L][×][d]_ is the word embedding sequence of **s** _τ_, and _f_ represents a transformation
function. Notice that, both **E** **s** _τ_ and **E** **s** _τ_ +1 are associated with the same positional encoding **U** 1: _L_ .
As a result, the model has no information to distinguish the positional difference between _x_ _τ,j_ and
_x_ _τ_ +1 _,j_ for any _j_ = 1 _, . . ., L_, resulting in a sheer
performance loss.
In order to avoid this failure mode, the fundamental idea is to only encode the _relative_ positional information in the hidden states. Conceptually, the positional encoding gives the model a
temporal clue or “bias” about how information
should be gathered, i.e., where to attend. For the
same purpose, instead of incorporating bias statically into the initial embedding, one can inject the
same information into the attention score of each

layer. More importantly, it is more intuitive and
generalizable to define the temporal bias in a relative manner. For instance, when a query vector _q_ _τ,i_
attends on the key vectors **k** _τ,≤i_, it does not need
to know the absolute position of each key vector
to identify the temporal order of the segment. Instead, it suffices to know the relative distance between each key vector _k_ _τ,j_ and itself _q_ _τ,i_, i.e. _i_ _−_ _j_ .
Practically, one can create a set of relative posi

tional encodings **R** _∈_ R _[L]_ [max] _[×][d]_, where the _i_ -th row
**R** _i_ indicates a relative distance of _i_ between two
positions. By injecting the relative distance dynamically into the attention score, the query vector
can easily distinguish the representations of _x_ _τ,j_
and _x_ _τ_ +1 _,j_ from their different distances, making
the state reuse mechanism feasible. Meanwhile,
we won’t lose any temporal information, as the absolute position can be recovered recursively from
relative distances.

Previously, the idea of relative positional encodings has been explored in the context of machine
translation (Shaw et al., 2018) and music generation (Huang et al., 2018). Here, we offer a different derivation, arriving at a new form of relative positional encodings, which not only has a
one-to-one correspondence to its absolute counterpart but also enjoys much better generalization
empirically (see Section 4). Firstly, in the standard
Transformer (Vaswani et al., 2017), the attention
score between query _q_ _i_ and key vector _k_ _j_ within
the same segment can be decomposed as



+ **E** _[⊤]_ _x_ _i_ **[W]** _q_ _[⊤]_ **[W]** _k_ **[U]** _j_
~~�~~ ~~�~~ � ~~�~~
( _b_ )



**A** [abs] _i,j_ [=] **[ E]** _[⊤]_ _x_ _i_ **[W]** _q_ _[⊤]_ **[W]** _k_ **[E]** _x_ _j_
� ~~�~~ � ~~�~~
( _a_ )

+ **U** _[⊤]_ _i_ **[W]** _q_ _[⊤]_ **[W]** _k_ **[E]** _x_ _j_
� ~~��~~ �
( _c_ )



+ **U** _[⊤]_ _i_ **[W]** _q_ _[⊤]_ **[W]** _k_ **[U]** _j_
� �� ~~�~~
( _d_ )



_._



Following the idea of only relying on relative positional information, we propose to reparameterize the four terms as follows



**A** [rel] _i,j_ [=] **[ E]** _[⊤]_ _x_ _i_ **[W]** _q_ _[⊤]_ **[W]** _k,E_ **[E]** _x_ _j_ + **E** _[⊤]_ _x_ _i_ **[W]** _q_ _[⊤]_ **[W]** _k,R_ **[R]** _i−j_
� ~~�~~ � ~~�~~ � �� ~~�~~
( _a_ ) ( _b_ )



+ _u_ _[⊤]_ **W** _k,E_ **E** _x_ _j_
~~�~~ ~~�~~ � ~~�~~
( _c_ )



+ _v_ _[⊤]_ **W** _k,R_ **R** _i−j_
� ~~�~~ � ~~�~~
( _d_ )



_._




_•_ The first change we make is to replace all appearances of the absolute positional embedding
**U** _j_ for computing key vectors in term ( _b_ ) and
( _d_ ) with its relative counterpart **R** _i−j_ . This essentially reflects the prior that only the relative
distance matters for where to attend. Note that

**R** is a sinusoid encoding matrix (Vaswani et al.,
2017) without learnable parameters.


_•_ Secondly, we introduce a trainable parameter

_u ∈_ R _[d]_ to replace the query **U** _[⊤]_ _i_ **[W]** _q_ _[⊤]_ [in term]
( _c_ ). In this case, since the query vector is the
same for all query positions, it suggests that the
attentive bias towards different words should re
main the same regardless of the query position.
With a similar reasoning, a trainable parameter
_v ∈_ R _[d]_ is added to substitute **U** _[⊤]_ _i_ **[W]** _q_ _[⊤]_ [in term]
( _d_ ).




_•_ Finally, we deliberately separate the two weight
matrices **W** _k,E_ and **W** _k,R_ for producing the
content-based key vectors and location-based
key vectors respectively.
Under the new parameterization, each term has
an intuitive meaning: term ( _a_ ) represents contentbased addressing, term ( _b_ ) captures a contentdependent positional bias, term ( _c_ ) governs a
global content bias, and ( _d_ ) encodes a global positional bias.

In comparison, the formulation in Shaw et al.
(2018) only has terms ( _a_ ) and ( _b_ ), dropping the
two bias terms ( _c_ ) and ( _d_ ). Moreover, Shaw et al.
(2018) merge the multiplication **W** _k_ **R** into a single trainable matrix **R** [ˆ], which abandons the inductive bias built into the original sinusoid positional
encoding (Vaswani et al., 2017). In contrast, our
relative positional embedding **R** adapts the sinusoid formulation. As a benefit of the inductive
bias, a model trained on a memory of some certain
length can automatically generalize to a memory
several times longer during evaluation.
Equipping the recurrence mechanism with our
proposed relative positional embedding, we finally
arrive at the Transformer-XL architecture. For

completeness, we summarize the computational
procedure for a _N_ -layer Transformer-XL with a
single attention head here. For _n_ = 1 _, . . ., N_ :


�
**h** _[n]_ _τ_ _[−]_ [1] = �SG( **m** _[n]_ _τ_ _[−]_ [1] ) _◦_ **h** _[n]_ _τ_ _[−]_ [1] �

**q** _τ_ _[n]_ _[,]_ **[ k]** _[n]_ _τ_ _[,]_ **[ v]** _τ_ _[n]_ [=] **[ h]** _[n]_ _τ_ _[−]_ [1] **W** _q_ _[n]_ _⊤_ _,_ � **h** _nτ_ _−_ 1 **W** _k,E_ _[n]_ _⊤_ _,_ � **h** _nτ_ _−_ 1 **W** _v_ _[n]_ _⊤_


_⊤_ _n_ _⊤_ _n_
**A** _[n]_ _τ,i,j_ [=] **[ q]** _[n]_ _τ,i_ **k** _τ,j_ [+] **[ q]** _[n]_ _τ,i_ **W** _k,R_ **[R]** _i−j_

+ _u_ _[⊤]_ **k** _τ,j_ + _v_ _[⊤]_ **W** _k,R_ _[n]_ **[R]** _i−j_
**a** _[n]_ _τ_ [=][ Masked-Softmax][(] **[A]** _[n]_ _τ_ [)] **[v]** _τ_ _[n]_

**o** _[n]_ _τ_ [=][ LayerNorm][(][Linear][(] **[a]** _[n]_ _τ_ [) +] **[ h]** _[n]_ _τ_ _[−]_ [1] )
**h** _[n]_ _τ_ [=][ Positionwise-Feed-Forward][(] **[o]** _[n]_ _τ_ [)]


with **h** [0] _τ_ := **E** **s** _τ_ defined as the word embedding sequence. In addition, it is worth mentioning that a naive way to compute **A** requires computing **W** _k,R_ _[n]_ **[R]** _[i][−][j]_ [ for all pairs][ (] _[i, j]_ [)][, whose cost]
is quadratic w.r.t. the sequence length. However, noticing that the value of _i −_ _j_ only ranges
from zero to the sequence length, we show a simple computation procedure in Appendix B, which
reduces the cost to be linear w.r.t. the sequence
length.


**4** **Experiments**


**4.1** **Main Results**


We apply Transformer-XL to a variety of datasets
on both word-level and character-level language


**Model** **#Param PPL**


Grave et al. (2016b) - LSTM - 48.7
Bai et al. (2018) - TCN - 45.2
Dauphin et al. (2016) - GCNN-8 - 44.9
Grave et al. (2016b) - LSTM + Neural cache - 40.8
Dauphin et al. (2016) - GCNN-14 - 37.2
Merity et al. (2018) - QRNN 151M 33.0
Rae et al. (2018) - Hebbian + Cache - 29.9
Ours - Transformer-XL Standard 151M **24.0**


Baevski and Auli (2018) - Adaptive Input _[⋄]_ 247M 20.5
Ours - Transformer-XL Large 257M **18.3**


Table 1: Comparison with state-of-the-art results on
WikiText-103. _[⋄]_ indicates contemporary work.


**Model** **#Param bpc**


Ha et al. (2016) - LN HyperNetworks 27M 1.34
Chung et al. (2016) - LN HM-LSTM 35M 1.32
Zilly et al. (2016) - RHN 46M 1.27
Mujika et al. (2017) - FS-LSTM-4 47M 1.25
Krause et al. (2016) - Large mLSTM 46M 1.24
Knol (2017) - cmix v13  - 1.23
Al-Rfou et al. (2018) - 12L Transformer 44M 1.11
Ours - 12L Transformer-XL 41M **1.06**


Al-Rfou et al. (2018) - 64L Transformer 235M 1.06
Ours - 18L Transformer-XL 88M 1.03

Ours - 24L Transformer-XL 277M **0.99**


Table 2: Comparison with state-of-the-art results on enwik8.


modeling to have a comparison with state-of-theart systems, including WikiText-103 (Merity et al.,
2016), enwik8 (LLC, 2009), text8 (LLC, 2009),
One Billion Word (Chelba et al., 2013), and Penn
Treebank (Mikolov and Zweig, 2012).
WikiText-103 is the largest available word-level
language modeling benchmark with long-term dependency. It contains 103M training tokens from
28K articles, with an average length of 3.6K tokens per article, which allows testing the ability of long-term dependency modeling. We set
the attention length to 384 during training and
1600 during evaluation. We adopted adaptive softmax and input representations (Baevski and Auli,
2018; Grave et al., 2016a). As shown in Table 1,
Transformer-XL reduces the previous state-of-theart (SoTA) perplexity from 20.5 to 18.3, which
demonstrates the superiority of the TransformerXL architecture.

The dataset enwik8 contains 100M bytes of unprocessed Wikipedia text. We compare our architecture with the previous results in Table 2.
Under the model size constraint, the 12-layer
Transformer-XL achieves a new SoTA result, outperforming the 12-layer vanilla Transformer from
Al-Rfou et al. (2018) by 0.05, while both Trans


**Model** **#Param bpc**


Cooijmans et al. (2016) - BN-LSTM  - 1.36
Chung et al. (2016) - LN HM-LSTM 35M 1.29
Zilly et al. (2016) - RHN 45M 1.27
Krause et al. (2016) - Large mLSTM 45M 1.27
Al-Rfou et al. (2018) - 12L Transformer 44M 1.18


Al-Rfou et al. (2018) - 64L Transformer 235M 1.13
Ours - 24L Transformer-XL 277M **1.08**


Table 3: Comparison with state-of-the-art results on text8.


**Model** **#Param PPL**


Shazeer et al. (2014) - Sparse Non-Negative 33B 52.9
Chelba et al. (2013) - RNN-1024 + 9 Gram 20B 51.3
Kuchaiev and Ginsburg (2017) - G-LSTM-2 - 36.0
Dauphin et al. (2016) - GCNN-14 bottleneck - 31.9
Jozefowicz et al. (2016) - LSTM 1.8B 30.6
Jozefowicz et al. (2016) - LSTM + CNN Input 1.04B 30.0
Shazeer et al. (2017) - Low-Budget MoE _∼_ 5B 34.1
Shazeer et al. (2017) - High-Budget MoE _∼_ 5B 28.0
Shazeer et al. (2018) - Mesh Tensorflow 4.9B 24.0
Baevski and Auli (2018) - Adaptive Input _[⋄]_ 0.46B 24.1
Baevski and Auli (2018) - Adaptive Input _[⋄]_ 1.0B 23.7


Ours - Transformer-XL Base 0.46B 23.5
Ours - Transformer-XL Large 0.8B **21.8**


Table 4: Comparison with state-of-the-art results on One Billion Word. _[⋄]_ indicates contemporary work.


former variants have a large margin over conventional RNN-based models. Notably, our 12-layer
architecture achieves the same result as the 64
layer network from Al-Rfou et al. (2018), using
only 17% of the parameter budget. In order to see
whether better performances can be obtained by
increasing the model size, we train 18-layer and
24-layer Transformer-XLs with increased model
sizes. With the attention length 784 during training and 3,800 during evaluation, we obtained a
new SoTA result and our method is the first to
break through 1.0 on widely-studied characterlevel benchmarks. Different from Al-Rfou et al.

(2018), Transformer-XL does not need any auxiliary losses, and thus all benefits are credited to a
better architecture.

Similar to but different from enwik8, text8 contains 100M processed Wikipedia characters created by lowering case the text and removing any
character other than the 26 letters a through z, and
space. Due to the similarity, we simply adapt the
best model and the same hyper-parameters on enwik8 to text8 without further tuning. The comparison with previous methods is summarized in Table
3. Again, Transformer-XL achieves the new SoTA
result with a clear margin.


**Model** **#Param PPL**


Inan et al. (2016) - Tied Variational LSTM 24M 73.2
Zilly et al. (2016) - Variational RHN 23M 65.4
Zoph and Le (2016) - NAS Cell 25M 64.0
Merity et al. (2017) - AWD-LSTM 24M 58.8
Pham et al. (2018) - Efficient NAS 24M 58.6
Liu et al. (2018) - Differentiable NAS 23M 56.1
Yang et al. (2017) - AWD-LSTM-MoS 22M 55.97
Melis et al. (2018) - Dropout tuning 24M 55.3


Ours - Transformer-XL 24M **54.52**


Merity et al. (2017) - AWD-LSTM+Finetune _[†]_ 24M 57.3
Yang et al. (2017) - MoS+Finetune _[†]_ 22M **54.44**


Table 5: Comparison with state-of-the-art results on Penn
Treebank. _†_ indicates using two-step finetuning.


One Billion Word does not preserve any longterm dependency because sentences have been
shuffled. Consequently, this dataset mainly tests
the ability of modeling only short-term dependency. The comparison between Transformer-XL
and the other methods is shown in Table 4. Al
though Transformer-XL is mainly designed to better capture longer-term dependency, it dramatically improves the single-model SoTA from 23.7
to 21.8. Specifically, Transformer-XL significantly outperforms a contemporary method using
vanilla Transformers (Baevski and Auli, 2018),
suggesting the advantage of Transformer-XL is
generalizable to modeling short sequences.
We also report the results on word-level Penn
Treebank in Table 5. Similar to AWD-LSTM

(Merity et al., 2017), we apply variational dropout
and weight average to Transformer-XL. With
proper regularization, Transformer-XL achieves a
new SoTA result among models without two-step
finetuning. Penn Treebank has only 1M training
tokens, which implies that Transformer-XL also
generalizes well even on small datasets.


**4.2** **Ablation Study**


We conduct two sets of ablation studies to exam
ine the effects of two proposed techniques used in
Transformer-XL: the recurrence mechanism and

the new positional encoding scheme.
The first study is performed on WikiText-103,
which requires modeling long-term dependency.
The results are reported in Table 6. Among the
compared encoding schemes, Shaw et al. (2018) is
relative, while Vaswani et al. (2017) and Al-Rfou
et al. (2018) are absolute. “Full” and “half” losses
refer to applying a cross entropy loss to all or the
recent half positions in the segment. We found



that absolute encodings only work well with half
losses because half losses exclude positions with
very short attention lengths during training for better generalization. Table 6 shows that both the
recurrence mechanism and our encoding scheme
are necessary to achieve the best performance, as
well as generalizing to longer attention sequences
during evaluation time. Although the backpropagation length during training is only 128, with
the two techniques the attention length can be increased to 640 at test time. In the standard setting
with 151M parameters, the perplexity decreases as
the attention length increases.

Since the recurrence mechanism costs addi
tional memory, we also compare Transformer-XL
with baselines under the same GPU memory constraints. As shown in Table 10 in Appendix A,
despite using a shorter backpropagation length,
Transformer-XL remains superior to the baselines.

The second study targets at isolating the effects of resolving the context fragmentation problem from the benefit of capturing longer context
length. In order to achieve this goal, we deliberately choose a dataset that does not require longterm dependency, so that any improvement from
establishing the recurrence can be attributed to
solving the context fragmentation. Specifically,
we perform this controlled experiment on the One
Billion Word dataset, which can only benefit from
removing the context fragmentation. We train
a 20-layer Transformer-XL with _∼_ 0.3B parameters for 400K steps. As shown in Table 7, using
segment-level recurrence substantially improves
performance even when long-term dependency is
not needed, which is consistent with our previous
discussion that the recurrence mechanism resolves

the context fragmentation problem. Moreover, our
relative positional encodings is also superior to
Shaw et al. (2018) on short sequences.


**4.3** **Relative Effective Context Length**


Khandelwal et al. (2018) proposed a method to
evaluate the _Effective Context Length_ (ECL) of a
sequence model. ECL is the longest length to
which increasing the context span would lead to
a gain more than a threshold. However, ECL ignores the fact that it is harder to get improvement when a model already achieves a lower perplexity using only a shorter context, and thus it
is not suitable for fair comparison among multiple models. We instead propose a new metric


**Remark** **Recurrence** **Encoding** **Loss** **PPL init** **PPL best** **Attn Len**


Transformer-XL (128M)  Ours Full **27.02** **26.77** **500**
     -  Shaw et al. (2018) Full 27.94 27.94 256
     -  Ours Half 28.69 28.33 460

     -  Ours Full 29.59 29.02 260
     -  Ours Half 30.10 30.10 120


     -  Shaw et al. (2018) Full 29.75 29.75 120
     -  Shaw et al. (2018) Half 30.50 30.50 120
     -  Vaswani et al. (2017) Half 30.97 30.97 120
Transformer (128M) _[†]_  Al-Rfou et al. (2018) Half 31.16 31.16 120


**23.09** **640**
Transformer-XL (151M)  Ours Full 23.43 23.16 450
23.35 300


Table 6: Ablation study on WikiText-103. For the first two blocks, we use a slightly smaller model (128M parameters).

_†_ indicates that the corresponding row is reduced to the same setting as the Transformer network in (Al-Rfou et al., 2018),
except that two auxiliary losses are not implemented in our experiments. “PPL init” refers to using the same length as training.
“PPL best” indicates the perplexity obtained by using the optimal length. “Attn Len” is the shortest possible attention length
during evaluation to achieve the corresponding result (PPL best). Increasing the attention length during evaluation improves
performance only when our positional encoding is used. The “Transformer-XL (151M)” setting uses a standard parameter
budget as previous work (Merity et al., 2018), where we observe a similar effect when increasing the attention length during
evaluation.



**Method** **PPL**


Ours **25.2**
With Shaw et al. (2018) encodings 25.7
Without recurrence 27.1


Table 7: Ablation study on One Billion Word, a dataset without long-term dependency.


**Model** _r_ = 0 _._ 1 _r_ = 0 _._ 5 _r_ = 1 _._ 0


Transformer-XL 151M **900** **800** **700**
QRNN 500 400 300
LSTM 400 300 200


Transformer-XL 128M **700** **600** **500**

 - use Shaw et al. (2018) encoding 400 400 300

 - remove recurrence 300 300 300

Transformer 128 128 128


Table 8: Relative effective context length (RECL) comparison. See text for the definition of RECL and _r_ . The first three
models and the last four models are compared as two _model_
_groups_ when we calculate RECL (RECL is computed on a
model group rather than a single model). Each group has the
same parameter budget.


called _Relative Effective Context Length_ (RECL).
RECL is defined on a model group instead of a
single model, and the gain of a long context is
measure by the relative improvement over the _best_
short context model. As such, the model group
shares the same baseline to enable fair comparison. RECL also has a parameter _r_, which means
constraining the comparison on top- _r_ hard examples. See Appedix C for more details about RECL.
As shown in Table 8, Transformer-XL manages
to model dependency of 900 words long on av


**Attn Len** **How much Al-Rfou et al. (2018) is slower**


3,800 1,874x
2,800 1,409x
1,800 773x
800 363x


Table 9: Slowdown in terms of running time during evaluation. Evaluation is based on per-token time on one GPU.


erage with _r_ = 0 _._ 1. The RECL of TransformerXL is 80% and 450% longer than recurrent networks and Transformer respectively. Both the recurrence mechanism and our positional encodings
contribute to a longer RECL. This further substantiates our argument that Transformer-XL is able to
model longer-term dependency.


**4.4** **Generated Text**


Trained only on WikiText-103 which is mediumsized, Transformer-XL is already able to generate
relatively coherent articles with thousands of tokens without manual cherry picking, despite minor flaws. Please refer to Appendix E for samples.


**4.5** **Evaluation Speed**


Finally, we compare the evaluation speed of our
model with the vanilla Transformer model (AlRfou et al., 2018). As shown in Table 9, due to
the state reuse scheme, Transformer-XL achieves
an up to 1,874 times speedup during evaluation.


**5** **Conclusions**


Transformer-XL obtains strong perplexity results,
models longer-term dependency than RNNs and
Transformer, achieves substantial speedup during
evaluation, and is able to generate coherent text
articles. We envision interesting applications of
Transformer-XL in the fields of text generation,
unsupervised feature learning, image and speech
modeling.


**Acknowledgments**

ZD and YY were supported in part by National
Science Foundation (NSF) under the grant IIS1546329 and by the DOE-Office of Science under the grant ASCR #KJ040201. ZY and RS
were supported in part by the Office of Naval
Research grant N000141812861, the NSF grant
IIS1763562, the Nvidia fellowship, and the Siebel
scholarship.


**References**


Rami Al-Rfou, Dokook Choe, Noah Constant, Mandy
Guo, and Llion Jones. 2018. Character-level language modeling with deeper self-attention. _arXiv_
_preprint arXiv:1808.04444_ .


Alexei Baevski and Michael Auli. 2018. Adaptive input representations for neural language modeling.
_arXiv preprint arXiv:1809.10853_ .


Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2014. Neural machine translation by jointly
learning to align and translate. _arXiv preprint_
_arXiv:1409.0473_ .


Shaojie Bai, J Zico Kolter, and Vladlen Koltun.
2018. An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. _arXiv preprint arXiv:1803.01271_ .


Yoshua Bengio, Réjean Ducharme, Pascal Vincent, and
Christian Jauvin. 2003. A neural probabilistic language model. _Journal of machine learning research_,
3(Feb):1137–1155.


Ciprian Chelba, Tomas Mikolov, Mike Schuster, Qi Ge,
Thorsten Brants, Phillipp Koehn, and Tony Robinson. 2013. One billion word benchmark for measuring progress in statistical language modeling. _arXiv_
_preprint arXiv:1312.3005_ .


Junyoung Chung, Sungjin Ahn, and Yoshua Bengio.
2016. Hierarchical multiscale recurrent neural networks. _arXiv preprint arXiv:1609.01704_ .


Tim Cooijmans, Nicolas Ballas, César Laurent, Ça˘glar
Gülçehre, and Aaron Courville. 2016. Recurrent batch normalization. _arXiv_ _preprint_
_arXiv:1603.09025_ .



Andrew M Dai and Quoc V Le. 2015. Semi-supervised
sequence learning. In _Advances in neural informa-_
_tion processing systems_, pages 3079–3087.


Yann N Dauphin, Angela Fan, Michael Auli, and
David Grangier. 2016. Language modeling with
gated convolutional networks. _arXiv preprint_
_arXiv:1612.08083_ .


Jacob Devlin, Ming-Wei Chang, Kenton Lee, and
Kristina Toutanova. 2018. Bert: Pre-training of deep
bidirectional transformers for language understanding. _arXiv preprint arXiv:1810.04805_ .


Adji B Dieng, Chong Wang, Jianfeng Gao, and John
Paisley. 2016. Topicrnn: A recurrent neural network with long-range semantic dependency. _arXiv_
_preprint arXiv:1611.01702_ .


Yarin Gal and Zoubin Ghahramani. 2016. A theoretically grounded application of dropout in recurrent
neural networks. In _Advances in neural information_
_processing systems_, pages 1019–1027.


Edouard Grave, Armand Joulin, Moustapha Cissé,
David Grangier, and Hervé Jégou. 2016a. Efficient
softmax approximation for gpus. _arXiv preprint_
_arXiv:1609.04309_ .


Edouard Grave, Armand Joulin, and Nicolas
Usunier. 2016b. Improving neural language
models with a continuous cache. _arXiv preprint_
_arXiv:1612.04426_ .


Alex Graves. 2013. Generating sequences with
recurrent neural networks. _arXiv_ _preprint_
_arXiv:1308.0850_ .


Alex Graves, Greg Wayne, and Ivo Danihelka.
2014. Neural turing machines. _arXiv preprint_
_arXiv:1410.5401_ .


David Ha, Andrew Dai, and Quoc V Le. 2016. Hypernetworks. _arXiv preprint arXiv:1609.09106_ .


Sepp Hochreiter, Yoshua Bengio, Paolo Frasconi, Jürgen Schmidhuber, et al. 2001. Gradient flow in recurrent nets: the difficulty of learning long-term dependencies.


Sepp Hochreiter and Jürgen Schmidhuber. 1997.
Long short-term memory. _Neural computation_,
9(8):1735–1780.


Cheng-Zhi Anna Huang, Ashish Vaswani, Jakob
Uszkoreit, Noam Shazeer, Curtis Hawthorne, Andrew M Dai, Matthew D Hoffman, and Douglas Eck.
2018. An improved relative self-attention mechanism for transformer with application to music generation. _arXiv preprint arXiv:1809.04281_ .


Hakan Inan, Khashayar Khosravi, and Richard Socher.
2016. Tying word vectors and word classifiers:
A loss framework for language modeling. _arXiv_
_preprint arXiv:1611.01462_ .


Yangfeng Ji, Trevor Cohn, Lingpeng Kong, Chris Dyer,
and Jacob Eisenstein. 2015. Document context language models. _arXiv preprint arXiv:1511.03962_ .


Rafal Jozefowicz, Oriol Vinyals, Mike Schuster, Noam
Shazeer, and Yonghui Wu. 2016. Exploring
the limits of language modeling. _arXiv preprint_
_arXiv:1602.02410_ .


Nal Kalchbrenner, Lasse Espeholt, Karen Simonyan,
Aaron van den Oord, Alex Graves, and Koray
Kavukcuoglu. 2016. Neural machine translation in
linear time. _arXiv preprint arXiv:1610.10099_ .


Sekitoshi Kanai, Yasuhiro Fujiwara, Yuki Yamanaka,
and Shuichi Adachi. 2018. Sigsoftmax: Reanalysis of the softmax bottleneck. _arXiv preprint_
_arXiv:1805.10829_ .


Nan Rosemary Ke, Anirudh Goyal ALIAS PARTH
GOYAL, Olexa Bilaniuk, Jonathan Binas,
Michael C Mozer, Chris Pal, and Yoshua Bengio. 2018. Sparse attentive backtracking: Temporal
credit assignment through reminding. In _Advances_
_in Neural Information Processing Systems_, pages
7650–7661.


Urvashi Khandelwal, He He, Peng Qi, and Dan Jurafsky. 2018. Sharp nearby, fuzzy far away: How
neural language models use context. _arXiv preprint_
_arXiv:1805.04623_ .


Bryon Knol. 2017. cmix v13. [http://www.](http://www.byronknoll.com/cmix.html)
[byronknoll.com/cmix.html.](http://www.byronknoll.com/cmix.html)


Jan Koutnik, Klaus Greff, Faustino Gomez, and Juergen Schmidhuber. 2014. A clockwork rnn. _arXiv_
_preprint arXiv:1402.3511_ .


Ben Krause, Liang Lu, Iain Murray, and Steve Renals.
2016. Multiplicative lstm for sequence modelling.
_arXiv preprint arXiv:1609.07959_ .


Oleksii Kuchaiev and Boris Ginsburg. 2017. Factorization tricks for lstm networks. _arXiv preprint_
_arXiv:1703.10722_ .


Quoc V Le, Navdeep Jaitly, and Geoffrey E Hinton. 2015. A simple way to initialize recurrent
networks of rectified linear units. _arXiv preprint_
_arXiv:1504.00941_ .


Shuai Li, Wanqing Li, Chris Cook, Ce Zhu, and Yanbo
Gao. 2018. Independently recurrent neural network
(indrnn): Building a longer and deeper rnn. In _Pro-_
_ceedings of the IEEE Conference on Computer Vi-_
_sion and Pattern Recognition_, pages 5457–5466.


Hanxiao Liu, Karen Simonyan, and Yiming Yang.
2018. Darts: Differentiable architecture search.
_arXiv preprint arXiv:1806.09055_ .


MultiMedia LLC. 2009. Large text compression
benchmark.



Gábor Melis, Charles Blundell, Tomáš Koˇcisk`y,
Karl Moritz Hermann, Chris Dyer, and Phil Blunsom. 2018. Pushing the bounds of dropout. _arXiv_
_preprint arXiv:1805.09208_ .


Stephen Merity, Nitish Shirish Keskar, and Richard
Socher. 2017. Regularizing and optimizing lstm language models. _arXiv preprint arXiv:1708.02182_ .


Stephen Merity, Nitish Shirish Keskar, and Richard
Socher. 2018. An analysis of neural language
modeling at multiple scales. _arXiv preprint_
_arXiv:1803.08240_ .


Stephen Merity, Caiming Xiong, James Bradbury, and
Richard Socher. 2016. Pointer sentinel mixture
models. _arXiv preprint arXiv:1609.07843_ .


Tomas Mikolov, Armand Joulin, Sumit Chopra,
Michael Mathieu, and Marc’Aurelio Ranzato. 2014.
Learning longer memory in recurrent neural networks. _arXiv preprint arXiv:1412.7753_ .


Tomáš Mikolov, Martin Karafiát, Lukáš Burget, Jan
ˇCernock`y, and Sanjeev Khudanpur. 2010. Recurrent neural network based language model. In
_Eleventh Annual Conference of the International_
_Speech Communication Association_ .


Tomas Mikolov and Geoffrey Zweig. 2012. Context
dependent recurrent neural network language model.
_SLT_, 12(234-239):8.


Frederic Morin and Yoshua Bengio. 2005. Hierarchical probabilistic neural network language model. In
_Aistats_, volume 5, pages 246–252. Citeseer.


Asier Mujika, Florian Meier, and Angelika Steger.
2017. Fast-slow recurrent neural networks. In _Ad-_
_vances in Neural Information Processing Systems_,
pages 5915–5924.


Razvan Pascanu, Tomas Mikolov, and Yoshua Bengio.
2012. Understanding the exploding gradient problem. _CoRR, abs/1211.5063_ .


Matthew E Peters, Mark Neumann, Mohit Iyyer, Matt
Gardner, Christopher Clark, Kenton Lee, and Luke
Zettlemoyer. 2018. Deep contextualized word representations. _arXiv preprint arXiv:1802.05365_ .


Hieu Pham, Melody Y Guan, Barret Zoph, Quoc V
Le, and Jeff Dean. 2018. Efficient neural architecture search via parameter sharing. _arXiv preprint_
_arXiv:1802.03268_ .


Ofir Press and Lior Wolf. 2016. Using the output
embedding to improve language models. _arXiv_
_preprint arXiv:1608.05859_ .


Alec Radford, Karthik Narasimhan, Tim Salimans, and
Ilya Sutskever. 2018. Improving language understanding by generative pre-training. _URL https://s3-_
_us-west-2. amazonaws. com/openai-assets/research-_
_covers/languageunsupervised/language_ _under-_
_standing paper. pdf_ .


Jack W Rae, Chris Dyer, Peter Dayan, and Timothy P Lillicrap. 2018. Fast parametric learning with activation memorization. _arXiv preprint_
_arXiv:1803.10049_ .


Peter Shaw, Jakob Uszkoreit, and Ashish Vaswani.
2018. Self-attention with relative position representations. _arXiv preprint arXiv:1803.02155_ .


Noam Shazeer, Youlong Cheng, Niki Parmar, Dustin
Tran, Ashish Vaswani, Penporn Koanantakool, Peter
Hawkins, HyoukJoong Lee, Mingsheng Hong, Cliff
Young, et al. 2018. Mesh-tensorflow: Deep learning
for supercomputers. In _Advances in Neural Infor-_
_mation Processing Systems_, pages 10434–10443.


Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz,
Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff
Dean. 2017. Outrageously large neural networks:
The sparsely-gated mixture-of-experts layer. _arXiv_
_preprint arXiv:1701.06538_ .


Noam Shazeer, Joris Pelemans, and Ciprian Chelba.
2014. Skip-gram language modeling using sparse
non-negative matrix probability estimation. _arXiv_
_preprint arXiv:1412.1454_ .


Trieu H Trinh, Andrew M Dai, Thang Luong, and
Quoc V Le. 2018. Learning longer-term dependencies in rnns with auxiliary losses. _arXiv preprint_
_arXiv:1803.00144_ .


Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob
Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz
Kaiser, and Illia Polosukhin. 2017. Attention is all
you need. In _Advances in Neural Information Pro-_
_cessing Systems_, pages 5998–6008.


Tian Wang and Kyunghyun Cho. 2015. Largercontext language modelling. _arXiv preprint_
_arXiv:1511.03729_ .


Wenlin Wang, Zhe Gan, Wenqi Wang, Dinghan Shen,
Jiaji Huang, Wei Ping, Sanjeev Satheesh, and
Lawrence Carin. 2017. Topic compositional neural
language model. _arXiv preprint arXiv:1712.09783_ .


Jason Weston, Sumit Chopra, and Antoine Bordes. 2014. Memory networks. _arXiv preprint_
_arXiv:1410.3916_ .


Yuhuai Wu, Saizheng Zhang, Ying Zhang, Yoshua
Bengio, and Ruslan R Salakhutdinov. 2016. On
multiplicative integration with recurrent neural networks. In _Advances in neural information process-_
_ing systems_, pages 2856–2864.


Zhilin Yang, Zihang Dai, Ruslan Salakhutdinov, and
William W Cohen. 2017. Breaking the softmax bottleneck: A high-rank rnn language model. _arXiv_
_preprint arXiv:1711.03953_ .


Wojciech Zaremba, Ilya Sutskever, and Oriol Vinyals.
2014. Recurrent neural network regularization.
_arXiv preprint arXiv:1409.2329_ .



Julian Georg Zilly, Rupesh Kumar Srivastava,
Jan Koutník, and Jürgen Schmidhuber. 2016.
Recurrent highway networks. _arXiv preprint_
_arXiv:1607.03474_ .


Barret Zoph and Quoc V Le. 2016. Neural architecture
search with reinforcement learning. _arXiv preprint_
_arXiv:1611.01578_ .


**A** **Ablation Study with Memory Constraints**


**Backprop Len** **Recurrence** **Encoding** **Loss** **pplx best** **pplx init** **Attn Len**


128  Ours Full **26.77** **27.02** **500**

128  Ours Partial 28.33 28.69 460


176  Ours Full 27.98 28.43 400
172  Ours Partial 28.83 28.83 120


Table 10: Ablation study on WikiText-103 with the same GPU memory constraints.


Table 10 compares Transformer-XL with baseline under the same memory budget. Transformer-XL
still outperforms the baseline even with a shorter backprop length.


**B** **Efficient Computation of the Attention with Relative Positional Embedding**


As we discussed in section 3.3, the naive way of computing the **W** _k,R_ **R** _i−j_ for all pairs ( _i, j_ ) is subject
to a quadratic cost. Here, we present a simple method with only a linear cost. Firstly, notice that the
relative distance _i −_ _j_ can only be integer from 0 to _M_ + _L −_ 1, where _M_ and _L_ are the memory length
and segment length respectively. Hence, the rows of the matrix





_∈_ R ( _M_ + _L_ ) _×d_











[ **W** _k,R_ **R** _M_ + _L−_ 1 ] _[⊤]_

[ **W** _k,R_ **R** _M_ + _L−_ 2 ] _[⊤]_



...

[ **W** _k,R_ **R** 1 ] _[⊤]_

[ **W** _k,R_ **R** 0 ] _[⊤]_



**Q** :=



**R** _[⊤]_ _M_ + _L−_ 1

 **R** _[⊤]_ _M_ + _L−_ 2


...
**R** _[⊤]_ 1

 **R** _[⊤]_ 0





_⊤_
**W** =
_k,R_




consist of all possible vector outputs of **W** _k,R_ **R** _i−j_ for any ( _i, j_ ). Note that we have defined **Q** in a
reversed order, i.e., **Q** _k_ = **W** _k,R_ **R** _M_ + _L−_ 1 _−k_, to make further discussion easier.
Next, we collect the term ( _b_ ) for all possible _i, j_ into the following _L ×_ ( _M_ + _L_ ) matrix,









**B** =


=













_q_ 0 _[⊤]_ **[W]** _[k,R]_ **[R]** _[M]_ _· · ·_ _q_ 0 _[⊤]_ **[W]** _[k,R]_ **[R]** [0] 0 _· · ·_ 0
_q_ 1 _[⊤]_ **[W]** _[k,R]_ **[R]** _[M]_ [+1] _· · ·_ _q_ 1 _[⊤]_ **[W]** _[k,R]_ **[R]** [1] _q_ 1 _[⊤]_ **[W]** _[k,R]_ **[R]** [0] _· · ·_ 0

... ... ... ... ... ...
_q_ _L_ _[⊤]_ _−_ 1 **[W]** _[k,R]_ **[R]** _[M]_ [+] _[L][−]_ [1] _· · ·_ _q_ _L_ _[⊤]_ _−_ 1 **[W]** _[k,R]_ **[R]** _[M]_ [+] _[L][−]_ [1] _q_ _L_ _[⊤]_ _−_ 1 **[W]** _[k,R]_ **[R]** _[L][−]_ [1] _· · ·_ _q_ _L_ _[⊤]_ _−_ 1 **[W]** _[k,R]_ **[R]** [0]



_q_ 0 _[⊤]_ **[Q]** _[L][−]_ [1] _· · ·_ _q_ 0 _[⊤]_ **[Q]** _[M]_ [+] _[L][−]_ [1] 0 _· · ·_ 0
_q_ 1 _[⊤]_ **[Q]** _[L][−]_ [2] _· · ·_ _q_ 1 _[⊤]_ **[Q]** _[M]_ [+] _[L][−]_ [2] _q_ 1 _[⊤]_ **[Q]** _[M]_ [+] _[L][−]_ [1] _· · ·_ 0

... ... ... ... ... ...
_q_ _L_ _[⊤]_ _−_ 1 **[Q]** [0] _· · ·_ _q_ _L_ _[⊤]_ _−_ 1 **[Q]** _[M]_ _q_ _L_ _[⊤]_ _−_ 1 **[Q]** _[M]_ [+1] _· · ·_ _q_ _L_ _[⊤]_ _−_ 1 **[Q]** _[M]_ [+] _[L][−]_ [1]









Then, we further define


�
**B** = **qQ** _[⊤]_ =



_q_ 0 _[⊤]_ **[Q]** [0] _· · ·_ _q_ 0 _[⊤]_ **[Q]** _[M]_ _q_ 0 _[⊤]_ **[Q]** _[M]_ [+1] _· · ·_ _q_ 0 _[⊤]_ **[Q]** _[M]_ [+] _[L][−]_ [1]

 _q_ 1 _[⊤]_ **[Q]** [0] _· · ·_ _q_ 1 _[⊤]_ **[Q]** _[M]_ _q_ 1 _[⊤]_ **[Q]** _[M]_ [+1] _· · ·_ _q_ 1 _[⊤]_ **[Q]** _[M]_ [+] _[L][−]_ [1]


... ... ... ... ... ...

 _q_ _L_ _[⊤]_ _−_ 1 **[Q]** [0] _· · ·_ _q_ _L_ _[⊤]_ _−_ 1 **[Q]** _[M]_ _q_ _L_ _[⊤]_ _−_ 1 **[Q]** _[M]_ [+1] _· · ·_ _q_ _L_ _[⊤]_ _−_ 1 **[Q]** _[M]_ [+] _[L][−]_ [1]






_._




Now, it is easy to see an immediate relationship between **B** and **B** [�], where the _i_ -th row of **B** is simply a
left-shifted version of _i_ -th row of **B** [�] . Hence, the computation of **B** only requires a matrix multiplication
**qQ** _[⊤]_ to compute **B** [�] and then a set of left-shifts.
Similarly, we can collect all term ( _d_ ) for all possible _i, j_ into another _L ×_ ( _M_ + _L_ ) matrix **D**,






_._




**D** =



_v_ _[⊤]_ **Q** _L−_ 1 _· · ·_ _v_ _[⊤]_ **Q** _M_ + _L−_ 1 0 _· · ·_ 0

 _v_ _[⊤]_ **Q** _L−_ 2 _· · ·_ _v_ _[⊤]_ **Q** _M_ + _L−_ 2 _v_ _[⊤]_ **Q** _M_ + _L−_ 1 _· · ·_ 0


... ... ... ... ... ...

 _v_ _[⊤]_ **Q** 0 _· · ·_ _v_ _[⊤]_ **Q** _M_ _v_ _[⊤]_ **Q** _M_ +1 _· · ·_ _v_ _[⊤]_ **Q** _M_ + _L−_ 1


Then, we can follow the same procedure to define


�
**d** = [ **Q** _v_ ] _[⊤]_ = � _v_ _[⊤]_ **Q** 0 _· · ·_ _v_ _[⊤]_ **Q** _M_ _v_ _[⊤]_ **Q** _M_ +1 _· · ·_ _v_ _[⊤]_ **Q** _M_ + _L−_ 1 � _._


Again, each row of **D** is simply a left-shift version of **d** [�] . Hence, the main computation cost comes from
the matrix-vector multiplication **d** [�] = [ **Q** _v_ ] _[⊤]_, which is not expensive any more.


**C** **Details About RECL**


(a) Transformer-XL vs RNNs (b) Transformer-XL vs Baseline


Figure 3: Visualizing unnormalized relative perplexity gains with _r_ = 0 _._ 1.


(a) Transformer-XL vs RNNs (b) Transformer-XL vs Baseline


Figure 4: Perplexity vs context length.


In this section, we describe the details of the metric RECL. Let _M_ = _{m_ 1 _, m_ 2 _, · · ·, m_ _N_ _}_ be a model
group consisting of _N_ models. Let _l_ _i_ ( _c, t_ ) denote the loss of model _m_ _i_ on the _t_ -th token in the corpus
with a context length _c_ . Concretely, the loss can be written as


_l_ _i_ ( _c, t_ ) = _−_ log _P_ _m_ _i_ ( _x_ _t_ _|x_ _t−_ 1 _, · · ·, x_ _t−c_ )


where _P_ _m_ _i_ is the probability distribution given by model _m_ _i_, and _x_ _t_ is the _t_ -th token in the corpus. Given
a short context length _c_ and a long context length _c_ _[′]_ such that _c_ _[′]_ _≥_ _c_, we can further define a baseline for
each position _t_,

_N_
_b_ ( _c, t_ ) = min
_i_ =1 _[l]_ _[i]_ [(] _[c, t]_ [)]


The _relative loss_ of _m_ _i_ w.r.t. the model group _M_ is written as



1
_f_ _i_ ( _c, c_ _[′]_ ) =
_|T |_



� min � _b_ ( _c, t_ ) _, l_ _i_ ( _c_ _[′]_ _, t_ )�

_t∈T_



The above equation uses the minimum loss of all models on the short length _c_ as a baseline, and only
losses smaller than the baseline will be effectively counted towards the relative loss. This enables fair


comparison between multiple models because all models with a long context length _c_ _[′]_ need to improve
over the same baseline. Sometimes we only care about those positions where the baseline performs
poorly (which means short-term dependency with context length _c_ is not sufficient), so given a ratio
parameter _r_, we define the set _T_ is the above equation as


_T_ = top- _r_ positions _t_ with largest _b_ ( _c, t_ )


The _relative gain_ is subsequently defined as the relative perplexity reduction:


_[f]_ _[i]_ [(] _[c][,][ c]_ [)] _[ −]_ [ex][p] _[f]_ _[i]_ [(] _[c][,][ c]_ _[′]_ [)]
_g_ _i_ ( _c, c_ _[′]_ ) = [ex][p]

exp _f_ _i_ ( _c, c_ )


Given a step size ∆, we then use an algorithm to find the RECL by thresholding the relative gain:


1. Set initial short context length _c_, and long context length _c_ _[′]_ = _c_ + ∆


2. Compute _g_ _i_ ( _c, c_ _[′]_ ). If _g_ _i_ ( _c, c_ _[′]_ ) _<_ 0 _._ 01, return RECL = _c_ . If _g_ _i_ ( _c, c_ _[′]_ ) _≥_ 0 _._ 01, set _c_ = _c_ _[′]_ _, c_ _[′]_ = _c_ + ∆
and go to step 1.


In Figure 3, we visualize the unnormalized relative perplexity gains (exp _f_ _i_ ( _c, c_ ) _−_ exp _f_ _i_ ( _c, c_ _[′]_ )) with
various pairs of ( _c, c_ _[′]_ ) when _r_ = 0 _._ 1. It is clear that Transformer-XL has a longer RECL compared to
RNNs and other baselines because the relative gains are substantially larger.
For reference, we plot the perplexities with varying context lengths in Figure 4. The y-axis denotes
the “normal” perplexity (not calibrated by baselines).


**D** **Attention Visualization**


In this section, we provide some visualization of the attention learned by the SoTA model on the
WikiText-103 validation set. Recall that, this model has 16 10-head transformer layers and relies on
a memory of length 640.


Figure 5: Average attention over the previous 640 tokens, where each row corresponds to a attention head and each
column corresponds to a relative location. There are totally 160 attention heads, and every 10 heads come from a
single layer. Darker colors indicate higher values.


The first visualization aims at revealing the overall trend of where the model is attending. Specifically,
for each attention head of each layer, we average the attention distributions of all tokens in the validation
set. This is shown in Fig. 5. As we can see, the overall trend is to focus more on the nearby tokens
than the faraway ones. However, it is also very clear that some attention heads have a wider attention
distribution over the entire memory span, notably the head 8 from layer 1, head 78 from layer 8, and the
head 158 from layer 16.
Since we are focused on learning long-range dependency, we are especially interested in these heads
with a wider attention span. Thus, in the second set of visualization, we pick the three notable heads
mentioned above, and visualize their attention behavior for a randomly chosen position, as shown in Fig.
6. Here, we see three different patterns of wider attention:


_•_ For the head 8 in the 1st layer, we see an almost uniform attention over the entire memory span. This
is quite intuitive, as lower-level layers needs to screen the entire memory span to decide where to focus
for higher-level layers


(a) Head 8 from layer 1.


(b) Head 78 from layer 8.


(c) Head 158 from layer 16.


Figure 6: Visualization of the three heads with a wide attention range. Each row corresponds to a target location/token and each column corresponds to a context location/token. Tokens in the memory that have top 20%
attention values are highlighted in red.


_•_ For the head 78 in the 8th layer (a middle-level layer), we see a very sparse attention pattern scattered
in all ranges of the memory. Again, this well fits our intuition that as information accumulates, the
network may focus on some particular position with special interests.


_•_ For the head 158 in the 16th layer (i.e. the last layer), each target location (corresponding to each row)
has its own distinct sparse focus, differing from head 78 where target locations largely share the same
attentive location in memory. Meanwhile, the pattern is also different from the case of head 8, where
a few locations are clearly attended more than others.


Finally, as we have discussed in section 3.3, the attention score can be decomposed into four intuitive
terms. Here, we want to further investigate how these four terms contribute to the overall attention trend
in Fig. 5. Since the term ( _c_ ) represents the global content bias, i.e., the prior importance of each word
regardless of the context, we will leave it out and focus on the terms ( _a_ ), ( _b_ ) and ( _d_ ). So, for each term,
we take the Softmax w.r.t. the memory span and average the resulted distribution of all tokens in the
validation set. The results are visualized in Fig. 7:


_•_ Since term ( _a_ ) is fully content-based addressing, when averaging over all target words, the result is
essentially uniform over the entire context, except for a few very close words, which are likely to be
semantically similar to the target word.


_•_ The overall trend of term ( _b_ ) highly resembles that of the entire attention distribution in Fig. 5. It
suggests that the global trend of focusing on the nearby context is largely contributed by this contentdependent positional bias.


_•_ The overall trend of term ( _d_ ) is also focusing more on nearby words. However, compared to the trend
of term ( _b_ ), it is clearly flatter and biases towards a longer context.


(a) Term ( _a_ ).


(b) Term ( _b_ ).


(c) Term ( _d_ ).


Figure 7: Visualization of the three terms in computing the attention score. Each row corresponds to a attention
head and each column corresponds to a relative location.


**E** **Generated Text**


In this section, we present some generated text from our best model trained the Wikitext-103 dataset.
We seed the our Transformer-XL with a context of at most 512 consecutive tokens randomly sampled
from the test set of Wikitext-103. Then, we run Transformer-XL to generate a _pre-defined_ number of
tokens (500 or 1,000 in our case). For each generation step, we first find the top-40 probabilities of the
next-step distribution and sample from top-40 tokens based on the re-normalized distribution. To help
reading, we detokenize the context, the generated text and the reference text. Three generated examples
are shown in Tables 11, 12, and 13. Note that we do not perform any cherry picking and present the
first three examples we generate in the paper. In the text, “= text =”, “= = text = =” and “= = = text = =
=” denote the Wikipedia page tile, section title and subsection title, respectively, due to the original data
preprocessing procedure of Wikitext-103 (Merity et al., 2016).

As we can see, though only trained on 100M tokens, Transformer-XL is a strong model at generating
long text articles, particularly in the following aspects:


_•_ Transformer-XL is able to structurally maintain the sectional arrangement of Wikipedia.

_•_ Transformer-XL manages to semantically stay on the same topic throughout the course of generation.

_•_ Long-range references are common in the generated text.

_•_ Transformer-XL often generates novel content that is not present in the training data.

For more detailed explanation of the interesting observations in each example, please refer to the corresponding caption.


Despite the overall excellence of the generation quality, the model can only perceive the seed context
and hallucinate what to generate based on the limited knowledge (100M tokens only) it is trained on.
As a result, the generated text sometimes looks clearly relevant but not close enough or to the point
compared to what human writer would do. That said, we believe this issue is mostly a problem of limited
training data size and could be alleviated by using a larger training set.


**Context:**
Kershaw started the 2010 season by posting a 3.07 ERA in April, but did so by walking 22 batters in 29 innings. On May 4, he had his worst start of his career
against the Milwaukee Brewers at Dodger Stadium, throwing just 57 pitches in 11 / 3 innings, while retiring only four of the 13 batters he faced — including the
pitcher. He was booed loudly upon being pulled from the game. Kershaw said after the game, " I didn’t give our team any kind of chance. It’s just not a good
feeling to let your teammates down, let everybody down. It stings, it hurts. I ’ve got to figure things out. " Kershaw rebounded his next start by pitching an 8 inning
two-hitter and out-dueling the then undefeated Ubaldo Jiménez. He credited his control of the slider being the major turning point for him. Later in the season, he
was suspended for five games after hitting Aaron Rowand of the Giants with a pitch in a game on July 20. The incident occurred after both teams were given a
warning following Giants ace Tim Lincecum hitting Matt Kemp earlier in the game. He threw his first career complete game shutout on September 14, 2010 also
against San Francisco and finished the season with a record of 13 – 10 and a 2.91 ERA in 32 starts, pitching 2041 / 3 innings and recording 212 strikeouts.
= = = = 2011 season: 1st Cy Young Award = = = =
After finishing the 2010 season strong, the Dodgers named Kershaw as the Opening Day Starter for the 2011 season. On May 29, he pitched the second completegame shutout of his career, striking out 10 while winning a two-hitter against the Florida Marlins, 8 – 0; he also had two singles and an RBI, scoring twice in the
game. He produced his third career shutout on June 20, a two-hit, 11-strikeout effort against the Detroit Tigers. Kershaw became the first Dodgers starter to strike
out the side in the 9th inning since Sandy Koufax’s perfect game. In his next start, on June 26, Kershaw pitched another complete game (against the Los Angeles
Angels of Anaheim ). He became the first Dodger starter to have back-to-back complete game victories since Jeff Weaver in the 2005 season and the first Dodger
to have double-digit strikeouts in consecutive starts since Chan-Ho Park in the 2000 season. He was awarded the National League Player of the Week award for the
week of June 20 – 26 as a result of those two starts. Midway through June, Kershaw had amassed 32 career victories


**XL Generation:** **Reference:**
, tying him with Luis Castillo for the third best start total in the league and trail-, a 3.15 ERA and 593 career strikeouts in 568.2 innings. According to the Elias
ing only Johnny Bench and Eddie Mathews (37). The Dodgers signed Kershaw Sports Bureau, Kershaw was the first 23-year-old pitcher to have that many
to a one-year deal on July 14, 2011, with a player option for 2012, in a deal victories, an ERA that low and an average of more than one strikeout per inning
worth just under $ 300,000. The Dodgers entered the 2011 season as the major since ERA became an official statistic in 1910. Kershaw was selected to the
league leaders, and Kershaw had the best record in the National League in saves National League team for the 2011 Major League Baseball All-Star Game, his
(39), games started (78) and walks (120); he also had the lowest win / loss ratio first All-Star selection. In the month of July, Kershaw was 4 – 1 with a 2.02
in the league (1.87), and he had the worst on-base percentage (.349). He had ERA and NL-leading 45 strikeouts, earning him the National League Pitcher of
the best ERA (2.00 ), most wins (16 ), or most shutouts (29) in his career (25), the Month Award. On August 23, he struck out Matt Holliday of the St. Louis
while leading the league in losses per nine innings pitched (4.01). Following the Cardinals for his 200th strikeout of the season and became the 10th Dodger
2011 season, the Dodgers traded Kershaw to the Kansas City Royals on July 29 pitcher to record back-to-back 200 strikeout seasons and the first since Chanfor pitcher Javier Torres. Ho Park did it in the 2001 season. Kershaw finished the 2011 season by leading
= = = Kansas City Royals = = = the NL with 21 wins, 248 strikeouts and a 2.28 ERA, winning the NL pitching
= = = = 2012 season = = = = Triple Crown, the first Triple Crown winner since Jake Peavy of the 2007 San
During spring training, Kershaw played very well. He was selected to spring Diego Padres and the first Dodger since Sandy Koufax won it in the 1966 season.
training as a relief pitcher for the Royals for the 2012 season. After an injury Justin Verlander of the Detroit Tigers won the American League Triple Crown
to closer Javier Vázquez, he was activated on April 29 to replace Matt Holliday the same season, marking the first major-league season since 1924 to feature
in the Royals ’ starting rotation. In his only start with the Royals, on August 6, Triple Crown-winning pitchers in both leagues. Kershaw’s 21 wins were the
2012, Kershaw struck out five batters in seven innings pitched to help the Royals most by a Dodger pitcher since Orel Hershiser won 23 during the 1988 season.
to their first victory in franchise history. On September 27, 2012, it appeared His ERA was the lowest by a Dodger since Hershiser’s 2.03 in the 1985 season,
Kershaw was going to pitch a complete game shutout against the Detroit Tigers, his strikeouts were the most by a Dodger since Koufax’s 317 in 1966 and his 233
but did not manage to do so since the Tigers won 3 – 1. At the conclusion 1 / 3 innings pitched were the most since Chan Ho Park pitched 234 in 2001.
of the season, Kershaw was named Major League Baseball’s Most Valuable Since 1965 when Koufax did it, Peavy and Kershaw are only two pitchers in the
Player, was chosen to the All-Star Game at Busch Stadium and was named to National League have led the league in wins, strikeouts, ERA, and WHIP (walks
the All-Star Game as the starting pitcher at shortstop. The Royals announced on plus hits per inning pitched). Kershaw also became just the second <unk> to
February 4, 2013 that Kershaw would spend 2013 as starting pitcher, though he have a 240-plus strikeouts in a season before the age of 24, joining Vida Blue.
was expected to miss the season as a result of a shoulder injury. After the season, Kershaw was awarded the Warren Spahn Award as the best
= = = = 2013 season = = = = left-handed pitcher in 2011, the Players Choice Award for Most Outstanding
On May 17, 2013, Kershaw sustained another back injury and did not start in National League pitcher, the Gold Glove Award as the top fielding pitcher in
August and October 2013. He appeared in 22 starts, all starts, finishing with the NL and the Sporting News (TSN) National League Pitcher of the Year. He
a strikeout-to-walk ratio of 1.50 and a 2.91 ERA. He also had the third most was additionally selected as the starting pitcher for the TSN NL All-Star Team.
strikeouts in the league: 10. On May 20, 2013, he On November 17, he was honored with the National League Cy Young Award,

making him the youngest Cy Young winner since Dwight Gooden


Table 11: Example 1 – 500 tokens generated by XL using a snippet from the Wikitext-103 test set as initial context. The
sample is randomly generated without any cherry picking.


[Original Wikipedia page: https://en.wikipedia.org/wiki/Clayton_Kershaw](https://en.wikipedia.org/wiki/Clayton_Kershaw)


There are many interesting observations from this example:


_•_ Firstly, Kershaw never went to Royals in real life. Despite that, Transformer-XL stays on the fully imagined topic and keeps
hallucinating the experience of Kershaw in Royals across the generated text.


_•_ Secondly, notice that XL correctly tracks the chronological order from 2011 to 2012 and to the finally 2013 season in the
section titles.


_•_ In addition, notice that Transformer-XL accurately uses the the phrase “another back injury” in the 2013 season paragraph,
since it has talked about one earlier injure in the 2012 season. This ~~shows again Transfo~~ rmer-XL’s ability of capturing
long-term dependency.


**Context:**

= = Distribution = =
Species range across the Neotropics from Mexico in the north to Bolivia, Paraguay, and southern Brazil in the south. According to <unk> and coauthors, three
species are found in Mexico, four in Central America, and 62 in South America. Three species are present in the Caribbean — two in Trinidad and Tobago, along
the southern edge of the region, and one in Haiti.
= = Habitat and ecology = =
<unk> includes both large trees and small acaulescent palms which occupy a number of different ecological niches. Dense stands of some of the larger species are
conspicuous elements on the landscape, while smaller species are found in both in the forest understorey and in savannas. Disturbance has been implicated in the
formation of vegetation dominated by large <unk> species. In seasonally dry Amazonian forests the density of large adult A. <unk> palms was correlated with
canopy openness; the species also dominates savannas formed by repeated forest fires in Trinidad and Tobago. <unk> speciosa forms pure stands in many parts of
Brazil where natural forest vegetation has been cleared. Similarly, stands of A. <unk> in Bahia, Brazil (which are cultivated for <unk> fibre) are managed using
fire — the seedlings survive cutting and burning, and are able to dominate burned forest patches. The fruit are dispersed by animals; fruit which are not dispersed
frequently suffer seed predation by <unk> beetles. Certain species of <unk> have been mentioned as examples of " anachronistic " species which are adapted
for dispersal by now-extinct Pleistocene megafauna. On <unk> Island, <unk>, in the Brazilian Amazon, <unk> <unk> fruit were consumed by tapirs, collared
peccaries, deer and primates. Rodents, including agoutis, fed upon the fruit and, as the fruit availability declined, they fed on the seeds. Other dispersers of <unk>
fruit include Crested <unk> which consume the fruit and disperse the seeds of A. <unk> in the Brazilian Pantanal.
= = Uses = =
<unk> species have a long history of human utilisation. <unk> <unk> <unk> seeds have been found in archaeological sites in Colombia dating back to 9000 BP.
A variety of species remain important sources of edible oil, thatch, edible seeds and fibre. The leaves of <unk> <unk> and A. <unk> are used extensively for
thatching. Several species are oil palms, with A. speciosa among the most important economically. Products extracted from A. speciosa were reported to support
over 300,000 households in the Brazilian state of Maranhão in 2005, and in 1985 it was estimated to support over 450,000 households throughout the Brazil. <unk>
fibres, extracted from the leaf bases of A. <unk>, are commercially important, and generated about US $


**XL Generation:** **Reference:**
700,000 in 2013 on the black market. In 2004 the plant was reported to be 20 million in annual income to Brazilian farmers in 1996.
responsible for a net profit of around US $ 3 million in Brazil. A 2012 study = The Heart of Ezra Greer =
found A. speciosa to cause agricultural production at least 2 – 3 times greater The Heart of Ezra Greer is a 1917 American silent drama film produced by the
than those achieved without prior knowledge of land management. Thanhouser Company and directed by Emile <unk>. The film focuses on Ezra
= The Tale of Mrs. Tittlemouse = Greer, a successful middle-aged man who searches for his college age daughter,
The Tale of Mrs. Tittlemouse is a children’s book written and illustrated by H. Mary. The wayward Mary was romanced and abandoned by Jack <unk>, later
M. Potter. It was first published in February 1910 by Frederick Warne & Co. bearing his child. Once Ezra becomes broke he finds employment as the valet
and included illustrations by Frederick Warne. Mrs. Tittlemouse was intended for Jack <unk>. After Jack’s engagement to a cabaret girl, Mary becomes upset
to be a children’s book, rather than a musical, and the tale describes her attempts and leaves her child at Jack’s home. Contrary to Jack’s wishes, Ezra keeps the
to find her own home. Mrs. Tittlemouse is an average kitten and the tale follows child and Jack ultimately reveals that the child is his own. Ezra convinces Jack
a mischievous rabbit named Mr. Tittlemouse who tries to kill her in an attempt to make things right and Ezra convinces the cabaret girl to leave Jack. After a
to get a ride on a horse. Potter later wrote of the rabbit and her attempts to carriage accident in which the baby is injured, Ezra and Jack rush to the hospital
kill her, " If [ she ] were a fox, I should like to have been able to show her the and find Mary as a nurse crying over the child. The film ends with the marriage
way. " Potter’s work was inspired by The Tale of Little Black Sambo and her of Jack and Mary. The film was released by Pathé on October 7, 1917. The film
two nieces, Petunia and Petunia. It was the first book in the Potter family to be was the final release from Thanhouser and was deemed to be an average film
published in a hardback format, and was also the first book in the Peter Rabbit by most reviewers. Criticism for the film hinged on far-fetched coincidences to
series. The tale was released on 13 November 1910 and was accompanied by drive the plot. The film is presumed lost.
a greeting card by Potter’s brother-in-law, Frederick Warne. The tale was well = = Plot = =
received by critics and children, and sold over 600,000 copies in its first few The film follows Ezra Greer, a middle-aged man who has worked hard since
years. In the first year of its publication, Potter’s The Tale of Mrs. Tittlemouse his youth. He cares deeply for his motherless daughter, Mary, but was unable
won children’s book awards in England, America and Australia. It was selected to attend the annual commencement at her co-educational college. He awaits
as a " recommended reading " by children in the US and Canada. for her to return from college, but Mary leaves with her romantic interest, Jack
= = Plot summary = = <unk>. On promise of marriage and wealth, Mary is romanced and gives birth
The tale opens with the pet rabbit Mrs. Tittlemouse wandering through a forest to a fatherless child. Without word from his daughter, Ezra resigns from his
in a small litter of four. He is shot and badly injured by a deer after she attempts job and attempts to seek her out and finds a poor motherless child, Marie. With
to escape. Mrs. Tittlemouse decides that she wants to find her own home, be- Ezra’s money exhausted he seeks employment and finds it as the valet of Jack.
cause she is eager to go on her own. She goes alone to the farm where she makes One day, Mary seeks an announcement of Jack’s engagement to a cabaret girl
a little money by selling a few seeds and building a small cabin in the woods. known as " The Baby Vamp ". Bitter over the prospect of her child’s future,
She is approached by a wealthy hunter named Mr. Tittlemouse, who tries to kill she leaves the child at Jack’s home during his absence with a note. Jack orders
her but Mrs. Tittlemouse kills him by stuffing a rope into his nose and killing Ezra to take the baby to an orphanage, but Marie begs Ezra to keep him. After
him. She is rescued by Mr. Tittlemouse’s wife Ruth, but Mrs. Tittlemouse then continually seeing the child, Jack is overcome with remorse and explains to Ezra
leaves the woodland with the baby. When she is spotted by and seeks his advice. Not knowing he was making the case for his own daughter,

Ezra convinces Jack to seek out Mary and forget the Baby Vamp. The Baby


Table 12: Example 2 – 500 tokens generated by XL using a snippet from the Wikitext-103 test set as initial context. The
sample is randomly generated without any cherry picking.


[Original Wikipedia page: https://en.wikipedia.org/wiki/The_Tale_of_Mrs._Tittlemouse.](https://en.wikipedia.org/wiki/The_Tale_of_Mrs._Tittlemouse)


This example exhibit some additional interesting properties of Transformer-XL:


_•_ After finishing the last paragraph of the seed context, both the reference and generated text start a new topic (i.e., Wikipedia
page), as marked by the single “= title =” line. This suggests the model has the ability of identifying the end of a topic / page,
and randomly starting with a new topic.


_•_ Even more interestingly, a newly-started page is on a book called “The Tale of Mrs. Tittlemouse”. Transformer-XL manages
to copy the same book title and some related information from the training set, but hallucinates _novel_ content of the book.
This demonstrates a degree of generalization instead of memorization. Please refer to the original book content at the
Wikipedia page.


**Context:**

= Battle of Dürenstein =

The Battle of Dürenstein (also known as the Battle of <unk>, Battle of <unk> and Battle of <unk>; German: <unk> bei <unk> ), on 11 November 1805 was
an engagement in the Napoleonic Wars during the War of the Third Coalition. Dürenstein (modern <unk>) is located in the <unk> Valley, on the River Danube,
73 kilometers (45 mi) upstream from Vienna, Austria. The river makes a crescent-shaped curve between <unk> and nearby Krems an der Donau and the battle
was fought in the flood plain between the river and the mountains. At Dürenstein a combined force of Russian and Austrian troops trapped a French division
commanded by Théodore Maxime Gazan. The French division was part of the newly created VIII Corps, the so-called Corps Mortier, under command of Édouard
Mortier. In pursuing the Austrian retreat from Bavaria, Mortier had over-extended his three divisions along the north bank of the Danube. Mikhail <unk> Kutuzov,
commander of the Coalition force, enticed Mortier to send Gazan’s division into a trap and French troops were caught in a valley between two Russian columns.
They were rescued by the timely arrival of a second division, under command of Pierre Dupont de l ’Étang. The battle extended well into the night. Both sides
claimed victory. The French lost more than a third of their participants, and Gazan’s division experienced over 40 percent losses. The Austrians and Russians
also had heavy <unk> to 16 <unk> perhaps the most significant was the death in action of Johann Heinrich von Schmitt, one of Austria’s most capable chiefs of
staff. The battle was fought three weeks after the Austrian capitulation at Ulm and three weeks before the Russo-Austrian defeat at the Battle of Austerlitz. After
Austerlitz Austria withdrew from the war. The French demanded a high indemnity and Francis II abdicated as Holy Roman Emperor, releasing the German states
from their allegiance to the Holy Roman Empire.
= = Background = =
In a series of conflicts from 1803-15 known as the Napoleonic Wars, various European powers formed five coalitions against the First French Empire. Like the wars
sparked by the French Revolution (1789 ), these further revolutionized the formation, organization and training of European armies and led to an unprecedented
militarization, mainly due to mass conscription. Under the leadership of Napoleon, French power rose quickly as the Grande Armée conquered most of Europe,
and collapsed rapidly


**XL Generation:** **Reference:**
after the French conquest of Italy. This victory marked the beginning of the after the disastrous invasion of Russia in 1812. Napoleon’s empire ultimately
Second Coalition. Napoleon’s rapid advance caused Austria – Russia, Britain suffered complete military defeat in the 1813 – 14 campaigns, resulting in the
and Russia to make peace negotiations. The Russian army under Prince Mikhail restoration of the Bourbon monarchy in France. Although Napoleon made a
Mikhailovich Mikhailovich Chaffee, commander of the Imperial Russian Army spectacular return in 1815, known as the Hundred Days, his defeat at the Battle
in Switzerland, was able to drive off the Austrians in the Battle of Stockach of Waterloo, the pursuit of his army and himself, his abdication and banishment
(1805) and to take Pressburg (modern Ö3 Austria) from the Austrians. At an to the Island of Saint Helena concluded the Napoleonic Wars.
early stage the Austro-Hungarian fleet had been damaged in a storm off Cape = = Danube campaign = =
Matapan; this allowed the Allies to launch a full-scale invasion of Italy in Febru- From 1803-06 the Third Coalition fought the First French Empire and its client
ary 1805. The Battle of Austerlitz was the decisive French victory against states (see table at right ). Although several naval battles determined control of
Napoleon and the largest naval battle in history, involving more modern-day the seas, the outcome of the war was decided on the continent, predominantly
European navies. The French military situation worsened as Napoleon faced in two major land operations in the Danube valley: the Ulm campaign in the
several threats related to his newly formed Army of Europe, particularly Russia upper Danube and the Vienna campaign, in the middle Danube valley. Political
itself, which was now on the verge of collapse. The Russian general and friend conflicts in Vienna delayed Austria’s entry into the Third Coalition until 1805.
of Napoleon, Alexander, had been dissatisfied with the conduct of the Austri- After hostilities of the War of the Second Coalition ended in 1801, Archduke
ans; he was still in conflict with Emperor Napoleon, the French Republic’s king, <unk> emperor’s <unk> advantage of the subsequent years of peace to develop
who had declared war on Austria during the summer of 1804. With the war over, a military restructuring plan. He carefully put this plan into effect beginning in
Napoleon decided to launch a second invasion of Italy in October 1805. 1803 – 04, but implementation was incomplete in 1805 when Karl Mack, Lieu= = Prelude = = tenant Field Marshal and Quartermaster-General of the Army, implemented his
In July 1805, the French 1st Army entered southern Italy. The army, under the own restructuring. Mack bypassed Charles ’ methodical approach. Occurring
command of Marshal Marmont, were reinforced by a few battalions of infantry in the field, Mack’s plan also undermined the overall command and organizaunder Claude General Auguste de Marmont at the town of Philippsburg and tional structure. Regardless, Mack sent an enthusiastic report to Vienna on the
another battalion at Belluno. On 17 September 1805, the army marched from military’s readiness. Furthermore, after misreading Napoleon’s maneuvers in
Belluno towards Krems. By 29 September, they had reached Belluno and con- Württemberg, Mack also reported to Vienna on the weakness of French dispoducted its advance against a small Austrian force. By 31 September, the whole sitions. His reports convinced the war party advising the emperor, Francis II,
force had been reinforced by a brigade from the Army of Tyrol under the com- to enter the conflict against France, despite Charles ’ own advice to the conmand of Pierre Augereau. The Austrians were now under the command of Mar- trary. Responding to the report and rampant anti-French fever in Vienna, Franshal Jean Victor Marie Moreau, a member of the Directory. Moreau had taken cis dismissed Charles from his post as generalissimo and appointed his <unk>
command of the Austrian invasion force in the spring of 1805. His command brother-in-law, Archduke Ferdinand, as commander. The inexperienced Ferdiincluded the VI Corps commanded by Jean Baptiste Drouet de Ney and the VI nand was a poor choice of replacement for the capable Charles, having neither
Corps commanded by Generals Jean Victor Marie Moreau and Joseph Souham. maturity nor aptitude for the assignment. Although Ferdinand retained nomiNey’s corps consisted of the III. Corps and VI. Corps, which consisted of the nal command, day-to-day decisions were placed in the hands of Mack, equally
III Corps and VI. Corps, located in the Austrian Netherlands, was commanded ill-suited for such an important assignment. When Mack was wounded early
by Friedrich Joseph, Count Baillet de Latour. Moreau’s army consisted of six in the campaign, he was unable to take full charge of the army. Consequently,
divisions and several associated brigades. command further devolved to Lieutenant Field Marshal Karl Philipp, Prince of
= = Aftermath = = Schwarzenberg, an able cavalry officer but inexperienced in the command of
= = = First Coalition forces = = = such a large army.
On 9 October 1805 the French Army of the Danube was attacked by an Aus- = = = Road to Ulm = = =
trian army under Archduke Charles at the Battle of Austerlitz. Although Charles The campaign in the upper Danube valley began in October, with several clashes
and Charles had not had much time to regroup, on 10 October, he launched his in Swabia. Near the Bavarian town of Wertingen, 40 kilometers (25 mi) northattack on the Polish forces under Friedrich Joseph, Count of Lauenburg. Af- west of Augsburg, on 8 October the 1st Regiment of dragoons, part of Murat’s
ter three days, Charles’ army captured Lauenburg. The French forces pursued Reserve Cavalry Corps, and grenadiers of Lannes ’ V Corps surprised an Austhe Austrians to the Silesian border, where they encountered strong Austrian trian force half its size. The Austrians were arrayed in a line and unable to form
resistance. These conflicts forced the Austrians to retreat into Tyrol and Aus- their defensive squares quickly enough to protect themselves from the 4,000
tria agreed to a truce. The Austrian army, commanded by Wenzel Anton Karl, dragoons and 8,000 grenadiers. Nearly 3,000 Austrians were captured and over
Count of Merveldt, was reduced to around 10,000 men. It was initially planned 400 were killed or wounded. A day later, at another small town, <unk> south
that Archduke Charles would launch a counter-attack against the French army of the Danube <unk> French 59th Regiment of the Line stormed a bridge over
on the same day, as Napoleon had hoped, but this was not carried out. On 25 the Danube and, humiliatingly, chased two large Austrian columns toward Ulm.
October, Merveldt left Styria for Tyrol. On the same day, Austria launched its The campaign was not entirely bad news for Vienna. At Haslach, Johann von
new offensive against the French at Ulm. Charles withdrew his army from the Klenau arranged his 25,000 infantry and cavalry in a prime defensive position
region for a third time at the Battle of Elchingen, under the overall command of and, on 11 October, the overly confident General of Division Pierre Dupont de
the Austrian generals, Ferdinand and Friedrich Wilhelm of Jülich-Cleves-Berg. l’Étang attacked Klenau’s force with fewer than 8,000 men. The French lost
To prevent Archduke Charles from escaping from the battlefield, the comman- 1,500 men killed and wounded. Aside from taking the Imperial Eagles and
der of the Habsburg army, Archduke Charles, planned to occupy the fortress <unk> of the 15th and 17th Dragoons, Klenau’s force also captured 900 men,
Linz; instead, he decided to force Franz von Hipper to surrender the city. How- 11 guns and 18 ammunition wagons. Klenau’s victory was a singular success.
ever, as Charles moved to the south, Moreau arrived on the scene with additional On 14 October Mack sent two columns out of Ulm in preparation for a breakout
soldiers – including the entire Imperial Guard – and defeated the Austrians at to the north: one under Johann Sigismund Riesch headed toward Elchingen to
the Battle of Hohenlinden on 28 October. The loss of Linz resulted in Austria’s secure the bridge there, and the other under Franz von Werneck went north with
complete defeat at Hohenlinden. In the meantime, the French Army of Obser- most of the heavy artillery. Recognizing the opportunity, Marshal Michel Ney
vation and Preparedness was reorganized into the Army of the Danube under hurried the rest of his VI Corps forward to re-establish contact with Dupont, who
Feldzeugmeister (Colonel-General) Friedrich Freiherr von Hotze. The army was still north of the Danube. In a two-pronged attack Ney sent one division to
was composed of the I, IV, VI, VI, VII, VIII and IX Corps. With reinforcements the south of Elchingen on the right bank of the Danube. This division began the
from Italy and France, it formed new battalions, companies, and squadrons in assault at Elchingen. At the same time another division crossed the river to the
the Austrian army. On 17 November 1804, at the Battle of Jena-Auerstadt the east and moved west against Riesch’s position. After clearing Austrian pickets
Army of Silesia and the Army of Silesia joined forces, but by the time that the from a bridge, the French attacked and captured a strategically located abbey at


French approached Vienna, the Prussians had already surrendered. As the Austrians did not want to allow the war to continue, they decided to abandon their
territories in the north and move their army to the north and west, cutting off
Charles from Vienna. The Battle of Warsaw was fought on 23 November 1805
between the French Army of the Danube and the Austrian Army of Styria in
the vicinity of Warsaw and Pressburg (modern Trnava, Slovakia). At that time
Habsburg forces



the top of the hill at bayonet point. The Austrian cavalry unsuccessfully tried to
fend off the French, but the Austrian infantry broke and ran. In this engagement
alone, the Austrians lost more than half their reserve artillery park, 6,000 (out
of 8,000 total participants) dead, wounded or captured and four colors. Reisch’s
column also failed to destroy the bridges across the Danube. Napoleon’s lightning campaign exposed the Austrian indecisive command structure and poor
supply apparatus. Mack



Table 13: Example 3 – 1,000 tokens generated by XL using a snippet from the Wikitext-103 test set as initial context. The
sample is randomly generated without any cherry picking.


[Original Wikipedia page: https://en.wikipedia.org/wiki/Battle_of_D%C3%BCrenstein.](https://en.wikipedia.org/wiki/Battle_of_D%C3%BCrenstein)


_•_ Although this example is significantly longer, we can see that Transformer-XL is still able to stay on the same topic and
makes up non-existing stories about the Napoleon wars.


_•_ Notably, from the second section on, the generated text correctly follows a fine-grained chronological order _on the level of_
_month and day_ to narrate events in 1805, except a mistake (1804 instead of 1805) near the end of the paragraph. To ease
reading which we have highlighted all the date related phrases by magenta in the generation.



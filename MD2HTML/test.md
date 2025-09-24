## Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context

Zihang Dai ∗ 12 , Zhilin Yang ∗ 12 , Yiming Yang 1 , Jaime Carbonell 1 , 2 1

Quoc V. Le , Ruslan Salakhutdinov

1 Carnegie Mellon University, 2 Google Brain

{dzihang,zhiliny,yiming,jgc,rsalakhu}@cs.cmu.edu, qvl@google.com

## Abstract

Transformers have a potential of learning longer-term dependency, but are limited by a fixed-length context in the setting of language modeling. We propose a novel neural architecture Transformer-XL that enables learning dependency beyond a fixed length without disrupting temporal coherence. It consists of a segment-level recurrence mechanism and a novel positional encoding scheme. Our method not only enables capturing longer-term dependency, but also resolves the context fragmentation problem. As a result, TransformerXL learns dependency that is 80% longer than RNNs and 450% longer than vanilla Transformers, achieves better performance on both short and long sequences, and is up to 1,800+ times faster than vanilla Transformers during evaluation. Notably, we improve the state-ofthe-art results of bpc/perplexity to 0.99 on enwiki8, 1.08 on text8, 18.3 on WikiText-103, 21.8 on One Billion Word, and 54.5 on Penn Treebank (without finetuning). When trained only on WikiText-103, Transformer-XL manages to generate reasonably coherent, novel text articles with thousands of tokens. Our code, pretrained models, and hyperparameters are available in both Tensorflow and PyTorch 1 .

## 1 Introduction

Language modeling is among the important problems that require modeling long-term dependency, with successful applications such as unsupervised pretraining (Dai and Le, 2015; Peters et al., 2018; Radford et al., 2018; Devlin et al., 2018). However, it has been a challenge to equip neural networks with the capability to model long-term dependency in sequential data. Recurrent neural networks (RNNs), in particular Long Short-

∗ Equal contribution. Order determined by swapping the one in Yang et al. (2017).

1

https://github.com/kimiyoung/ transformer-xl

Term Memory (LSTM) networks (Hochreiter and Schmidhuber, 1997), have been a standard solution to language modeling and obtained strong results on multiple benchmarks. Despite the wide adaption, RNNs are difficult to optimize due to gradient vanishing and explosion (Hochreiter et al., 2001), and the introduction of gating in LSTMs and the gradient clipping technique (Graves, 2013) might not be sufficient to fully address this issue. Empirically, previous work has found that LSTM language models use 200 context words on average (Khandelwal et al., 2018), indicating room for further improvement.

On the other hand, the direct connections between long-distance word pairs baked in attention mechanisms might ease optimization and enable the learning of long-term dependency (Bahdanau et al., 2014; Vaswani et al., 2017). Recently, Al-Rfou et al. (2018) designed a set of auxiliary losses to train deep Transformer networks for character-level language modeling, which outperform LSTMs by a large margin. Despite the success, the LM training in Al-Rfou et al. (2018) is performed on separated fixed-length segments of a few hundred characters, without any information flow across segments. As a consequence of the fixed context length, the model cannot capture any longer-term dependency beyond the predefined context length. In addition, the fixed-length segments are created by selecting a consecutive chunk of symbols without respecting the sentence or any other semantic boundary. Hence, the model lacks necessary contextual information needed to well predict the first few symbols, leading to inefficient optimization and inferior performance. We refer to this problem as context fragmentation .

To address the aforementioned limitations of fixed-length contexts, we propose a new architecture called Transformer-XL (meaning extra long). We introduce the notion of recurrence into our deep self-attention network. In particular, instead of computing the hidden states from scratch for each new segment, we reuse the hidden states obtained in previous segments. The reused hidden states serve as memory for the current segment, which builds up a recurrent connection between the segments. As a result, modeling very longterm dependency becomes possible because information can be propagated through the recurrent connections. Meanwhile, passing information from the previous segment can also resolve the problem of context fragmentation. More importantly, we show the necessity of using relative positional encodings rather than absolute ones, in order to enable state reuse without causing temporal confusion. Hence, as an additional technical contribution, we introduce a simple but more effective relative positional encoding formulation that generalizes to attention lengths longer than the one observed during training.

Transformer-XL obtained strong results on five datasets, varying from word-level to characterlevel language modeling. Transformer-XL is also able to generate relatively coherent long text articles with thousands of tokens (see Appendix E), trained on only 100M tokens.

Our main technical contributions include introducing the notion of recurrence in a purely selfattentive model and deriving a novel positional encoding scheme. These two techniques form a complete set of solutions, as any one of them alone does not address the issue of fixed-length contexts. Transformer-XL is the first self-attention model that achieves substantially better results than RNNs on both character-level and word-level language modeling.

## 2 Related Work

In the last few years, the field of language modeling has witnessed many significant advances, including but not limited to devising novel architectures to better encode the context (Bengio et al., 2003; Mikolov et al., 2010; Merity et al., 2016; Al-Rfou et al., 2018), improving regularization and optimization algorithms (Gal and Ghahramani, 2016) , speeding up the Softmax computation (Grave et al., 2016a) , and enriching the output distribution family (Yang et al., 2017).

To capture the long-range context in language modeling, a line of work directly feeds a representation of the wider context into the network as an additional input. Existing works range from ones where context representations are manually defined (Mikolov and Zweig, 2012; Ji et al., 2015; Wang and Cho, 2015) to others that rely on document-level topics learned from data (Dieng et al., 2016; Wang et al., 2017).

More broadly, in generic sequence modeling, how to capture long-term dependency has been a long-standing research problem. From this perspective, since the ubiquitous adaption of LSTM, many efforts have been spent on relieving the vanishing gradient problem, including better initialization (Le et al., 2015), additional loss signal (Trinh et al., 2018), augmented memory structure (Ke et al., 2018) and others that modify the internal architecture of RNNs to ease the optimization (Wu et al., 2016; Li et al., 2018). Different from them, our work is based on the Transformer architecture and shows that language modeling as a real-world task benefits from the ability to learn longer-term dependency.

## 3 Model

Given a corpus of tokens x = ( x 1 , . . . , x T ) , the task of language modeling is to estimate the joint probability P ( x ) , which is often auto-regressively factorized as P ( x ) = ∏ t P ( x t | x &lt;t ) . With the factorization, the problem reduces to estimating each conditional factor. In this work, we stick to the standard neural approach to modeling the conditional probability. Specifically, a trainable neural network is used to encode the context x &lt;t into a fixed size hidden state, which is multiplied with the word embeddings to obtain the logits. The logits are then fed into the Softmax function, yielding a categorical probability distribution over the next token.

## 3.1 Vanilla Transformer Language Models

In order to apply Transformer or self-attention to language modeling, the central problem is how to train a Transformer to effectively encode an arbitrarily long context into a fixed size representation. Given infinite memory and computation, a simple solution would be to process the entire context sequence using an unconditional Transformer decoder, similar to a feed-forward neural network. However, this is usually infeasible with the limited resource in practice.

One feasible but crude approximation is to split the entire corpus into shorter segments of man-

Figure 1: Illustration of the vanilla model with a segment length 4.

<!-- image -->

ageable sizes, and only train the model within each segment, ignoring all contextual information from previous segments. This is the idea adopted by Al-Rfou et al. (2018). We call it the vanilla model and visualize it in Fig. 1a. Under this training paradigm, information never flows across segments in either the forward or backward pass. There are two critical limitations of using a fixedlength context. First, the largest possible dependency length is upper bounded by the segment length, which is a few hundred on character-level language modeling (Al-Rfou et al., 2018). Therefore, although the self-attention mechanism is less affected by the vanishing gradient problem compared to RNNs, the vanilla model is not able to fully exploit this optimization advantage. Second, though it is possible to use padding to respect the sentence or other semantic boundaries, in practice it has been standard practice to simply chunk long text into fixed-length segments due to improved efficiency (Peters et al., 2018; Devlin et al., 2018; Al-Rfou et al., 2018). However, simply chunking a sequence into fixed-length segments will lead to the context fragmentation problem as discussed in Section 1.

During evaluation, at each step, the vanilla model also consumes a segment of the same length as in training, but only makes one prediction at the last position. Then, at the next step, the segment is shifted to the right by only one position, and the new segment has to be processed all from scratch. As shown in Fig. 1b, this procedure ensures that each prediction utilizes the longest possible context exposed during training, and also relieves context fragmentation issue encountered in training. However, this evaluation procedure is extremely expensive. We will show that our proposed architecture is able to substantially improve the evaluation speed.

## 3.2 Segment-Level Recurrence with State Reuse

To address the limitations of using a fixed-length context, we propose to introduce a recurrence mechanism to the Transformer architecture. During training, the hidden state sequence computed for the previous segment is fixed and cached to be reused as an extended context when the model processes the next new segment, as shown in Fig. 2a. Although the gradient still remains within a segment, this additional input allows the network to exploit information in the history, leading to an ability of modeling longer-term dependency and avoiding context fragmentation. Formally, let the two consecutive segments of length L be s τ = [ x τ, 1 , · · · , x τ,L ] and s τ +1 = [ x τ +1 , 1 , · · · , x τ +1 ,L ] respectively. Denoting the n -th layer hidden state sequence produced for the τ -th segment s τ by h n τ ∈ R L × d , where d is the hidden dimension. Then, the n -th layer hidden state for segment s τ +1 is produced (schematically) as follows,

<!-- formula-not-decoded -->

where the function SG ( · ) stands for stop-gradient, the notation [ h u ◦ h v ] indicates the concatenation of two hidden sequences along the length dimension, and W · denotes model parameters. Compared to the standard Transformer, the critical difference lies in that the key k n τ +1 and value v n τ +1 are conditioned on the extended context ˜ h n -1 τ +1 and hence h n -1 τ cached from the previous segment. We emphasize this particular design by the green paths in Fig. 2a.

With this recurrence mechanism applied to every two consecutive segments of a corpus, it essentially creates a segment-level recurrence in the hidden states. As a result, the effective context being utilized can go way beyond just two segments. However, notice that the recurrent dependency between h n τ +1 and h n -1 τ shifts one layer downwards

Figure 2: Illustration of the Transformer-XL model with a segment length 4.

<!-- image -->

per-segment, which differs from the same-layer recurrence in conventional RNN-LMs. Consequently, the largest possible dependency length grows linearly w.r.t. the number of layers as well as the segment length, i.e., O ( N × L ) , as visualized by the shaded area in Fig. 2b. This is analogous to truncated BPTT (Mikolov et al., 2010), a technique developed for training RNNLMs. However, different from truncated BPTT, our method caches a sequence of hidden states instead of the last one, and should be applied together with the relative positional encoding technique described in Section 3.3.

Besides achieving extra long context and resolving fragmentation, another benefit that comes with the recurrence scheme is significantly faster evaluation. Specifically, during evaluation, the representations from the previous segments can be reused instead of being computed from scratch as in the case of the vanilla model. In our experiments on enwiki8, Transformer-XL is up to 1,800+ times faster than the vanilla model during evaluation (see Section 4).

Finally, notice that the recurrence scheme does not need to be restricted to only the previous segment. In theory, we can cache as many previous segments as the GPU memory allows, and reuse all of them as the extra context when processing the current segment. Thus, we can cache a predefined lengthM old hidden states spanning (possibly) multiple segments, and refer to them as the memory m n τ ∈ R M × d , due to a clear connection to the memory augmented neural networks (Graves et al., 2014; Weston et al., 2014). In our experiments, we set M equal to the segment length during training, and increase it by multiple times during evaluation.

## 3.3 Relative Positional Encodings

While we found the idea presented in the previous subsection very appealing, there is a crucial technical challenge we haven't solved in or- der to reuse the hidden states. That is, how can we keep the positional information coherent when we reuse the states? Recall that, in the standard Transformer, the information of sequence order is provided by a set of positional encodings, denoted as U ∈ R L max × d , where the i -th row U i corresponds to the i -th absolute position within a segment and L max prescribes the maximum possible length to be modeled. Then, the actual input to the Transformer is the element-wise addition of the word embeddings and the positional encodings. If we simply adapt this positional encoding to our recurrence mechanism, the hidden state sequence would be computed schematically by

<!-- formula-not-decoded -->

where E s τ ∈ R L × d is the word embedding sequence of s τ , and f represents a transformation function. Notice that, both E s τ and E s τ +1 are associated with the same positional encoding U 1: L . As a result, the model has no information to distinguish the positional difference between x τ,j and x τ +1 ,j for any j = 1 , . . . , L , resulting in a sheer performance loss.

In order to avoid this failure mode, the fundamental idea is to only encode the relative positional information in the hidden states. Conceptually, the positional encoding gives the model a temporal clue or 'bias' about how information should be gathered, i.e., where to attend. For the same purpose, instead of incorporating bias statically into the initial embedding, one can inject the same information into the attention score of each layer. More importantly, it is more intuitive and generalizable to define the temporal bias in a relative manner. For instance, when a query vector q τ,i attends on the key vectors k τ, ≤ i , it does not need to know the absolute position of each key vector to identify the temporal order of the segment. Instead, it suffices to know the relative distance between each key vector k τ,j and itself q τ,i , i.e. i -j . Practically, one can create a set of relative posi- tional encodings R ∈ R L max × d , where the i -th row R i indicates a relative distance of i between two positions. By injecting the relative distance dynamically into the attention score, the query vector can easily distinguish the representations of x τ,j and x τ +1 ,j from their different distances, making the state reuse mechanism feasible. Meanwhile, we won't lose any temporal information, as the absolute position can be recovered recursively from relative distances.

Previously, the idea of relative positional encodings has been explored in the context of machine translation (Shaw et al., 2018) and music generation (Huang et al., 2018). Here, we offer a different derivation, arriving at a new form of relative positional encodings, which not only has a one-to-one correspondence to its absolute counterpart but also enjoys much better generalization empirically (see Section 4). Firstly, in the standard Transformer (Vaswani et al., 2017), the attention score between query q i and key vector k j within the same segment can be decomposed as

<!-- formula-not-decoded -->

Following the idea of only relying on relative positional information, we propose to reparameterize the four terms as follows

<!-- formula-not-decoded -->

- The first change we make is to replace all appearances of the absolute positional embedding U j for computing key vectors in term ( b ) and ( d ) with its relative counterpart R i -j . This essentially reflects the prior that only the relative distance matters for where to attend. Note that R is a sinusoid encoding matrix (Vaswani et al., 2017) without learnable parameters.
- Secondly, we introduce a trainable parameter u ∈ R d to replace the query U glyph[latticetop] i W glyph[latticetop] q in term ( c ) . In this case, since the query vector is the same for all query positions, it suggests that the attentive bias towards different words should remain the same regardless of the query position. With a similar reasoning, a trainable parameter v ∈ R d is added to substitute U glyph[latticetop] i W glyph[latticetop] q in term ( d ) .
- Finally, we deliberately separate the two weight matrices W k,E and W k,R for producing the content-based key vectors and location-based key vectors respectively.

Under the new parameterization, each term has an intuitive meaning: term ( a ) represents contentbased addressing, term ( b ) captures a contentdependent positional bias, term ( c ) governs a global content bias, and ( d ) encodes a global positional bias.

In comparison, the formulation in Shaw et al. (2018) only has terms ( a ) and ( b ) , dropping the two bias terms ( c ) and ( d ) . Moreover, Shaw et al. (2018) merge the multiplication W k R into a single trainable matrix ˆ R , which abandons the inductive bias built into the original sinusoid positional encoding (Vaswani et al., 2017). In contrast, our relative positional embedding R adapts the sinusoid formulation. As a benefit of the inductive bias, a model trained on a memory of some certain length can automatically generalize to a memory several times longer during evaluation.

Equipping the recurrence mechanism with our proposed relative positional embedding, we finally arrive at the Transformer-XL architecture. For completeness, we summarize the computational procedure for a N -layer Transformer-XL with a single attention head here. For n = 1 , . . . , N :

<!-- formula-not-decoded -->

with h 0 τ := E s τ defined as the word embedding sequence. In addition, it is worth mentioning that a naive way to compute A requires computing W n k,R R i -j for all pairs ( i, j ) , whose cost is quadratic w.r.t. the sequence length. However, noticing that the value of i -j only ranges from zero to the sequence length, we show a simple computation procedure in Appendix B, which reduces the cost to be linear w.r.t. the sequence length.

## 4 Experiments

## 4.1 Main Results

We apply Transformer-XL to a variety of datasets on both word-level and character-level language

Table 1: Comparison with state-of-the-art results on WikiText-103. glyph[diamondmath] indicates contemporary work.

| Model                                                       | #Param   |   PPL |
|-------------------------------------------------------------|----------|-------|
| Grave et al. (2016b) - LSTM                                 | -        |  48.7 |
| Bai et al. (2018) - TCN                                     | -        |  45.2 |
| Dauphin et al. (2016) - GCNN-8                              | -        |  44.9 |
| Grave et al. (2016b) - LSTM + Neural cache                  | -        |  40.8 |
| Dauphin et al. (2016) - GCNN-14                             | -        |  37.2 |
| Merity et al. (2018) - QRNN                                 | 151M     |  33   |
| Rae et al. (2018) - Hebbian + Cache                         | -        |  29.9 |
| Ours - Transformer-XL Standard                              | 151M     |  24   |
| Baevski and Auli (2018) - Adaptive Input glyph[diamondmath] | 247M     |  20.5 |
| Ours - Transformer-XL Large                                 | 257M     |  18.3 |

Table 2: Comparison with state-of-the-art results on enwik8.

| Model                                   | #Param   |   bpc |
|-----------------------------------------|----------|-------|
| Ha et al. (2016) - LN HyperNetworks     | 27M      |  1.34 |
| Chung et al. (2016) - LN HM-LSTM        | 35M      |  1.32 |
| Zilly et al. (2016) - RHN               | 46M      |  1.27 |
| Mujika et al. (2017) - FS-LSTM-4        | 47M      |  1.25 |
| Krause et al. (2016) - Large mLSTM      | 46M      |  1.24 |
| Knol (2017) - cmix v13                  | -        |  1.23 |
| Al-Rfou et al. (2018) - 12L Transformer | 44M      |  1.11 |
| Ours - 12L Transformer-XL               | 41M      |  1.06 |
| Al-Rfou et al. (2018) - 64L Transformer | 235M     |  1.06 |
| Ours - 18L Transformer-XL               | 88M      |  1.03 |
| Ours - 24L Transformer-XL               | 277M     |  0.99 |

modeling to have a comparison with state-of-theart systems, including WikiText-103 (Merity et al., 2016), enwik8 (LLC, 2009), text8 (LLC, 2009), One Billion Word (Chelba et al., 2013), and Penn Treebank (Mikolov and Zweig, 2012).

WikiText-103 is the largest available word-level language modeling benchmark with long-term dependency. It contains 103M training tokens from 28K articles, with an average length of 3.6K tokens per article, which allows testing the ability of long-term dependency modeling. We set the attention length to 384 during training and 1600 during evaluation. We adopted adaptive softmax and input representations (Baevski and Auli, 2018; Grave et al., 2016a). As shown in Table 1, Transformer-XL reduces the previous state-of-theart (SoTA) perplexity from 20.5 to 18.3, which demonstrates the superiority of the TransformerXL architecture.

The dataset enwik8 contains 100M bytes of unprocessed Wikipedia text. We compare our architecture with the previous results in Table 2. Under the model size constraint, the 12-layer Transformer-XL achieves a new SoTA result, outperforming the 12-layer vanilla Transformer from Al-Rfou et al. (2018) by 0.05, while both Trans-

Table 3: Comparison with state-of-the-art results on text8.

| Model                                   | #Param   |   bpc |
|-----------------------------------------|----------|-------|
| Cooijmans et al. (2016) - BN-LSTM       | -        |  1.36 |
| Chung et al. (2016) - LN HM-LSTM        | 35M      |  1.29 |
| Zilly et al. (2016) - RHN               | 45M      |  1.27 |
| Krause et al. (2016) - Large mLSTM      | 45M      |  1.27 |
| Al-Rfou et al. (2018) - 12L Transformer | 44M      |  1.18 |
| Al-Rfou et al. (2018) - 64L Transformer | 235M     |  1.13 |
| Ours - 24L Transformer-XL               | 277M     |  1.08 |

Table 4: Comparison with state-of-the-art results on One Billion Word. glyph[diamondmath] indicates contemporary work.

| Model                                                       | #Param   |   PPL |
|-------------------------------------------------------------|----------|-------|
| Shazeer et al. (2014) - Sparse Non-Negative                 | 33B      |  52.9 |
| Chelba et al. (2013) - RNN-1024 + 9 Gram                    | 20B      |  51.3 |
| Kuchaiev and Ginsburg (2017) - G-LSTM-2                     | -        |  36   |
| Dauphin et al. (2016) - GCNN-14 bottleneck                  | -        |  31.9 |
| Jozefowicz et al. (2016) - LSTM                             | 1.8B     |  30.6 |
| Jozefowicz et al. (2016) - LSTM + CNN Input                 | 1.04B    |  30   |
| Shazeer et al. (2017) - Low-Budget MoE                      | ∼ 5B     |  34.1 |
| Shazeer et al. (2017) - High-Budget MoE                     | ∼ 5B     |  28   |
| Shazeer et al. (2018) - Mesh Tensorflow                     | 4.9B     |  24   |
| Baevski and Auli (2018) - Adaptive Input glyph[diamondmath] | 0.46B    |  24.1 |
| Baevski and Auli (2018) - Adaptive Input glyph[diamondmath] | 1.0B     |  23.7 |
| Ours - Transformer-XL Base                                  | 0.46B    |  23.5 |
| Ours - Transformer-XL Large                                 | 0.8B     |  21.8 |

former variants have a large margin over conventional RNN-based models. Notably, our 12-layer architecture achieves the same result as the 64layer network from Al-Rfou et al. (2018), using only 17% of the parameter budget. In order to see whether better performances can be obtained by increasing the model size, we train 18-layer and 24-layer Transformer-XLs with increased model sizes. With the attention length 784 during training and 3,800 during evaluation, we obtained a new SoTA result and our method is the first to break through 1.0 on widely-studied characterlevel benchmarks. Different from Al-Rfou et al. (2018), Transformer-XL does not need any auxiliary losses, and thus all benefits are credited to a better architecture.

Similar to but different from enwik8, text8 contains 100M processed Wikipedia characters created by lowering case the text and removing any character other than the 26 letters a through z , and space. Due to the similarity, we simply adapt the best model and the same hyper-parameters on enwik8 to text8 without further tuning. The comparison with previous methods is summarized in Table 3. Again, Transformer-XL achieves the new SoTA result with a clear margin.

Table 5: Comparison with state-of-the-art results on Penn Treebank. † indicates using two-step finetuning.

| Model                                      | #Param   |   PPL |
|--------------------------------------------|----------|-------|
| Inan et al. (2016) - Tied Variational LSTM | 24M      | 73.2  |
| Zilly et al. (2016) - Variational RHN      | 23M      | 65.4  |
| Zoph and Le (2016) - NAS Cell              | 25M      | 64    |
| Merity et al. (2017) - AWD-LSTM            | 24M      | 58.8  |
| Pham et al. (2018) - Efficient NAS         | 24M      | 58.6  |
| Liu et al. (2018) - Differentiable NAS     | 23M      | 56.1  |
| Yang et al. (2017) - AWD-LSTM-MoS          | 22M      | 55.97 |
| Melis et al. (2018) - Dropout tuning       | 24M      | 55.3  |
| Ours - Transformer-XL                      | 24M      | 54.52 |
| Merity et al. (2017) - AWD-LSTM+Finetune † | 24M      | 57.3  |
| Yang et al. (2017) - MoS+Finetune †        | 22M      | 54.44 |

One Billion Word does not preserve any longterm dependency because sentences have been shuffled. Consequently, this dataset mainly tests the ability of modeling only short-term dependency. The comparison between Transformer-XL and the other methods is shown in Table 4. Although Transformer-XL is mainly designed to better capture longer-term dependency, it dramatically improves the single-model SoTA from 23.7 to 21.8. Specifically, Transformer-XL significantly outperforms a contemporary method using vanilla Transformers (Baevski and Auli, 2018), suggesting the advantage of Transformer-XL is generalizable to modeling short sequences.

We also report the results on word-level Penn Treebank in Table 5. Similar to AWD-LSTM (Merity et al., 2017), we apply variational dropout and weight average to Transformer-XL. With proper regularization, Transformer-XL achieves a new SoTA result among models without two-step finetuning. Penn Treebank has only 1M training tokens, which implies that Transformer-XL also generalizes well even on small datasets.

## 4.2 Ablation Study

We conduct two sets of ablation studies to examine the effects of two proposed techniques used in Transformer-XL: the recurrence mechanism and the new positional encoding scheme.

The first study is performed on WikiText-103, which requires modeling long-term dependency. The results are reported in Table 6. Among the compared encoding schemes, Shaw et al. (2018) is relative, while Vaswani et al. (2017) and Al-Rfou et al. (2018) are absolute. 'Full' and 'half' losses refer to applying a cross entropy loss to all or the recent half positions in the segment. We found that absolute encodings only work well with half losses because half losses exclude positions with very short attention lengths during training for better generalization. Table 6 shows that both the recurrence mechanism and our encoding scheme are necessary to achieve the best performance, as well as generalizing to longer attention sequences during evaluation time. Although the backpropagation length during training is only 128, with the two techniques the attention length can be increased to 640 at test time. In the standard setting with 151M parameters, the perplexity decreases as the attention length increases.

Since the recurrence mechanism costs additional memory, we also compare Transformer-XL with baselines under the same GPU memory constraints. As shown in Table 10 in Appendix A, despite using a shorter backpropagation length, Transformer-XL remains superior to the baselines.

The second study targets at isolating the effects of resolving the context fragmentation problem from the benefit of capturing longer context length. In order to achieve this goal, we deliberately choose a dataset that does not require longterm dependency, so that any improvement from establishing the recurrence can be attributed to solving the context fragmentation. Specifically, we perform this controlled experiment on the One Billion Word dataset, which can only benefit from removing the context fragmentation. We train a 20-layer Transformer-XL with ∼ 0.3B parameters for 400K steps. As shown in Table 7, using segment-level recurrence substantially improves performance even when long-term dependency is not needed, which is consistent with our previous discussion that the recurrence mechanism resolves the context fragmentation problem. Moreover, our relative positional encodings is also superior to Shaw et al. (2018) on short sequences.

## 4.3 Relative Effective Context Length

Khandelwal et al. (2018) proposed a method to evaluate the Effective Context Length (ECL) of a sequence model. ECL is the longest length to which increasing the context span would lead to a gain more than a threshold. However, ECL ignores the fact that it is harder to get improvement when a model already achieves a lower perplexity using only a shorter context, and thus it is not suitable for fair comparison among multiple models. We instead propose a new metric

| Remark                        | Recurrence   | Encoding                                                  | Loss           | PPL init                                  | PPL best                                        | Attn Len            |
|-------------------------------|--------------|-----------------------------------------------------------|----------------|-------------------------------------------|-------------------------------------------------|---------------------|
| Transformer-XL (128M) - - - - | 3 3 3        | Ours Shaw et al. (2018) Ours Ours Ours Shaw et al. (2018) | Full Full Half | 27.02 27.94 28.69 29.59 30.10 29.75 30.50 | 26.77 27.94 28.33 29.02 30.10 29.75 30.50 30.97 | 500 256 460 260 120 |
| Transformer (128M) †          | 7 7 7        | et al. et al. et al.                                      | Full           | 30.97 31.16                               | 23.09                                           | 640                 |
|                               | 7            |                                                           | Full           |                                           |                                                 |                     |
|                               | 7            |                                                           | Half           |                                           |                                                 |                     |
| -                             | 7            |                                                           | Full           |                                           |                                                 | 120                 |
| -                             |              | Shaw (2018)                                               | Half           |                                           |                                                 | 120                 |
| -                             |              | Vaswani (2017)                                            | Half           |                                           |                                                 | 120                 |
|                               |              | Al-Rfou (2018)                                            | Half           |                                           | 31.16                                           | 120                 |
| Transformer-XL (151M)         | 3            | Ours                                                      |                | 23.43                                     | 23.16 23.35                                     | 450 300             |

Table 6: Ablation study on WikiText-103. For the first two blocks, we use a slightly smaller model (128M parameters). † indicates that the corresponding row is reduced to the same setting as the Transformer network in (Al-Rfou et al., 2018), except that two auxiliary losses are not implemented in our experiments. 'PPL init' refers to using the same length as training. 'PPL best' indicates the perplexity obtained by using the optimal length. 'Attn Len' is the shortest possible attention length during evaluation to achieve the corresponding result (PPL best). Increasing the attention length during evaluation improves performance only when our positional encoding is used. The 'Transformer-XL (151M)' setting uses a standard parameter budget as previous work (Merity et al., 2018), where we observe a similar effect when increasing the attention length during evaluation.

Table 7: Ablation study on One Billion Word, a dataset without long-term dependency.

| Method                            |   PPL |
|-----------------------------------|-------|
| Ours                              |  25.2 |
| With Shaw et al. (2018) encodings |  25.7 |
| Without recurrence                |  27.1 |

Table 8: Relative effective context length (RECL) comparison. See text for the definition of RECL and r . The first three models and the last four models are compared as two model groups when we calculate RECL (RECL is computed on a model group rather than a single model). Each group has the same parameter budget.

| Model                             |   r = 0 . 1 r |   = 0 . 5 |   r = 1 . 0 |
|-----------------------------------|---------------|-----------|-------------|
| Transformer-XL 151M               |           900 |       800 |         700 |
| QRNN                              |           500 |       400 |         300 |
| LSTM                              |           400 |       300 |         200 |
| Transformer-XL 128M               |           700 |       600 |         500 |
| - use Shaw et al. (2018) encoding |           400 |       400 |         300 |
| - remove recurrence               |           300 |       300 |         300 |
| Transformer                       |           128 |       128 |         128 |

called Relative Effective Context Length (RECL). RECL is defined on a model group instead of a single model, and the gain of a long context is measure by the relative improvement over the best short context model. As such, the model group shares the same baseline to enable fair comparison. RECL also has a parameter r , which means constraining the comparison on topr hard examples. See Appedix C for more details about RECL. As shown in Table 8, Transformer-XL manages to model dependency of 900 words long on av-

Table 9: Slowdown in terms of running time during evaluation. Evaluation is based on per-token time on one GPU.

| Attn Len   | How much Al-Rfou et al. (2018) is slower   |
|------------|--------------------------------------------|
| 3,800      | 1,874x                                     |
| 2,800      | 1,409x                                     |
| 1,800      | 773x                                       |
| 800        | 363x                                       |

erage with r = 0 . 1 . The RECL of TransformerXL is 80% and 450% longer than recurrent networks and Transformer respectively. Both the recurrence mechanism and our positional encodings contribute to a longer RECL. This further substantiates our argument that Transformer-XL is able to model longer-term dependency.

## 4.4 Generated Text

Trained only on WikiText-103 which is mediumsized, Transformer-XL is already able to generate relatively coherent articles with thousands of tokens without manual cherry picking, despite minor flaws. Please refer to Appendix E for samples.

## 4.5 Evaluation Speed

Finally, we compare the evaluation speed of our model with the vanilla Transformer model (AlRfou et al., 2018). As shown in Table 9, due to the state reuse scheme, Transformer-XL achieves an up to 1,874 times speedup during evaluation.

## 5 Conclusions

Transformer-XL obtains strong perplexity results, models longer-term dependency than RNNs and Transformer, achieves substantial speedup during evaluation, and is able to generate coherent text articles. We envision interesting applications of Transformer-XL in the fields of text generation, unsupervised feature learning, image and speech modeling.
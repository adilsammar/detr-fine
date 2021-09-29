<!-- # All About Detr

DETR stands for DEtection TRansformer, is a end to end object detection and segmentation model by FAIR.

Q: We take the encoded image (dxH/32xW/32) and send it to Multi-Head Attention?

A: We Pass out image input image to backbone (example ResNet50) which gives us features this features is then passed on to transformer which gives us this hidden state that is (d, h/32, w/32)

Q: We do something here to generate NxMxH/32xW/32 maps?

A: We apply Einstein summation convention, to convert `bqnc,bnchw->bqnhw`
    
    torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)

Q: Then we concatenate these maps with Res5 Block

A: These features are comming from backbone feature layers.


References: 

* https://github.com/waspinator/pycococreator -->


# DETR in depth

In DETR, object detection problem is modeled as a direct set prediction problem. This approach does not require hand crafted algorithms like `non-maximum suppression` procedure or `anchor generation` that explicitly encode our prior knowledge about the task. It makes the detection pipeline a simple end to end unified architecture. The two components of the new framework, called `DEtection TRansformer or DETR`

* Set-based global loss that forces unique predictions via bipartite matching.
* Transformer encoder-decoder architecture.

Given a fixed small set of learned object queries, DETR reasons about relations of  objects and global image context to directly output final set of predictions in parallel.


**How DETR differs from other object detection methods?**

DETR formulates the object detection task as an image-to-set problem. Given an image, the model predicts an unordered set of all objects present, each represented by its class and tight bounding box surrounding each one. Transformer then acts as a reasoning agent between the image features and the prediction.

## What is a transformer?

The paper ‘Attention Is All You Need’ introduces a novel architecture called Transformer. As the title indicates, it uses the attention-mechanism. Like LSTM, Transformer is an architecture for transforming one sequence into another one with the help of two parts (Encoder and Decoder), but it differs from the previously described/existing sequence-to-sequence models because it does not imply any Recurrent Networks (GRU, LSTM, etc.)

![Transformer Architecture](./assets/attention_arch.png)

In the above architecture lest part is Encoder and right part us Decoder. Both Encoder and Decoder are composed of modules that can be stacked on top of each other multiple times, which is shown as `Nx` above. These modules consist mainly of Multi-Head Attention and Feed Forward layers. 


> Transformers rely on a simple yet powerful mechanism called attention, which enables AI models to selectively focus on certain parts of their input and thus reason more effectively.


Transformers have been widely applied on problems with sequential data, in particular in natural language processing (NLP) tasks such as [language modeling](https://ai.facebook.com/blog/roberta-an-optimized-method-for-pretraining-self-supervised-nlp-systems/) and [machine translation](https://ai.facebook.com/blog/facebook-leads-wmt-translation-competition/), and have also been extended to tasks as diverse as [speech recognition](https://engineering.fb.com/ai-research/wav2letter/), [symbolic mathematics](https://ai.facebook.com/blog/using-neural-networks-to-solve-advanced-mathematics-equations/), and [reinforcement learning](https://arxiv.org/abs/2002.09402). But, perhaps surprisingly, computer vision was not swept up by the Transformer revolution before DETR came into existance.


> DETR completely changes the architecture compared with previous object detection systems. It is the first object detection framework to successfully integrate Transformers as a central building block in the detection pipeline


## DETR Pipeline

![detr_Pipeline](./assets/detr_pipeline.png)



At a high level these are the tasks detr perform 

1. Calculate image features from a backbone.
2. Transform image features using Encoder Decoder Architecture.
3. Calculate Set loss function which performs bipartite matching between predicted and ground-truth objects to remove false or extra detection's.
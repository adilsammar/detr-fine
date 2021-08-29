# All About Detr

DETR stands for DEtection TRansformer, is a end to end object detection and segmentation model by FAIR.

Q: We take the encoded image (dxH/32xW/32) and send it to Multi-Head Attention?

A: We Pass out image input image to backbone (example ResNet50) which gives us features this features is then passed on to transformer which gives us this hidden state that is (d, h/32, w/32)

Q: We do something here to generate NxMxH/32xW/32 maps?

A: We apply Einstein summation convention, to convert `bqnc,bnchw->bqnhw`
    
    torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)

Q: Then we concatenate these maps with Res5 Block

A: These features are comming from backbone feature layers.

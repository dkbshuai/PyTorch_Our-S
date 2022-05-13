# Our-S model and DistillWrapper

![](https://raw.githubusercontent.com/dkbshuai/PyTorch-Our-S/main/model%20pipeline.png)

```
import torch
from model import CvT
from distill import DistillWrapper
from torchvision.models import wide_resnet50_2

teacher = wide_resnet50_2(pretrained = False)

model = CvT(
    num_classes,
    
    s1_emb_dim1 = 96, 
    s1_emb_dim2 = 128, 
    s1_emb_kernel = 7, 
    s1_padding = 3,
    s1_max_kernel = 2, 
    s1_stride = 2,
    s1_heads = 1, 
    s1_depth = 3, 
    s1_mlp_mult = 4, 

        
    s2_emb_dim1 = 192, 
    s2_emb_dim2 = 256, 
    s2_emb_kernel = 3, 
    s2_padding = 1,
    s2_max_kernel = 2, 
    s2_stride = 1,
    s2_heads = 3, 
    s2_depth = 6, 
    s2_mlp_mult = 4, 

        
    s3_emb_dim1 = 256, 
    s3_emb_dim2 = 384, 
    s3_emb_kernel = 3, 
    s3_padding = 1,
    s3_max_kernel = 2, 
    s3_stride = 1,
    s3_heads = 6, 
    s3_depth = 6, 
    s3_mlp_mult = 4,    
)

distiller = DistillWrapper(
    student = model,
    teacher = teacher,
    temperature = 5,         
    alpha = 0.5,               
)

image = torch.randn(1, 1, 224, 224)

student_logits = model(image)

loss = distiller(image, label, student_logits)    # student_logits is model output
loss.backward()
```

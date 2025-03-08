# [CVPR 2025] Align-KD for Mobile Vision-Language Model Distillation
[【CVPR 2025】Align-KD: Distilling Cross-Modal Alignment Knowledge for Mobile Vision-Language Model](https://arxiv.org/abs/2412.01282)  
Qianhan Feng, Wenshuo Li, Tong Lin, Xinghao Chen* 

## Abstract
Vision-Language Models (VLMs) bring powerful understanding and reasoning capabilities to multimodal tasks. Meanwhile, the great need for capable aritificial intelligence on mobile devices also arises, such as the AI assistant software. Some efforts try to migrate VLMs to edge devices to expand their application scope. Simplifying the model structure is a common method, but as the model shrinks, the trade-off between performance and size becomes more and more difficult. Knowledge distillation (KD) can help models improve comprehensive capabilities without increasing size or data volume. However, most of the existing large model distillation techniques only consider applications on single-modal LLMs, or only use teachers to create new data environments for students. None of these methods take into account the distillation of the most important cross-modal alignment knowledge in VLMs. We propose a method called Align-KD to guide the student model to learn the cross-modal matching that occurs at the shallow layer. The teacher also helps student learn the projection of vision token into text embedding space based on the focus of text. Under the guidance of Align-KD, the 1.7B MobileVLM V2 model can learn rich knowledge from the 7B teacher model with light design of training loss, and achieve an average score improvement of 2.0 across 6 benchmarks under two training subsets respectively.

![main](figures/main_00.png 'Overview of Align-KD.')

## Instructions
### How to use
We provide incremental code developement based on the original MobileVLM repo. Please follow the following steps to get the full code.  
1. Download .zip document from our repo and also download the original code of MobileVLM.
2. Place Align-KD.zip under the file of MobileVLM.  
3. ```bash
   unzip -o Align-KD.zip 
   ```
   This will replace the orginal codes with our increamental codes.

### Environment
Please refer to requirements.txt  
Notice that we provide a new version of code with fixed seed using seed() function. This would lead to more stable final results, but will slightly cut down the baseline.

### Data
We follow MobileVLM v2 to collect the training data as well as the benchmark. Please refer to their work. We here express our appreciation to their contribution.  
Here are some noticings on data issues:  
1. Some of the datasets are updated from time to time, it is possible that the dataset you obtain is different.
2. During distillation, it is possible that the "cuda out of memory" error occurs, this is relevant to the environment as well as the data setup. One solution is to remove some variables after use during training, another more efficient solution is to store the inference feature of teacher model on samples whoes length is to long. This requires large storage, so it depends on the resources you have.

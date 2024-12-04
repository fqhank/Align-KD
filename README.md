# Align-KD for Mobile Vision-Language Model Distillation
[【arxiv】Align-KD: Distilling Cross-Modal Alignment Knowledge for Mobile Vision-Language Model](https://arxiv.org/abs/2412.01282)  
Qianhan Feng, Wenshuo Li, Tong Lin, Xinghao Chen*

***Full training code coming soon, scheduled to be released before 31/12/2024.***  
***Working on incremental code developement based on the original MobileVLM repo.***  

## Abstract
Vision-Language Models (VLMs) bring powerful understanding and reasoning capabilities to multimodal tasks. Meanwhile, the great need for capable aritificial intelligence on mobile devices also arises, such as the AI assistant software. Some efforts try to migrate VLMs to edge devices to expand their application scope. Simplifying the model structure is a common method, but as the model shrinks, the trade-off between performance and size becomes more and more difficult. Knowledge distillation (KD) can help models improve comprehensive capabilities without increasing size or data volume. However, most of the existing large model distillation techniques only consider applications on single-modal LLMs, or only use teachers to create new data environments for students. None of these methods take into account the distillation of the most important cross-modal alignment knowledge in VLMs. We propose a method called Align-KD to guide the student model to learn the cross-modal matching that occurs at the shallow layer. The teacher also helps student learn the projection of vision token into text embedding space based on the focus of text. Under the guidance of Align-KD, the 1.7B MobileVLM V2 model can learn rich knowledge from the 7B teacher model with light design of training loss, and achieve an average score improvement of 2.0 across 6 benchmarks under two training subsets respectively.

![main](figures/main_00.png 'Overview of Align-KD.')

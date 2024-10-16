## Literature Review
Whenever I am given data from any domain, be it health science, plant science, finance, etc., I try my best to first gain a foundational understanding of the problem from a domain-specific perspective. In this case, I started by studying literature related to glomeruli to understand what happens in sclerotic and non-sclerotic glomeruli. My goal was to understand the biological significance of the problem and explore how machine learning has been applied in similar tasks. I reviewed papers that solved similar classification problems, which helped me understand the previous methodologies used and the results that could be expected. Having experience across different domains such as health science and plant science, I understand that expectations for results and approaches vary across fields.

One paper I found by Pesce et al. [1], discussed the implementation of machine learning to classify sclerotic and non-sclerotic glomeruli. This paper introduced two approaches: traditional feature extraction and deep learning. In the feature extraction approach, I learned about using morphological and textural features for classification. Morphological features such as measurements of Bowman's capsule and glomerular diameters, and textural features such as LBP and Haralick texture descriptors were discussed. I was previously unfamiliar with these specific features, so I did some research on how they are defined and calculated. Based on my reading, I understood that morphological features are crucial because sclerosis alters the structure of the glomeruli. Hence, quantitative metrics like changes in area, width, or shape could help identify sclerotic glomeruli. On the other hand, textural features may depend on the image collection process. The second approach mentioned in the paper was deep learning-based. The machine learning model based on hand-crafted features seemed more explainable, however I realized that I would need detailed information about the images, such as how they were collected and acceptable parameter ranges which would require help from medical experts. Therefore, I decided to focus on automatic feature extraction using deep learning models.

Since morphological features are critical, I read another paper by Pinter et al. [2], which used variational autoencoders (VAE) for automatic morphological feature extraction. It showed promise for extracting relevant features across several classes. These two papers laid the foundation for my approach, and I decided to use deep learning for automatic feature extraction rather than relying on manual feature extraction, which could introduce errors/bias without medical expertise.

## Materials and Methods

I read a few more papers in the medical field that implemented recent and advanced DL approaches. For example, a paper by Dimitri et al. [3] utilized the DeepLab V2 model to segment sclerotic and non-sclerotic glomeruli from scanned Whole Slide Images (WSIs). I was specifically interested in applying a self-supervised modeling approach, particularly because I used this appoach to solve one of my master's thesis projects, which was one of the toughest problems I had encountered. My previous experience played a role here, as I believed that using self-supervised learning could help extract features that we humans may not easily see. As we all know, the famous AI scientist Yann LeCun named self-supervised learning as the 'dark matter of intelligence' (LeCun and Misra [4]). A paper by Nielsen et al. [5] discussed the implementation of self-supervised learning even with small training samples, as low as 100. We had an imbalanced dataset with 1,054 sclerotic images and 4,704 non-sclerotic images. I decided to include self-supervised learning as one of the models for this problem because I expected it to perform well.

I also decided to implement the VAE approach by Pinter et al. [2] to see if morphological features extracted using this method could help differentiate the two classes. Apart from these two advanced models, one based on self-supervised learning and the other on VAE, I included a third simpler model VGG-16. VGG-16 is a well-known model that is widely used across different fields and has been very popular for image classification tasks.

###
a) Self-supervised Learning

b) VAE

c) VGG-16

## References

1. Pesce F, Albanese F, Mallardi D, Rossini M, Pasculli G, Suavo-Bulzis P, Granata A, Brunetti A, Cascarano GD, Bevilacqua V, Gesualdo L. Identification of glomerulosclerosis using IBM Watson and shallow neural networks. J Nephrol. 2022 May;35(4):1235-1242. doi: 10.1007/s40620-021-01200-0. Epub 2022 Jan 18. PMID: 35041197; PMCID: PMC8765108.

2. Pinter, D., Beckmann, C.F., Fazekas, F. et al. Morphological MRI phenotypes of multiple sclerosis differ in resting-state brain function. Sci Rep 9, 16221 (2019). https://doi.org/10.1038/s41598-019-52757-7

3. Dimitri, Giovanna & Andreini, Paolo & Bonechi, Simone & Bianchini, Monica & Mecocci, Alessandro & Scarselli, Franco & Zacchi, Alberto & Garosi, Guido & Marcuzzo, Thomas & Tripodi, Sergio. (2022). Deep Learning Approaches for the Segmentation of Glomeruli in Kidney Histopathological Images. Mathematics. 10. 1934. 10.3390/math10111934. 

4. LeCun Y, Misra, I. Self-supervised learning: The dark matter of intelligence. https://ai.meta.com/blog/self-supervised-learning-the-dark-matter-of-intelligence/ 

5. Nielsen, M.; Wenderoth, L.; Sentker, T.; Werner, R. Self-Supervision for Medical Image Classification: State-of-the-Art Performance with ~100 Labeled Training Samples per Class. Bioengineering 2023, 10, 895. https://doi.org/10.3390/bioengineering10080895
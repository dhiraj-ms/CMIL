## Literature Review
Whenever I am given data from any domain, be it health science, plant science, finance, etc., I try my best to first gain a foundational understanding of the problem from a domain-specific perspective. In this case, I started by studying literature related to glomeruli to understand what happens in sclerotic and non-sclerotic glomeruli. My goal was to understand the biological significance of the problem and explore how machine learning has been applied in similar tasks. I reviewed papers that solved similar classification problems, which helped me understand the previous methodologies used and the results that could be expected. Having experience across different domains such as health science and plant science, I understand that expectations for results and approaches vary across fields.

One paper I found by Pesce et al. [1], discussed the implementation of machine learning to classify sclerotic and non-sclerotic glomeruli. This paper introduced two approaches: traditional feature extraction and deep learning. In the feature extraction approach, I learned about using morphological and textural features for classification. Morphological features such as measurements of Bowman's capsule and glomerular diameters, and textural features such as LBP and Haralick texture descriptors were discussed. I was previously unfamiliar with these specific features, so I did some research on how they are defined and calculated. Based on my reading, I understood that morphological features are crucial because sclerosis alters the structure of the glomeruli. Hence, quantitative metrics like changes in area, width, or shape could help identify sclerotic glomeruli. On the other hand, textural features may depend on the image collection process. The second approach mentioned in the paper was deep learning-based. The machine learning model based on hand-crafted features seemed more explainable, however I realized that I would need detailed information about the images, such as how they were collected and acceptable parameter ranges which would require help from medical experts. Therefore, I decided to focus on automatic feature extraction using deep learning models.

Since morphological features are critical, I read another paper by Tsutsumi et al. [2], which used variational autoencoders (VAE) for automatic morphological feature extraction. It showed promise for extracting relevant features across several classes. These two papers laid the foundation for my approach, and I decided to use deep learning for automatic feature extraction rather than relying on manual feature extraction, which could introduce errors/bias without medical expertise.

## Materials and Methods

I read a few more papers in the medical field that implemented recent and advanced DL approaches. For example, a paper by Dimitri et al. [3] utilized the DeepLab V2 model to segment sclerotic and non-sclerotic glomeruli from scanned Whole Slide Images (WSIs). I was specifically interested in applying a self-supervised modeling approach, particularly because I used this appoach to solve one of my master's thesis projects, which was one of the toughest problems I had encountered. My previous experience played a role here, as I believed that using self-supervised learning could help extract features that we humans may not easily see. As we all know, the famous AI scientist Yann LeCun named self-supervised learning as the 'dark matter of intelligence' (LeCun and Misra [4]). A paper by Nielsen et al. [5] discussed the implementation of self-supervised learning even with small training samples, as low as 100. We had an imbalanced dataset with 1,054 sclerotic images and 4,704 non-sclerotic images. I decided to include self-supervised learning as one of the models for this problem because I expected it to perform well.

I also decided to implement the VAE approach by Tsutsumi et al. [2] to see if morphological features extracted using this method could help differentiate the two classes. Apart from these two advanced models, one based on self-supervised learning and the other on VAE, I included a third simpler model VGG-16. VGG-16 is a well-known model that is widely used across different fields and has been very popular for image classification tasks.

###
**a) Self-supervised Learning:** Training self-supervised learning requires two phases. The first phase involves training on unlabeled images, which acts like 'unsupervised learning' where the model learns the features of the images from the unlabeled dataset. After this phase, the model weights are saved. These saved weights act as a feature extractor. In the second phase, a small labeled dataset is passed through this saved model to extract features. Since we now know the features of the labeled classes, these labeled features are passed through another machine learning model to make predictions, which is the second phase of training.

I wrote a python program to take 70% of the images from each class and put them into an unlabeled folder named 'images' https://www.dropbox.com/scl/fo/mo12wsuzs8rkmyjl17z9m/AAXdQowRocn1kDHTqH0QYjI?rlkey=sx9bw6rfu5knhlx2rojrfwso9&st=u3psqgkz&dl=0, which contains both classes but acts as unlabeled data. The remaining 30% of the images were labeled and kept in two separate folders corresponding to their respective classes. I used the lightly Python framework https://docs.lightly.ai/self-supervised-learning/lightly.html, which is popular for self-supervised learning, and trained the SimCLR architecture on 4,029 images for 50 epochs in the first phase. The trained model was saved as simclr_model.ckpt, which is available at https://www.dropbox.com/scl/fi/izp38wwto6xdx3ykpizhu/simclr_model.ckpt?rlkey=g76ep923pvw84e8fh2pkxpz73&st=wbjks0xf&dl=0  This saved model extracts 512-dimensional features (embeddings). It was then used to extract features from the labeled images of 317 sclerotic and 1,042 non-sclerotic glomeruli. These images can be found here: https://www.dropbox.com/scl/fo/rxctbnx27l8t9q669nigu/AOG0_kUsJgqzAF2oizzHyz0?rlkey=455wjotgqxzzggj1zgkdhazaa&st=wiopignx&dl=0. Since the extracted features are in 512 dimensions, many dimensions may be unnecessary and could introduce noise. To handle this, I applied PCA to reduce the dimensionality to 150. The embeddings for both sclerotic and non-sclerotic glomeruli (512-dimensional features) have been saved as NumPy arrays at **"simclr_embeddings"**, which can be used in the future to try different methods for the second phase of training on this small sample if needed. After reducing the features to 150 dimensions, the embeddings for sclerotic and non-sclerotic samples were merged, and four supervised models: logistic regression, random forest, support vector machine, and a neural network were trained on these features using an 80% training and 20% testing split. Stratified sampling was used to ensure the class distribution was maintained. The reason a validation set was not used during the second phase of training is due to two key factors. First, the main purpose of this approach is to leverage as many images as possible from both classes during the first phase of training to ensure the model learns strong and meaningful features. In self-supervised learning, the success of the entire process heavily depends on how well the model extracts features during this first phase. Hence, in the second phase the focus was on optimizing the training based on these pre-learned features. Since the model had already learned a robust representation of the images in the first phase, using a validation set during the second phase was less crucial especially due to the smaller number of labeled images available. Therefore, the final model was directly evaluated on the test set to evaluate its performance.

The neural network model had the following architecture:

Input layer: (150,), Dense layer: 128 units with ReLU activation, Dropout layer: 30% dropout rate, Dense layer: 64 units with ReLU activation, Dropout layer: 30% dropout rate, Output layer: 2 units with softmax activation

All models were trained for 50 epochs. The entire training workflow for both phases can be found here:**"model_trainings/selfsupervised.ipynb"**. The final neural network model weights were saved here:https://www.dropbox.com/scl/fi/7p3ckeea2jzmdk2hpk0l7/simclr_neuralnetwork_model.h5?rlkey=zhs6202vphkj96m09cdv7su9m&st=ygf07hec&dl=0. 


**b) VAE:** This was the first model I tried for training, out of the three models implemented. I started by reviewing the paper by Tsutsumi et al. [2] to understand the model architecture used by the authors. I also explored the GitHub repository of the paper, specifically focusing on the functions used to train the model: https://github.com/masa10223/Morpho-VAE/blob/main/CODE/functions.py .  I reviewed the create_model function and the loss functions, but they needed adjustments to fit our dataset. I wrote and ran each line of the code to fully understand the workflow, making changes slowly based on the errors I encourntered in jupyter. Once I resolved all the issues, I converted the adjusted lines of code into functions. The final functions used in this process are available in the notebook  **"model_training_workflow/VAE.ipynb"**. This paper used convolutions on each channel of the image separately. I modified this by applying convolutions to the entire image at once, rather than processing each channel individually. I also broke down the loss function into three separate functions. Additionally, I included hyperparameter tuning to optimize the number of layers, filter counts, and activation functions. Hyperparameter tuning plays a critical role in obtaining the best model for optimal performance. 10 trials were initially planned for tuning. Before setting up the training, I wrote a script to process the public.csv file, which contained the ground truth labels for the images. I did quality check there were no NaN values in the CSV and verified that the number of instances matched the image counts in the two folders. The dataset was then split into training and testing sets (80/20) using stratified sampling. During hyperparameter tuning the training images were further split using 5-fold cross-validation to identify the best hyperparameters. A random seed was set to ensure reproducibility in both model training and dataset splitting. Once optimization began, I realized that first trial took approximately 6 hours, means 10 trials would take around 60 hours. After running the first trial, I decided to interrupt the process and used the best hyperparameters obtained from that first trial. I used Optuna a popular hyperparameter optimization framework https://optuna.org/ . 

MLOps is an important part of the machine learning lifecycle especially when it comes to production. I wanted to showcase my skills in MLOps as well. So once I had the optimal hyperparameters, I wrote another function for training the optimal model using MLFlow, an open-source framework https://mlflow.org/ to present how this tool can manage different stages of the ML lifecycle including deployment. In my case, I implemented only the basic functionalities of MLFlow, but more advanced versions could be used in the future for managing the ML lifecycle at different stages or for deployment purposes. After obtaining the optimal hyperparameters, I manually passed those values into this new function to further train the model. This second round of training used the entire training dataset without splitting it into train and validation sets, as we already had the best hyperparameters from a similar distribution. Increasing the dataset size can help improve the model's performance especially due to the imbalanced data. Once the second training phase was complete using the best hyperparameters, weights of the final model for deployment was saved here: https://www.dropbox.com/scl/fi/j16o4kymhajpazflri11w/best_deployed_vae_model.h5?rlkey=ngx8xoafivky2tr83lsc26nwp&st=ohu96459&dl=0 . This saved model was then evaluated on the test set. Throughout both training stages, 20 epochs were used for the first phase during optimization, and 50 epochs were used for the second phase to train the optimal model.

Model explainability is important for understanding how the model predicts classes. I used a popular technique called GradCAM to generate heatmaps on the images, which highlight the areas that play a significant role in distinguishing between classes. I wrote the GradCAM code, loaded the final deployed model weights, and used the weights just before the flattening layer to plot the gradients. I used 10 images from each class, where the predicted label matched the ground truth to get an overall idea of the activation patterns for each class. This approach provides a biological perspective that medical experts who may not be familiar with deep learning can understand. Model explainability is vital for all kinds of stakeholders, as it helps increase the reliability of the model during deployment especially in the medical field where patient lives are involved. The workflow for GradCAM is available in **"GradCAM.ipynb"**

**c) VGG-16:** VGG-16 is a popular supervised learning model for classification tasks. In this case, I used pre-trained model weights on the ImageNet dataset. The last layer of the model was removed, and the weights of all the previous layers were frozen, means they were set as untrainable. This allowed the pre-trained model to act as a feature extractor since it had already been trained on a large number of images.

After modifying the pre-trained model, the custom architecture was added for training on the custom dataset. To train the VGG-16 model on the custom dataset, I wrote Python code to split the dataset into training, validation and test set and the split was 80% for training, 10% for validation, and 10% for testing. Each of these sets contains two folders for the respective classes. The modified dataset for training, validation, and testing is available here: https://www.dropbox.com/scl/fo/p703rw9d0qf39nnc6r0ph/APrpq4QP9LZYn3k7i3LIOeo?rlkey=pmiz9tu47m6egslsqkhhstn7a&st=61ibqtqb&dl=0.  The model was trained for 50 epochs, and after each epoch the model was evaluated on the validation set. The final classification report was generated on the test set for performance evaluation. TensorFlow Keras was used for building and training the model. The entire workflow of training is available in the notebook: **"model_trainings/VGG.ipynb"** The weights of the final trained classification model are available here: https://www.dropbox.com/scl/fi/vgeyfztem46m72tsx7nyw/vgg16_model.h5?rlkey=ht9wbl7wceorf6bso7rmn7dk0&st=26p48651&dl=0

## References

1. Pesce F, Albanese F, Mallardi D, Rossini M, Pasculli G, Suavo-Bulzis P, Granata A, Brunetti A, Cascarano GD, Bevilacqua V, Gesualdo L. Identification of glomerulosclerosis using IBM Watson and shallow neural networks. J Nephrol. 2022 May;35(4):1235-1242. doi: 10.1007/s40620-021-01200-0. Epub 2022 Jan 18. PMID: 35041197; PMCID: PMC8765108.

2. Tsutsumi, M., Saito, N., Koyabu, D. et al. A deep learning approach for morphological feature extraction based on variational auto-encoder: an application to mandible shape. npj Syst Biol Appl 9, 30 (2023). https://doi.org/10.1038/s41540-023-00293-6

3. Dimitri, Giovanna & Andreini, Paolo & Bonechi, Simone & Bianchini, Monica & Mecocci, Alessandro & Scarselli, Franco & Zacchi, Alberto & Garosi, Guido & Marcuzzo, Thomas & Tripodi, Sergio. (2022). Deep Learning Approaches for the Segmentation of Glomeruli in Kidney Histopathological Images. Mathematics. 10. 1934. 10.3390/math10111934. 

4. LeCun Y, Misra, I. Self-supervised learning: The dark matter of intelligence. https://ai.meta.com/blog/self-supervised-learning-the-dark-matter-of-intelligence/ 

5. Nielsen, M.; Wenderoth, L.; Sentker, T.; Werner, R. Self-Supervision for Medical Image Classification: State-of-the-Art Performance with ~100 Labeled Training Samples per Class. Bioengineering 2023, 10, 895. https://doi.org/10.3390/bioengineering10080895
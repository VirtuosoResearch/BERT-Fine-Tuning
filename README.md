## A primer of transformer models and fine-tuning on language tasks  

###Abstract  
Transformer-based BERT Language Models have become very popular recently 
and have shown outstanding results on some popular Natural Language 
Problems. However, there is very little understanding of what makes 
them so successful and what kind of information is learned, 
and what are the limitations of such models. In this research work,
we study the properties of BERT and its applications in different 
fine-tuning tasks. We also explore the modification of BERT for 
the distributionally Robust Neural Networks for group shifts and
work on some famous datasets like Movie Re- views, Civil Comments, 
and the MultiNLI dataset. We show how such robust algorithms for training 
can improve the performance of the BERT on worst group accuracies.

###Prerequisites
python 3.6.8  
matplotlib 3.0.3  
numpy 1.16.2  
pandas 0.24.2  
pillow 5.4.1  
pytorch 1.1.0  
pytorch_transformers 1.2.0  
torchvision 0.5.0  
tqdm 4.32.2  
NVIDIA apex

###Datasets

####Movie Review dataset (https://ai.stanford.edu/~amaas/data/sentiment/) :  
We use movie review dataset to do a sentiment analysis and predict the nature of the review
given the comment. A review could be positive or negative.  
For our code the data will be in the following format :  
a. scripts/LanguageFineTuning/data/imdb_train.txt  
b. scripts/LanguageFineTuning/data/imdb_test.txt  
c. scripts/LanguageFineTuning/data/weights/added_tokens.json  
d. scripts/LanguageFineTuning/data/weights/config.json  
e. scripts/LanguageFineTuning/data/weights/pytorch_model.bin  
f. scripts/LanguageFineTuning/data/weights/special_tokens_map.json  
g. scripts/LanguageFineTuning/data/weights/tokenizer_config.json  
h. scripts/LanguageFineTuning/data/weights/vocab.txt  

The python scripts are following :  
a. scripts/LanguageFineTuning/BERT_sentiment_main.py  
b. scripts/LanguageFineTuning/BERT_sentiment_model.py  
c. scripts/LanguageFineTuning/BERT_sentiment_dataset.py  

To run the code use the following command :  
**Training** : $python BERT_sentiment_main.py --train --ratio x  
To train a BERT model with x% positive and (100-x)% negative samples in the 
training data.  
**Testing** : $python BERT_sentiment_main.py --evaluate --ratio x  
To test a BERT model with x% positive and (100-x)% negative samples in the 
testing data.
**Prediction** : $python BERT_sentiment_main.py --predict  
To predict a single random entry or a movie comment in the dataset.  


####Civil Comments dataset (https://www.tensorflow.org/datasets/catalog/civil_comments) 
We use Civil Comments dataset to identify the Subgroup robustness of the BERT model in 
different toxic comments to identify which comments are offensive to the various subgroups.
Detailed description is given in the project document.
This exercise was also a part of the Kaggle competition held here : 
(https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification)
For our code the data will be in the following format :  
a. Download BERT pretrained models from here : https://github.com/google-research/bert  
Put the data files in the scripts/CivilComments/input/bert-pretrained-models  
b. Download pytorch-pretrained-BERT from here :
https://github.com/huggingface/transformers/tree/3d78e226e68a5c5d0ef612132b601024c3534e38/pytorch_pretrained_bert  
Put the data at scripts/CivilComments/input/pytorch-pretrained-BERT  
c. Download the toxic comments data from kaggle competition 
https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data  
Put this data at scripts/CivilComments/input/jigsaw-unintended-bias-in-toxicity-classification  
d. After running the code, the output files will be bert_pytorch.bin and bert_config.json

To run the code use the following command to run the main file for training and testing, 
The code will report the AUC metrics score for various kinds defined in the kaggle competition:  
$python Main.py  
different hyperparameters can be used in the Main.py  like epochs, output model file, and
number of data entries to be used.  
The output metrics consist of :  
a. Subgroup_AUC  
b. BPSN_AUC  
c. BNSP_AUC  
d. Final overall AUC  
This will save the model in the results folder.

####MultiNLI dataset
Here we have pairs of sentences and given the first sentence, we have
to predict that second sentence is in support of first sentence, neutral with the first sentence or
contradiction with the first sentence. 
We use MultiNLI dataset with the Group_DRO algorithm which is a distributionally robust algorithm 
used for training models that can predict with the high worst subgroup accuracy.  
The data and algorithm can be found at scripts/Group_DRO. To run the code, use the following script:  
$python run_expt.py -s confounder -d MultiNLI -t gold_label_random 
-c sentence2_has_negation --lr 2e-05 --batch_size 32 --weight_decay 0 --model bert 
--n_epochs 3 --reweight_groups --robust --generalization_adjustment 0

The final results will be the accuracy values on the train, validation and test set.

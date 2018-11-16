# Prediction  

Usage:  
```
python Prediction.py [--model MODEL_PATh] [--txtdata INPUT_DATA_PATH]
                     [--word2vec WORD2VEC_MODEL] [--output OUTPUT_PATH]
```
Please download model from : https://drive.google.com/open?id=132242ZyfvShFFOvU1021UvT4_nCuJKcl  
word2vec model is at ```./word2vec/wv_new.wv```   
MODEL_PATH is the path of the model.  
WORD2VEC is the path of the word2vec model.  
INPUT_DATA_PATH is the path of input txt file.    
OUTPUT_PATH is the path of output file path.    
INPUT_DATA_PATH and OUTPUT_PATH are assigned by user.
  
# Training and Visualization

This part is carried out by the file ```train_with_schedule.py```. Usage:

```
usage: Chatbot_train.py [-h] [--model_name MODEL_NAME] [--figure_name FIGURE_NAME]
                        [--epoch_number EPOCH_NUMBER] [--batch_size BATCH_SIZE]
                        [--optimizer OPTIMIZER] [--word2vec WORD2VEC]
                        [--force] (--load_model | --new_model | --remove_model)
```

One of the three arguments ```--load_model/-l --new_model/-n --remove_model/-r``` is required, all of which requires a model name to operate properly. The model name defaults to ```chatbot_model_default```, which corresponds to model file name ```chatbot_model_default.pkl``` and optimizer state file name ```chatbot_model_default_optim.pkl```. The model name can be specified by the flag ```--model_name```. 

Users can train arbitrary number of epochs by using the flag ```--epoch/-e```.

The model also relies on a Gensim Word2Vec model with its path specified by the flag ```--word2vec```, which defaults to ```./Jeffery/word2vec_wv_Jeff.wv```.

Users can force the program to train a new model by feeding it the flag ```--force```. Beware of existing models.

Currently this program loads first 2000 entries of ```question_array1.npy``` and ```answer_array.npy```, written at line 99 and 100. May need to change this to achieve more desirable results.

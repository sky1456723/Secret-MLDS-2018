# Prediction  

Usage:  
```
python Prediction.py [-h] [-d] [--input INPUT] 
                     model txtdata output word2vec
```
Please download model from : https://drive.google.com/open?id=132242ZyfvShFFOvU1021UvT4_nCuJKcl  
word2vec model is at ```./word2vec/wv_new.wv```   
```model``` is the path of the model.  
```txtdata``` is the path of input txt file.    
```output``` is the path of output file path.  
```word2vec``` is the path of the word2vec model.  
It's no need to use ```--input```.  
  
# Training

This part is carried out by the file ```train_with_schedule.py```. Usage:

```
usage: train_with_schedule.py [-h] [--model_name MODEL_NAME] [--figure_name FIGURE_NAME]
                              [--epoch_number EPOCH_NUMBER] [--batch_size BATCH_SIZE]
                              [--optimizer OPTIMIZER] [--word2vec WORD2VEC]
                              [--force] (--load_model | --new_model | --remove_model)
                              data_number
```

One of the three arguments ```--load_model/-l --new_model/-n --remove_model/-r``` is required, all of which requires a model name to operate properly. The model name defaults to ```chatbot_model_default```, which corresponds to model file name ```chatbot_model_default.pkl``` and optimizer state file name ```chatbot_model_default_optim.pkl```. The model name can be specified by the flag ```--model_name```. 

Users can train arbitrary number of epochs by using the flag ```--epoch/-e```.

The model also relies on a Gensim Word2Vec model with its path specified by the flag ```--word2vec```, which defaults to ```./Jeffery/word2vec_wv_Jeff.wv```.

Users can force the program to train a new model by feeding it the flag ```--force```. Beware of existing models.

```data_number``` means how many data do you want to use for training.  
The training data must be put at ```hw2-2```, which is a numpy array.  

# Training and Visualization

This part is carried out by the file ```Chatbot_train.py```. Usage:

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

# To use Dataloader
Some requirement of input data type is in Data.py
Please do something like:
```
import Data
import torch.utils.data
dataset = Data.ChatbotDataset(data, label)
dataloader = torch.utils.data.DataLoader(dataset = dataset, collate_fn = collate_fn, ...)
```

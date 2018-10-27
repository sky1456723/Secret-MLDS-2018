# Training and Visualization

This part is carried out by the file ```Chatbot_train.py```. Usage:

```
Chatbot_train.py [-h] [--model_name MODEL_NAME]
                 [--optim_name OPTIM_NAME] [--figure_name FIGURE_NAME]
                 [--epoch_number EPOCH_NUMBER]
                 [--batch_size BATCH_SIZE] [--optimizer OPTIMIZER]
                 (--load_model | --new_model | --remove_model)
```

One of the three arguments ```--load_model/-l --new_model/-n --remove_model/-r``` is required. The model name defaults to ```chatbot_model_default.pkl``` and optimizer state name ```chatbot_optim_default.pkl``` unless otherwise specified by flags ```--model_name``` and ```--optim_name``` respectively. By default, this program trains the model for 5 epochs with batch size of 16 and Adam optimizer.

This program expects some behavior of model and dala loader:

## Model

```Chatbot_train.py``` only calls ```.forward()``` and feeds the model with 1) input sentence batch, 2) target batch (for scheduled sampling), and 3) current epoch number (for scheduled sampling).

The program calls ```.train()``` and ```.eval()```. You can modify those functions to achieve your goals.

## Dataloader

This program expects two data loaders, namely ```train_loader``` and ```test_loader```, at a call of the function ```DataLoader``` with one argument being the batch size. Both data loaders must be iterables and support ```__len__```, with ```__getitem___``` returning an input sentence batch, target batch, and actual sentence lengths. In other words, data loaders must support these syntaxes:

```
for i, (sentence, target, act) in enumerate(train_loader):
    ...
loss_avg = loss_sum / len(train_loader)
```

Also, the program expects an int_to_word function.

import os
import torch
import torch.nn as nn
# TODO: import customary dataloader
# import ??? as DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
start_time = time.time()

#-------------------------------------------------------------#
# DELETE THIS SECTION WHEN OTHER MODULES ARE DONE.
import torchvision
import torchvision.transforms as transforms


train_dataset = torchvision.datasets.CIFAR10(root='./',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

test_dataset = torchvision.datasets.CIFAR10(root='./',
    train=False,
    transform=transforms.ToTensor(),
)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
    batch_size=20,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
    batch_size=20,
    shuffle=False
)

class Model(nn.Module):
    def __init__(self, num_classes=10):
        super(Model, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*16*16, 1024),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
#-----------------------------------------------------------#


### DEVICE CONFIGURATION ###

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

### HYPERPARAMETERS ###

batch_size = 20


### DATA PREPROCESSING ###

# TODO: wait for DataLoader
# train_loader, test_loader = DataLoader(batch_size=batch_size)



### CRITERION ###

'''
def criterion(prediction, target, actual_length):
    # prediction is of shape (batch_size, max_sentence_length, vocabulary_size)
    # target is of shape     (batch_size, max_sentence_length)
    loss_sum = 0.0
    c = nn.CrossEntropyLoss()

    for i in prediction.shape[0]:
        # trim both prediction and target to actual length
        p = prediction[i, :actual_length[i], :]
        t = target[i, :actual_length[i]]
        loss_sum += c(p, t)

    return loss_sum
'''

# TODO: change back definition of criterion when DataLoader is done.
criterion = nn.CrossEntropyLoss()


### MAIN ###

def main(opts):

    ### MODEL ###
    if opts.load_model:
        model = torch.load(opts.model_name).to(device)

    if opts.new_model:
        # create a model
        if os.path.isfile(opts.model_name):
            # model name conflict
            confirm = input('Model \'{}\' already exists. Overwrite it? [y/N] ' \
                            .format(opts.model_name.strip('.pkl')))

            if confirm not in ['y', 'Y', 'yes', 'Yes', 'yeah']:
                print('Process aborted.')
                exit()

        # TODO: decide model name and parameters
        model = Model(10).to(device)

    if opts.remove_model:
        if not os.path.isfile(opts.model_name):
            print('Model \'%s\' does not exist!' % opts.model_name.strip('.pkl'))
            exit()
        else:
            try:
                os.remove(opts.model_name)
                os.remove(opts.optim_name)
            finally:
                print('Model \'%s\' removed successfully.' % opts.model_name.strip('.pkl'))
                exit()

    ### OPTIMIZATION ###
    optim_dict = {'Adam': torch.optim.Adam(model.parameters(), lr=0.001),
                'SGD': torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)}

    try:
        optim = optim_dict[opts.optimizer]
    except:
        print('Optimizer \'%s\' is not supported yet!' % opts.optimizer)
        print('Supported optimizers are:', *(optim_dict.keys()), sep=' ')
        exit()

    if opts.load_model:
        if not os.path.isfile(opts.optim_name):
            print('Warning: missing optimizer state!')
        else:
            optim_state = torch.load(opts.optim_name)
            optim.load_state_dict(optim_state)

    x = range(opts.epoch_number)
    ytr = [] # loss of training over epochs
    ytt = [] # loss of testing over epochs

    for i in range(opts.epoch_number):

        ### TRAIN ###
        model.train()
        loss_sum = 0.0
        #for j, (sentence, target, act) in enumerate(train_loader):
        for j, (img, lbl) in enumerate(train_loader):
            # <sentence> is of shape (...)
            # <target> is of shape (...)
            # <act> records actual length of each sentence
            #sentence, target, act = sentence.to(device), target.to(device), act.to(device)
            img, lbl = img.to(device), lbl.to(device)

            optim.zero_grad()
            #prediction = model(sentence, target, current_epoch)
            #loss = criterion(prediction, sentence, act)
            out = model(img)
            loss = criterion(out, lbl)

            loss.backward()
            optim.step()

            # loss of batch J
            loss_sum += loss.item()

        ltr = loss_sum / len(train_loader)
        ytr.append(ltr)

        ### TEST ###
        loss_sum = 0.0
        model.eval()
        with torch.no_grad():
            '''
            for j, (sentence, target, act) in enumerate(train_loader):
                sentence, target, act = sentence.to(device), target.to(device), act.to(device)
                prediction = model(sentence, target, current_epoch)
                loss_sum += criterion(sentence, target, act).item()
            '''
            for j, (img, lbl) in enumerate(test_loader):
                img, lbl = img.to(device), lbl.to(device)
                out = model(img)
                loss_sum += criterion(out, lbl)

        ltt = loss_sum / len(test_loader)
        ytt.append(ltt)

        print('Epoch: %d, train loss = %.4f, validation loss = %.4f' % (i+1, ltr, ltt))

    plt.figure()
    plt.plot(x, ytr, label = 'Train')
    plt.plot(x, ytt, label = 'Test')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(opts.figure_name)
    plt.show()

    torch.save(model, opts.model_name)
    torch.save(optim.state_dict(), opts.optim_name)

    print('Finished training %d epochs.\n%.4f minutes have elapsed.' % (opts.epoch_number, (time.time() - start_time) / 60))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='S2VT')
    parser.add_argument('--model_name', type=str, default='chatbot_model_default.pkl')
    parser.add_argument('--optim_name', type=str, default='chatbot_optim_default.pkl')
    parser.add_argument('--figure_name', type=str, default='chatbot_loss.png')
    parser.add_argument('--epoch_number', '-e', type=int, default=5)
    parser.add_argument('--batch_size', '-b', type=int, default=16)
    parser.add_argument('--optimizer', type=str, default='Adam')

    mutex = parser.add_mutually_exclusive_group(required = True)
    mutex.add_argument('--load_model', '-l', action='store_true', help = 'load a pre-existing model')
    mutex.add_argument('--new_model', '-n', action='store_true', help = 'create a new model')
    mutex.add_argument('--remove_model', '-r', action='store_true', help = 'remove a model')

    opts = parser.parse_args()
    main(opts)

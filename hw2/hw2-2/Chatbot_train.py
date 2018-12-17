import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import Data
import Seq2seqModel as seq2seq
import numpy as np
import matplotlib.pyplot as plt
import time
import gensim
import argparse
start_time = time.time()


### DEVICE CONFIGURATION ###

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


### MODEL ###

class Model(nn.Module):
    def __init__(self, encoder_layer=1, decoder_layer=1, w2v_size=250, one_hot_size=71347, optimizer='NONE'):
        super(Model, self).__init__()
        self.global_epoch = 0
        self.w2v_size = w2v_size
        self.hidden_size = 128
        self.one_hot_size = one_hot_size
        self.optimizer = optimizer
        self.batch_size = 20
        
        self.encoder_embedding = nn.Linear(self.w2v_size, self.hidden_size).to(device)
        self.decoder_embedding = nn.Linear(self.one_hot_size, self.hidden_size).to(device)
        
        self.encoder = seq2seq.EncoderRNN(hidden_size=self.hidden_size,
                                          embedding=self.encoder_embedding,
                                          n_layers=encoder_layer,
                                          dropout=0.5).to(device)
        self.decoder = seq2seq.LuongAttnDecoderRNN(attn_model='dot',
                                                   hidden_size=self.hidden_size,
                                                   embedding=self.decoder_embedding,
                                                   output_size=self.one_hot_size,
                                                   n_layers=decoder_layer).to(device)    

    def forward(self, x, x_len, y):
        # x: (batch size, max length, w2v dim)
        # x_len: (batch_size, 1)
        # input: (seq length, batch size, one hot size)
        # encoder_outputs: (max length, batch size, hidden size) - bidirectional
        # decoder_output: (batch size, one hot size)
        x = x.transpose(0, 1) # encoder requires (max_length, batch_size, w2v_dim)
        x_len = x_len.long().squeeze() # to int64
        y = y.long()
        pred_max_len = y.shape[1]
        decoder_outputs_list = []
        
        ### ENCODER ###
        encoder_outputs, encoder_hidden = self.encoder(x, x_len)

        ### INPUT TENSORS ###
        empty_input = torch.zeros((1, self.batch_size, self.one_hot_size), dtype=torch.float32).to(device)
        randn_input = torch.randn((1, self.batch_size, self.one_hot_size), dtype=torch.float32).to(device)
        
        ### FIRST DECODER INPUT ###
        decoder_output, decoder_hidden = self.decoder(randn_input, encoder_hidden[0:1], encoder_outputs)
        decoder_outputs_list.append(decoder_output)
        
        ### ONE-HOT TEACHER FORCING ###
        one_hot_sentence_list = []
        for i in range(self.batch_size):
            # convert index of words into one-hot vectors
            one_hot_list = []
            for j in range(pred_max_len):
                one_hot = torch.zeros((self.one_hot_size), dtype=torch.float32).to(device)
                one_hot[y[i, j].item()] = 1.0
                one_hot_list.append(one_hot)
            one_hot_sentence = torch.stack(one_hot_list, dim=0) # tensor of shape (max_length, one_hot_size)
            one_hot_sentence_list.append(one_hot_sentence)
        teacher_forcing_input = torch.stack(one_hot_sentence_list, dim=1) # shape (max_length, batch_size, one_hot_size)
        
        ### FOLLOWING DECODER INPUT ###
        for step in range(pred_max_len - 1):
            # length of prediction = max length of answer
            decoder_output, decoder_hidden = self.decoder(teacher_forcing_input[step:(step+1), :, :], decoder_hidden, encoder_outputs)
            decoder_outputs_list.append(decoder_output)
        
        # Turn a list of tensors into a tensor, e.g. n tensors of shape (x, y, z) stacked at dim=0
        # returns a tensor of shape (n, x, y, z)
        # Lastly, make sure that we return a tensor of shape (batch_size, max_length, one_hot_size)
        return torch.stack(decoder_outputs_list, dim=1)


### DATA PREPROCESSING ###

data_x1 = np.load('question_array1.npy')
#data_x2 = np.load('question_array2.npy')
#data_x2 = np.delete(data_x2, -1)   #I don't know why input is one more than label
#data_x = np.concatenate((data_x1,data_x2), axis = 0)
data_x = data_x1[:2000]
data_y = np.load('answer_array.npy')[:2000]
# length of data_x and data_y: 880623

# 880623 is too many

dataset = Data.ChatbotDataset(data_x=data_x, data_y=data_y, from_file=False)
train_loader = torch.utils.data.DataLoader(dataset=dataset, shuffle=True, batch_size=20, collate_fn=Data.collate_fn)
test_loader = torch.utils.data.DataLoader(dataset=dataset, shuffle=True, batch_size=20, collate_fn=Data.collate_fn)
# len(train_loader) = 44032


### CRITERION ###

def criterion(prediction, target, actual_length):
    # prediction is of shape (batch_size, max_sentence_length, one_hot_size)
    # target is of shape     (batch_size, max_sentence_length, 1)
    # nn.CrossEntropyLoss requires x = (batch_size, number_of_classes, *) and y = (batch_size, *)
    # y[0] corresponds to expected index of maximum of x[0]
    loss_sum = None
    prediction = prediction.transpose(1, 2)
    target = target.squeeze(2).long()
    c = nn.CrossEntropyLoss()

    for i in range(prediction.shape[0]):
        # trim both prediction and target to actual length
        # cross entropy between y' (prediction) and y (true label):
        # -ln(y'_i) where i is the index of the only non-zero element of y
        # p = prediction[i:(i+1), :, :(actual_length[i]-1)] <- trims EOS
        p = prediction[i:(i+1), :, :actual_length[i]]
        t = target[i:(i+1), :actual_length[i]]

        if loss_sum is None:
            loss_sum = c(p, t)
        else:
            loss_sum += c(p, t)

    # return the sum of losses of each entry in a batch
    return loss_sum


### MAIN ###

def main(opts):
    ### MODEL ###
    model_name = opts.model_name
    optim_name = model_name + '_optim.pkl'
    model_name += '.pkl'
    
    if not os.path.isfile(opts.word2vec):
        print('The Gensim model \'%s\' does not exist!' % opts.word2vec)
        exit()
    wv = gensim.models.KeyedVectors.load(opts.word2vec)
    # wv.vector_size is 250
    # len(wv.vocab) is 71473
    
    if opts.load_model:
        if not os.path.isfile(optim_name):
            print('Model \'%s\' does not exist!' % model_name)
            exit()
        else:
            model = torch.load(model_name).to(device)
            
    if opts.new_model:
        # create a model
        if os.path.isfile(model_name) and not opts.force:
            # model name conflict
            confirm = input('Model \'{}\' already exists. Overwrite it? [y/N] ' \
                            .format(model_name.strip('.pkl')))

            if confirm not in ['y', 'Y', 'yes', 'Yes', 'yeah']:
                print('Process aborted.')
                exit()

        # TODO: decide model name and parameters
        model = Model(w2v_size=wv.vector_size,
                      one_hot_size=len(wv.vocab),
                      optimizer=opts.optimizer).to(device)
        #### opts.optimizer is a string, 'Adam' or 'SGD' ####

    if opts.remove_model:
        if not os.path.isfile(opts.model_name):
            print('Model \'%s\' does not exist!' % model_name.strip('.pkl'))
            exit()
        else:
            try:
                os.remove(model_name)
                os.remove(optim_name)
            finally:
                print('Model \'%s\' removed successfully.' % model_name.strip('.pkl'))
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
        if not os.path.isfile(optim_name):
            print('Missing optimizer state!')
            exit()
        else:
            if opts.optimizer != model.optimizer:
                print('The specified optimizer \'%s\' is different from that \'%s\' of model!' %
                      (opts.optimizer, model.optimizer))
                exit()
            optim_state = torch.load(optim_name)
            optim.load_state_dict(optim_state)
            #### sometimes the parameter setting of optim of the load_model is not the same as line 192~193 ####
        

    x = range(opts.epoch_number)
    ytr = [] # loss of training over epochs
    ytt = [] # loss of testing over epochs


    for i in range(opts.epoch_number):

        ### TRAIN ###
        model.train()
        loss_sum = 0.0
        for j, (question, answer, q_len, a_len) in enumerate(train_loader):
            # <question> is of shape (batch_size=20, sentence_max_len, w2v_dim=250)
            # <answer> is of shape (batch_size=20, sentence_max_len, 1)
            # <q_len> and <a_len> of shape (20, 1) records actual length of each sentence

            print('Epoch: %d, batch: %d/%d' % (i+1+model.global_epoch, j+1, len(train_loader)), end='\r')
            question, answer, q_len, a_len = question.to(device), answer.to(device), q_len.to(device), a_len.to(device)

            optim.zero_grad()
            prediction = model(question, q_len, answer)
            loss = criterion(prediction, answer, a_len)

            loss.backward()
            optim.step()

            # loss of batch J
            loss_sum += loss.item()

        print('')
        ltr = loss_sum / len(train_loader)
        ytr.append(ltr)

        ### TEST ###
        loss_sum = 0.0
        model.eval()
        with torch.no_grad():
            for j, (question, answer, q_len, a_len) in enumerate(test_loader):
                question, answer, q_len, a_len = question.to(device), answer.to(device), q_len.to(device), a_len.to(device)
                prediction = model(question, q_len, answer)
                loss_sum += criterion(prediction, answer, a_len).item()
                if (j > 100):
                    # test at most 100 batch of data
                    break

        ltt = loss_sum / len(test_loader)
        ytt.append(ltt)

        #      Epoch: i
        print('       ', ' '*len(str(i)), ', train loss = %.4f, validation loss = %.4f' % (ltr, ltt), sep='')
        
        
    model.global_epoch += opts.epoch_number

    ### EVALUATION ###
    
    for i, (question, answer, q_len, a_len) in enumerate(test_loader):
        #question = question[0:1]
        #answer = answer[0:1]
        #q_len = q_len[0:1]
        #a_len = a_len[0:1]
        question_wordlist = []
        answer_wordlist = []
        prediction_wordlist = []

        model.eval()
        with torch.no_grad():
            question, answer, q_len, a_len = question.to(device), answer.to(device), q_len.to(device), a_len.to(device)
            prediction = model(question, q_len, answer)
        for j in range(q_len[0][0]):
            question_wordlist.append(wv.most_similar(positive=[question[0][j].cpu().numpy()], topn=1)[0][0])
        for j in range(a_len[0][0]):
            answer_wordlist.append(wv.index2entity[answer[0][j].long().item()])
            prediction_wordlist.append(wv.index2entity[prediction[0][j].max(0)[1].item()])
        print("\nQuestion: ", *question_wordlist, sep='')
        print("Answer: ", *answer_wordlist, sep='')
        print("Prediction: ", *prediction_wordlist, sep='')
        print("")
        if i > 2:
            break
    

    plt.figure()
    plt.plot(x, ytr, label = 'Train')
    plt.plot(x, ytt, label = 'Test')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(opts.figure_name)
    plt.show()

    torch.save(model, model_name)
    torch.save(optim.state_dict(), optim_name)

    print('Finished training %d epochs.\n%.4f minutes have elapsed.' % (opts.epoch_number, (time.time() - start_time) / 60))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='chatbot')
    parser.add_argument('--model_name', type=str, default='chatbot_model_default')
    parser.add_argument('--figure_name', type=str, default='chatbot_loss.png')
    parser.add_argument('--epoch_number', '-e', type=int, default=5)
    parser.add_argument('--batch_size', '-b', type=int, default=16)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--word2vec', type=str, default='./Jeffery/word2vec_wv_Jeff.wv', help='the gensim model')
    parser.add_argument('--force', action='store_true', help='force a new model')

    mutex = parser.add_mutually_exclusive_group(required = True)
    mutex.add_argument('--load_model', '-l', action='store_true', help='load a pre-existing model')
    mutex.add_argument('--new_model', '-n', action='store_true', help='create a new model')
    mutex.add_argument('--remove_model', '-r', action='store_true', help='remove a model')

    opts = parser.parse_args()
    main(opts)

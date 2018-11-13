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

Inheritance = False

class Model(nn.Module):
    def __init__(self, encoder_layer=1, decoder_layer=1, w2v_size=250, one_hot_size=71347, batch_size=20, optimizer='NONE'):
        super(Model, self).__init__()
        self.encoder_layer = encoder_layer
        self.global_epoch = 0
        self.w2v_size = w2v_size
        self.hidden_size = 1024
        self.one_hot_size = one_hot_size
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.global_loss_lost = []
        self.schedule_rate = 1
        self.batch_num = 0
        
        self.encoder_embedding = nn.Linear(self.w2v_size, self.hidden_size).to(device)
        self.decoder_embedding = nn.Linear(self.one_hot_size, self.hidden_size).to(device)
        # old models such as model_2000 and model_800000 uses Linear(one_hot_size=71347, hidden_size=128)
        # new models use Linear(w2v_size=250, hidden_size=128)
        
        self.encoder = seq2seq.EncoderRNN(hidden_size=self.hidden_size,
                                          embedding=self.encoder_embedding,
                                          n_layers=encoder_layer,
                                          dropout=0.0).to(device)
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
        pred_max_len = y.shape[1]
        decoder_outputs_list = []
        
        ### ENCODER ###
        encoder_outputs, encoder_hidden = self.encoder(x, x_len)

        ### INPUT TENSORS ###
        empty_input = torch.zeros((1, x.shape[1], self.one_hot_size), dtype=torch.float32).to(device)
        
        ### FIRST DECODER INPUT ###
        decoder_output, decoder_hidden = self.decoder(empty_input, encoder_hidden[0:int(self.encoder_layer/2+1)], encoder_outputs)
        decoder_outputs_list.append(decoder_output)
        
        ### ONE-HOT TEACHER FORCING ###
        if self.decoder_embedding.in_features == self.one_hot_size:
            y = y.long()
            # y: (batch size, max length, 1)
            assert y.shape[-1] == 1, "Wrong size of tensor {} for one-hot teacher forcing!".format(y.shape)
            one_hot = torch.zeros(y.shape[0], y.shape[1], self.one_hot_size).to(device).scatter_(2, y, 1)
            '''
            one_hot_sentence_list = []
            for i in range(x.shape[1]):
                # convert index of words into one-hot vectors
                one_hot_list = []
                for j in range(pred_max_len):
                    one_hot = torch.zeros((self.one_hot_size), dtype=torch.float32).to(device)
                    one_hot[y[i, j].item()] = 1.0
                    one_hot_list.append(one_hot)
                one_hot_sentence = torch.stack(one_hot_list, dim=0) # tensor of shape (max_length, one_hot_size)
                one_hot_sentence_list.append(one_hot_sentence)
            teacher_forcing_input = torch.stack(one_hot_sentence_list, dim=1)
            '''
            teacher_forcing_input = one_hot.transpose(1,0)
            # shape (max_length, batch_size, one_hot_size)
            
        elif self.decoder_embedding.in_features == self.w2v_size:
            # y: (batch size, max length, w2v size)
            assert y.shape[-1] == self.w2v_size, "Wrong size of tensor {} for w2v teacher forcing!"
        
        else:
            raise NotImplementedError("Unknown decoder embedding layer size: (%d, %d)" %
                                      self.decoder.in_features, self.decoder.out_features)
        
        ### FOLLOWING DECODER INPUT ###
        for step in range(pred_max_len - 1):
            # length of prediction = max length of answer
            if np.random.rand() < self.schedule_rate:
                decoder_output, decoder_hidden = self.decoder(teacher_forcing_input[step:(step+1), :, :], decoder_hidden, encoder_outputs)
                decoder_outputs_list.append(decoder_output)
            else:
                last = decoder_output.reshape(1, decoder_output.shape[0], decoder_output.shape[1])
                decoder_output, decoder_hidden = self.decoder(last, decoder_hidden, encoder_outputs)
                decoder_outputs_list.append(decoder_output)
        ### Change schedule sampling rate ###
        self.batch_num += 1
        self.schedule_rate = 1000/(1000+np.exp(self.batch_num/1000))
        # Turn a list of tensors into a tensor, e.g. n tensors of shape (x, y, z) stacked at dim=0
        # returns a tensor of shape (n, x, y, z)
        # Lastly, make sure that we return a tensor of shape (batch_size, max_length, one_hot_size)
        return torch.stack(decoder_outputs_list, dim=1)


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

    # return the average of losses of each entry in a batch
    return loss_sum / prediction.shape[0]


###

def index2vector(t, l, wv):
    # t: (batch size, max len, 1), word indices
    # l: (batch size, 1), respective length of sentences
    # wv: word2vec model
    vector_sentence_list = []
    
    for i in range(t.shape[0]):
        # for each sentence
        coded_eos = false
        vector_list = []
        
        for j in range(t.shape[1]):
            # for each index
            index = t[0][1].long().item()
            vector = torch.Tensor(wv.vectors[index])
            if index == 0 and coded_eos:
                # override vector with zeros
                vector = torch.zeros((wv.vector_size), dtype=torch.float32)
            elif index == 0 and not coded_eos:
                coded_eos = True
            vector_list.append(vector)
        
        vector_sentence = torch.stack(vector_list, dim=0) # tensor of shape (max_length, w2v_size)
        vector_sentence_list.append(vector_sentence)
    return torch.stack(vector_sentence_list, dim=0) # tensor of shape (batch size, max len, w2v size)


### MAIN ###

def main(opts):
    ### DATA ###
    if opts.data_number < opts.batch_size:
        print("Data number less than a batch!")
        exit()
    
    data_x1 = np.load('../question_array1.npy')
    
    if opts.data_number > 440312:
        data_x2 = np.load('../question_array2.npy')
        data_x2 = np.delete(data_x2, -1)   #I don't know why input is one more than label
        data_x = np.concatenate((data_x1,data_x2), axis = 0)[:opts.data_number]
    else:
        data_x = data_x1[:opts.data_number]
    
    data_y = np.load('../answer_array.npy')[:opts.data_number]
    # max length of data_x and data_y: 880623

    dataset = Data.ChatbotDataset(data_x=data_x, data_y=data_y, from_file=False)
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               shuffle=True,
                                               batch_size=opts.batch_size,
                                               collate_fn=Data.collate_fn)
    test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              shuffle=True,
                                              batch_size=opts.batch_size,
                                              collate_fn=Data.collate_fn)
    # len(train_loader) = opts.data_number / opts.batch_size = 44032 (max data, default batch size)
    
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
        model = Model(encoder_layer=2,
                      decoder_layer=2,
                      w2v_size=wv.vector_size,
                      one_hot_size=len(wv.vocab),
                      optimizer=opts.optimizer).to(device)
        if Inheritance:
            inheritance_model = torch.load(Inheritance)
            model.encoder_embedding = inheritance_model.encoder_embedding
            model.decoder_embedding = inheritance_model.decoder_embedding
            model.encoder = inheritance_model.encoder
            model.decoder = inheritance_model.decoder

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
        

    x = range(opts.epoch_number)
    ytr = [] # loss of training over epochs
    ytt = [] # loss of testing over epochs

    ### TRAIN ###
    for i in range(opts.epoch_number):
        model.train()
        loss_sum = 0.0
        try:
            for j, (question, answer, q_len, a_len) in enumerate(train_loader):
                # <question> is of shape (batch_size=20, sentence_max_len, w2v_dim=250)
                # <answer> is of shape (batch_size=20, sentence_max_len, 1)
                # <q_len> and <a_len> of shape (20, 1) records actual length of each sentence

                if model.decoder_embedding.in_features == wv.vector_size:
                    # convert answer (indices) into vectors
                    # otherwise the model turns indices into one-hot vectors
                    answer = index2vector(answer, wv)
                question, answer, q_len, a_len = question.to(device), answer.to(device), q_len.to(device), a_len.to(device)

                optim.zero_grad()
                prediction = model(question, q_len, answer)
                loss = criterion(prediction, answer, a_len)

                loss.backward()
                total_norm = nn.utils.clip_grad_norm_(model.parameters(), 5)
                
                norm = ''
                for param in model.parameters():
                    if param.grad.norm().item() < 5:
                        norm = "OK"
                    else:
                        norm = str(param.grad.norm().item())
                
                optim.step()

                # loss of batch J
                loss_sum += loss.item()
                print('Epoch: %d, batch: %d/%d, loss: %.4f, schedule_rate: %.4f, clip:%s' %
                      (i+1+model.global_epoch, j+1, len(train_loader), loss.item(), model.schedule_rate, norm), end='\r')
        except KeyboardInterrupt:
            confirm = input("\b\bProcess interrupted. Save current progress? [Y/n] ")
            if confirm not in ['n', 'N', 'no', 'No', 'dont', 'don\'t', 'do not', 'do not save it',
                               'Do not save current progress.']:
                model.global_epoch += i
                torch.save(model, model_name)
                torch.save(optim.state_dict(), optim_name)
            print('Process aborted.')
            exit()

        print('')
        ltr = loss_sum / len(train_loader)
        ytr.append(ltr)
        print("Saving model")
        torch.save(model, model_name)
        torch.save(optim.state_dict(), optim_name)

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
    
    '''
    # TODO: RECORD LOSS
    #       HANDLE AttributeError
    plt.figure()
    plt.plot(x, ytr, label = 'Train')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(opts.figure_name)
    plt.show()
    '''

    print('Finished training %d epochs.\n%.4f minutes have elapsed.' % (opts.epoch_number, (time.time() - start_time) / 60))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='chatbot')
    parser.add_argument('data_number', type=int) # required
    parser.add_argument('--model_name', type=str, default='chatbot_model_default')
    parser.add_argument('--figure_name', type=str, default='chatbot_loss.png')
    parser.add_argument('--epoch_number', '-e', type=int, default=5)
    parser.add_argument('--batch_size', '-b', type=int, default=20)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--word2vec', type=str, default='../Jeffery/word2vec_wv_Jeff.wv', help='the gensim model')
    parser.add_argument('--force', action='store_true', help='force a new model')

    mutex = parser.add_mutually_exclusive_group(required = True)
    mutex.add_argument('--load_model', '-l', action='store_true', help='load a pre-existing model')
    mutex.add_argument('--new_model', '-n', action='store_true', help='create a new model')
    mutex.add_argument('--remove_model', '-r', action='store_true', help='remove a model')

    opts = parser.parse_args()
    main(opts)

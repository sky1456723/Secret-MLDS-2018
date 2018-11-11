import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#print(device)

'''
my embedding is used to embed the word to hidden vector
if we have a pretrained embedding layer, we can use it strictly
else we can use one Linear layer
'''
no_glove = True
one_hot_size = 20
hidden_size = 10
batch_size = 3
my_embedding = nn.Linear(one_hot_size,hidden_size) if no_glove else None



'''
input:
input_seq: batch of input sentences; shape=(max_length, batch_size)
input_lengths: list of sentence lengths corresponding to each sentence in the batch;
shape=(batch_size)
hidden: hidden state; shape=(n_layers x num_directions, batch_size, hidden_size)

output:
outputs: output features from the last hidden layer of the GRU (sum of bidirectional outputs); 
shape=(max_length, batch_size, hidden_size);
outputs can be used for attention
hidden: updated hidden state from GRU; 
shape=(n_layers x num_directions, batch_size, hidden_size);
you can choose (:decoder_layers=1,batch_size,hidden_size) as input of the initial decoder hidden state
'''
##### input length must be descending order ######
class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden) 
        # outputs size(max length of sequence,batch,hidden * 2)
        # Unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        # print(outputs.shape)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden

class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))
            # after cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)),the third dimension is
            # 2*hidden_size,and then use Linear to transfer to hidden_size

    def dot_score(self, hidden, encoder_output):
        # print(torch.sum(hidden * encoder_output, dim=2).shape)
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)
        # v is a parameter just like w and b which can be leart by SGD

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)
        # return size(batch_size,1,sequence_length)
        # pay attention to the return size which is batch first rather than sequence_length first

'''
input:
input_step: one time step (one word) of input sequence batch; shape=(1, batch_size)
last_hidden: final hidden layer of GRU; shape=(n_layers x num_directions, batch_size, hidden_size)
encoder_outputs: encoder model’s output; shape=(max_length, batch_size, hidden_size)

output:
output: softmax normalized tensor giving probabilities of each word being the correct next word in the 
decoded sequence; shape=(batch_size, voc.num_words)
hidden: final hidden state of GRU; shape=(n_layers x num_directions, batch_size, hidden_size)

Luong is a person who proposed Luong attention method.
For the decoder, we will manually feed our batch one time step at a time. 
This means that our embedded word tensor and GRU output will both have shape (1, batch_size, hidden_size).
'''
class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        # transfer the hidden_size to word_size

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        # rnn_output size: (n_layers = 1,batch_size,hidden_size)
        rnn_output, hidden = self.gru(embedded, last_hidden)  
        
        # Calculate attention weights from the current GRU output
        # encoder_ouput size: (max time_sequence,encoder batch size,hidden size)
        # attn_weights size : (batch_size,1,sequence_length)
        attn_weights = self.attn(rnn_output, encoder_outputs)

        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        # attn_weights size: (batch size,1, hidden size)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        # concat_input size : (batch size, hidden size*2)
        
        concat_output = torch.tanh(self.concat(concat_input))
        # concat_output size: (batch size, hidden siez)
        output = self.out(concat_output)
        #output = F.softmax(output, dim=1)
        return output, hidden
        # output size: (batch,one hot size)
        # hidden will be used as last_hidden for the next time step(for attention or as input to gru next time step)

'''
encoder_layer = 1
decoder_layer = 1

test_encoder = EncoderRNN(hidden_size = hidden_size,embedding = my_embedding,n_layers = encoder_layer,dropout = 0)
test_decoder = LuongAttnDecoderRNN('dot',my_embedding, hidden_size, output_size = one_hot_size,n_layers =1)
print(test_encoder)
print(test_decoder)

test_en_input = torch.zeros((5,3,20),dtype = torch.float)
test_en_length = torch.tensor([5,2,1])
test_en_output,test_en_hidden = test_encoder(test_en_input,test_en_length)
print(test_en_output.shape)
print(test_en_hidden.shape)

decoder_hidden = test_en_hidden[:decoder_layer]
test_de_input = torch.rand((1,3,20))

# decoder will get hidden vector of gru of last time step and the word vector, 
# output hidden state of this time step and the prediction of this time step
test_de_out,test_de_next_hidden = test_decoder(test_de_input,decoder_hidden,test_en_output)
print(test_de_out.shape)
print(test_de_next_hidden.shape)
'''

def main():
    ee = nn.Linear(250, hidden_size)
    de = nn.Linear(75000, hidden_size)
    e = EncoderRNN(hidden_size = hidden_size,
                   embedding = ee,
                   n_layers = 1,
                   dropout = 0.5).to(device)
    d = LuongAttnDecoderRNN('dot',
                            embedding = de,
                            hidden_size = hidden_size, 
                            output_size = 75000,
                            n_layers = 1).to(device)
    print(device)
    input = torch.randn((7, 3, 250), dtype=torch.float32).to(device)
    # batch size = 3
    # max len = 7
    # feature = 250
    input_len = torch.tensor([7, 4, 3]).to(device)
    randn_input = torch.randn((1, 3, 75000), dtype=torch.float32).to(device)
    encoder_output, encoder_hidden = e(input, input_len)
    
    decoder_output, decoder_hidden = d(randn_input, encoder_hidden[0:1], encoder_output)
    for step in range(7 - 1):
            decoder_output, decoder_hidden = d(decoder_output.unsqueeze(0), decoder_hidden, encoder_output)
            print(decoder_output)

if __name__ == '__main__':
    main()
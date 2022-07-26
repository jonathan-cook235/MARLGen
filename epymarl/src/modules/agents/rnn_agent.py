import torch.nn as nn
import torch.nn.functional as F
import torch as torch

class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args
        # Below is for gathering
        # input_shape = int((input_shape+2)/2)
        # Below is for herding
        input_shape = int((input_shape - 200))
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.args.use_rnn = False
        if self.args.use_rnn:
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        # print('NETWORK SHAPES')
        # print(self.input_shape)
        # print(inputs.shape)
        # inputs = torch.reshape(inputs, (1, 544))
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        q = self.fc2(h)
        return q, h


import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTM(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, num_layers, dropout):
        super(LSTM, self).__init__()
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.drop = nn.Dropout(dropout)  # dropout used for embedding and final layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=input_size)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, hidden_state):
        embedding = self.drop(self.embedding(input_ids))
        output, hidden_state = self.lstm(embedding, hidden_state)
        output = self.drop(output)
        logits = self.fc(output)
        return logits, (hidden_state[0].detach(), hidden_state[1].detach())

    def init_states(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))

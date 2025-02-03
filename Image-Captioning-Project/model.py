import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # Using ResNet-34 for a lighter model
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2, batch_size=64, drop_out = 0.2):
        super(DecoderRNN, self).__init__()
        # TODO: Complete this function
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.drop_out = drop_out
        # Define layers
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=self.drop_out)
        self.fc = nn.Linear(hidden_size, vocab_size)
        # initialize the hidden state 
        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        ''' At the start of training, we need to initialize a hidden state;
           there will be none because the hidden state is formed based on perviously seen data.
           So, this function defines a hidden state with all zeroes and of a specified size.'''
        # The axes dimensions are (n_layers, batch_size, hidden_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device))

    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])  # Exclude the <end> token (1, caption_length - 1, embedding_dim)
        # TODO: Complete this function
        features = features.view(features.size()[0], 1, -1) # (batch_size, 1, embed_size)
        # Combine features and embeddings to feed the LSTM model
        inputs = torch.cat((features, embeddings), dim =1) # (batch_size, caption_length, embed_size)
        # Pass through LSTM
        lstm_outputs, self.hidden = self.lstm(inputs, self.hidden) # (batch_size, caption_length, hidden_size)
        lstm_outputs_shape = list(lstm_outputs.shape)
        lstm_outputs = lstm_outputs.reshape(lstm_outputs_shape[0]*lstm_outputs_shape[1], -1) # (batch_size*caption_length, hidden_size)
        # Get the probability for the next word
        outputs = self.fc(lstm_outputs) # (batch_size, caption_length, vocab_size)
        outputs = outputs.reshape(lstm_outputs_shape[0], lstm_outputs_shape[1], -1) # (batch_size, caption_length, vocab_size)
 
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        "accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len)"
#         predicted_sentence = []
#         if states == None:
#             states = (torch.randn(1, 1, self.hidden_size).to(inputs.device),
#                       torch.randn(1, 1, self.hidden_size).to(inputs.device))
#         for i in range(max_len):
#             hiddens, states = self.lstm(inputs, states)
#             outputs = self.fc(hiddens.squeeze(1))
#             _, predicted = outputs.max(1)
#             predicted_sentence.append(predicted.item())
#             inputs = self.embed(predicted).unsqueeze(1)
#         return predicted_sentence
        output = []
        batch_size = inputs.shape[0]
        hidden = (torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device),
              torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device))
        while True:
            lstm_out, hidden = self.lstm(inputs, hidden)
            outputs = self.fc(lstm_out)
            outputs = outputs.squeeze(1)
            _, max_pred_index = torch.max(outputs, dim = 1)
            output.append(max_pred_index.cpu().numpy()[0].item())
            if (max_pred_index == 1):
                break
            inputs = self.embed(max_pred_index)
            inputs = inputs.unsqueeze(1)
        return output

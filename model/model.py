import torch
import torch.nn as nn
import torch.nn.functional as F

class OlidCnnNet(nn.Module):

    def __init__(self, seq_len, num_vocab, embedding_size=64, conv_out=2, threshold=0.5):
        super().__init__()
        self.threshold = threshold

        # Embedding layer definition
        self.embedding = nn.Embedding(num_vocab, embedding_size, padding_idx=0)
      
        # Convolution layers definition
        self.conv_1 = nn.Conv2d(1, conv_out, (2, embedding_size))
        self.conv_2 = nn.Conv2d(1, conv_out, (3, embedding_size))
        self.conv_3 = nn.Conv2d(1, conv_out, (4, embedding_size))
      
        # Max pooling layers definition
        self.pool_1 = nn.MaxPool2d((1, 2))
        self.pool_2 = nn.MaxPool2d((1, 3))
        self.pool_3 = nn.MaxPool2d((1, 4))
        
        # dropout
        self.dropout = nn.Dropout(0.5)
        
        # find fc layer input size
        test = torch.randint(10, (1, seq_len))
        out = self.conv_layers(test)
      
        # Fully connected layer definition
        self.fc = nn.Linear(out.size(1), 1)
        
    def conv_layers(self, x):

        x = self.embedding(x).unsqueeze(1)
      
        # Convolution layer 1 is applied
        x1 = self.conv_1(x).squeeze(-1)
        x1 = F.relu(x1)
        x1 = self.pool_1(x1)
      
        # Convolution layer 2 is applied
        x2 = self.conv_2(x).squeeze(-1)
        x2 = F.relu(x2)
        x2 = self.pool_1(x2)
   
        # Convolution layer 3 is applied
        x3 = self.conv_3(x).squeeze(-1)
        x3 = F.relu(x3)
        x3 = self.pool_1(x3)
      
        # The output of each convolutional layer is concatenated into a unique vector
        union = torch.cat((x1, x2, x3), 2)
        union = union.reshape(union.size(0), -1)
        return union

    def forward(self, x):

        union = self.conv_layers(x)
        union = self.dropout(union)

        out = self.fc(union)

        out = torch.sigmoid(out)
        return out.squeeze()
    
    def set_threshold(self, threshold):
        self.threshold = threshold
        
    def inference(self, data):
        output = self.forward(data)
        label = torch.where(output > self.threshold, 1., 0.)
        return label
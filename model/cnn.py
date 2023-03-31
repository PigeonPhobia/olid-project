import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
        self.conv_4 = nn.Conv2d(1, conv_out, (5, embedding_size))
      
        # Max pooling layers definition
        self.pool_1 = nn.MaxPool2d((1, 2))
        self.pool_2 = nn.MaxPool2d((1, 3))
        self.pool_3 = nn.MaxPool2d((1, 4))
        self.pool_4 = nn.MaxPool2d((1, 5))
        
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
        x2 = self.pool_2(x2)
   
        # Convolution layer 3 is applied
        x3 = self.conv_3(x).squeeze(-1)
        x3 = F.relu(x3)
        x3 = self.pool_3(x3)
        
        # Convolution layer 4 is applied
        x4 = self.conv_4(x).squeeze(-1)
        x4 = F.relu(x4)
        x4 = self.pool_4(x4)
      
        # The output of each convolutional layer is concatenated into a unique vector
        union = torch.cat((x1, x2, x3, x4), 2)
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

class OlidCnnAdvance(nn.Module):

    def __init__(self, params):
        super().__init__()
        self.params = params

        # Embedding layer
        self.embedding = nn.Embedding(self.params.vocab_size, self.params.embed_size)

        self.flatten_size = 0

        self.conv_layers = nn.ModuleDict()
        for ks in self.params.conv_kernel_sizes:
            self.conv_layers['conv_{}'.format(ks)] = nn.Conv2d(in_channels=1,
                                                               out_channels=self.params.out_channels,
                                                               kernel_size=(ks, self.params.embed_size),
                                                               stride=self.params.conv_stride,
                                                               padding=(self.params.conv_padding, 0))
            # Calculate the length of the conv output
            conv_out_size = self._calc_conv_output_size(self.params.seq_len,
                                                        ks,
                                                        self.params.conv_stride,
                                                        self.params.conv_padding)
            # Calculate the length of the maxpool output
            maxpool_out_size = self._calc_maxpool_output_size(conv_out_size,
                                                              self.params.maxpool_kernel_size,
                                                              self.params.maxpool_padding,
                                                              self.params.maxpool_kernel_size,
                                                              1)
            # Add all lengths together
            self.flatten_size += maxpool_out_size
            
        self.flatten_size *= self.params.out_channels

        self.maxpool_layers = nn.ModuleDict()
        for ks in self.params.conv_kernel_sizes:
            #self.maxpool_layers['maxpool_{}'.format(ks)] = nn.MaxPool2d(kernel_size=(self.params.maxpool_kernel_size, self.params.embed_size))
            self.maxpool_layers['maxpool_{}'.format(ks)] = nn.MaxPool2d(kernel_size=(1, self.params.maxpool_kernel_size))
            #self.maxpool_layers['maxpool_{}'.format(ks)] = nn.MaxPool1d(kernel_size=self.params.maxpool_kernel_size)

        self.linear_sizes = [self.flatten_size] + self.params.linear_sizes

        # Define set of fully connected layers (Linear Layer + Activation Layer) * #layers
        self.linears = nn.ModuleList()
        for i in range(0, len(self.linear_sizes)-1):
            self.linears.append(nn.Linear(self.linear_sizes[i], self.linear_sizes[i+1]))
            self.linears.append(nn.ReLU())
            if self.params.linear_dropout > 0.0:
                self.linears.append(nn.Dropout(p=self.params.linear_dropout))


        self.out = nn.Linear(self.linear_sizes[-1], self.params.output_size)

    def forward(self, inputs):
        batch_size, seq_len = inputs.shape
        X = self.embedding(inputs)
        # Embedding output shape: N x S x E
        # Turn (N x S x E) into (N x C_in=1 x S x E) for CNN
        # (note: embedding dimension = input channels)
        X = X.unsqueeze(1)
        # Conv1d input shape: batch size x input channels x input length
        all_outs = []
        for ks in self.params.conv_kernel_sizes:
            out = self.conv_layers['conv_{}'.format(ks)](F.relu(X))
            out = self.maxpool_layers['maxpool_{}'.format(ks)](out.squeeze(-1))
            out = out.view(batch_size, -1)
            all_outs.append(out)
        # Concatenate all outputs from the different conv layers
        X = torch.cat(all_outs, 1)
        # Go through all layers (dropout, fully connected + activation function)
        for l in self.linears:
            X = l(X)    
        # Push through last linear layer
        X = self.out(X)
        # Return log probabilities
        return F.log_softmax(X, dim=1)


    def _calc_conv_output_size(self, seq_len, kernel_size, stride, padding):
        return int(((seq_len - kernel_size + 2*padding) / stride) + 1)

    def _calc_maxpool_output_size(self, seq_len, kernel_size, padding, stride, dilation):
        return int(math.floor( ( (seq_len + 2*padding - dilation*(kernel_size-1) - 1) / stride ) + 1 ))        


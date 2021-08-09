import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import TensorDataset, DataLoader

from models.BaseClassifier import BaseClassifier
from torch import nn, lstm, LongTensor, FloatTensor


class MyLSTMSequenceModule(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class, pretrained_embedding=None, cuda=False):
        super(MyLSTMSequenceModule, self).__init__()
        if pretrained_embedding is not None:
            self.embedding_layer = nn.Embedding.from_pretrained(pretrained_embedding)
        else:
            self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim//2, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(embedding_dim, num_class)
        self.softmax = nn.Softmax(dim=1)
        self.hidden_dim = hidden_dim
        self.cuda = cuda

    def init_hidden(self, batch_size):
        h0, c0 = torch.zeros(2, batch_size, self.hidden_dim // 2), torch.zeros(2, batch_size, self.hidden_dim // 2)
        h0 = FloatTensor(h0)
        c0 = FloatTensor(c0)
        if self.cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()
        return h0, c0

    def forward(self, x, xlen):
        # print("x lstm", x.shape)
        # print("x len", xlen.shape)
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        hidden = self.init_hidden(batch_size)
        x = self.embedding_layer(x)
        x = pack_padded_sequence(x, xlen.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, cell) = self.lstm(x, hidden)
        x = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        x = self.linear(x)
        y_prob = self.softmax(x)
        y_pred = torch.argmax(y_prob, dim=1)
        return y_pred, y_prob


class MyLSTMSequenceClassifier(BaseClassifier):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class, pretrained_embedding=None, cuda=False, num_epoch=100):
        super(MyLSTMSequenceClassifier, self).__init__()
        self.cuda = cuda
        if pretrained_embedding is not None:
            pretrained_embedding = FloatTensor(pretrained_embedding)
            if self.cuda:
                pretrained_embedding = pretrained_embedding.cuda()
        self.classifier = MyLSTMSequenceModule(vocab_size, embedding_dim, hidden_dim, num_class, pretrained_embedding=pretrained_embedding, cuda=self.cuda)
        if self.cuda:
            self.classifier.to(pretrained_embedding.get_device())
    def train(self, train_x, train_y, test_x, test_y, num_epoch=100):
        ce_loss = nn.CrossEntropyLoss()
        if self.cuda:
            ce_loss = ce_loss.cuda()
        optimizer = torch.optim.SGD(self.classifier.parameters(), lr=0.01, momentum=0.1)
        train_x_text = np.asarray(list(train_x['text']))
        test_x_text = np.asarray(list(test_x['text']))
        train_x_len = np.asarray(list(train_x['text_length']))
        test_x_len = np.asarray(list(test_x['text_length']))
        train_y = np.asarray(list(train_y))
        test_y = np.asarray(list(test_y))
        train_x_text_tensor, train_x_length_tensor, train_y_tensor = torch.from_numpy(train_x_text), torch.from_numpy(train_x_len), torch.from_numpy(train_y)
        test_x_text_tensor, test_x_length_tensor, test_y_tensor = torch.from_numpy(test_x_text), torch.from_numpy(test_x_len), torch.from_numpy(test_y)
        if self.cuda:
            train_x_text_tensor, train_x_length_tensor, train_y_tensor = train_x_text_tensor.cuda(), train_x_length_tensor.cuda(), train_y_tensor.cuda()
            test_x_text_tensor, test_x_length_tensor, test_y_tensor = test_x_text_tensor.cuda(), test_x_length_tensor.cuda(), test_y_tensor.cuda()
        train_y_hat = []
        dataset = TensorDataset(train_x_text_tensor, train_x_length_tensor, train_y_tensor)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        self.classifier.train()
        for i in range(num_epoch):
            total_loss = 0
            for x, x_len, y in dataloader:
                optimizer.zero_grad()
                y_pred, y_prob = self.classifier(x, x_len)
                if i == num_epoch - 1:
                    train_y_hat.extend(y_pred.cpu().detach().numpy())
                loss = ce_loss(y_prob, y)
                loss.backward()
                total_loss = total_loss + loss
                optimizer.step()
            if i % 10 == 0:
                print(f"epoch {i} loss: ", total_loss)

        test_y_hat = []
        test_dataset = TensorDataset(test_x_text_tensor, test_x_length_tensor, test_y_tensor)
        test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)
        self.classifier.eval()
        for x, x_len, y in test_dataloader:
            y_pred, y_prob = self.classifier(x, x_len)
            test_y_hat.extend(y_pred.cpu().detach().numpy())

        train_y_hat, test_y_hat = np.asarray(train_y_hat), np.asarray(test_y_hat)
        train_accuracy = accuracy_score(train_y, train_y_hat)
        test_accuracy = accuracy_score(test_y, test_y_hat)
        train_f1 = f1_score(train_y, train_y_hat)
        test_f1 = f1_score(test_y, test_y_hat)

        return train_accuracy, train_f1, test_accuracy, test_f1

    def predict(self, x):
        x_t, x_l = x
        y_pred, y_prob = self.classifier(x_t, x_l)
        return y_pred.cpu().detach().numpy(), y_prob.cpu().detach().numpy()

    def load_checkpoint(self, checkpoint_path):
        self.classifier.load_state_dict(torch.load(checkpoint_path))
        self.classifier.eval()

    def save_checkpoint(self, checkpoint_path):
        torch.save(self.classifier.state_dict(), checkpoint_path)

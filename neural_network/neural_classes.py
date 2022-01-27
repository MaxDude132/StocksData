import numpy
import torch

from torch import nn


class Normalizer:
	def __init__(self):
		self.mu = None
		self.std = None

	def fit_transform(self, x):
		self.mu = numpy.mean(x, axis=(0), keepdims=True)
		self.std = numpy.std(x, axis=(0), keepdims=True)
		return (x - self.mu) / self.std

	def inverse_transform(self, x):
		return(x * self.std) + self.mu


class Dataset(torch.utils.data.Dataset):
	def __init__(self, x, y):
		x = numpy.expand_dims(x, 2)
		self.x = x.astype(numpy.float32)
		self.y = y.astype(numpy.float32)

	def __len__(self):
		return len(self.x)

	def __getitem__(self, idx):
		return (self.x[idx], self.y[idx])


class LSTMModel(torch.nn.Module):
	def __init__(self, input_size=1, hidden_layer_size=32, num_layers=2, output_size=1, dropout=0.2):
		super().__init__()
		self.hidden_layer_size = hidden_layer_size

		self._linear_1 = nn.Linear(input_size, hidden_layer_size)
		self._relu = nn.ReLU()
		self._lstm = nn.LSTM(hidden_layer_size, hidden_size=self.hidden_layer_size, num_layers=num_layers, batch_first=True)
		self._dropout = nn.Dropout(dropout)
		self._linear_2 = nn.Linear(num_layers * hidden_layer_size, output_size)

		self._init_weights()

	def _init_weights(self):
		for name, parameter in self._lstm.named_parameters():
			if "bias" in name:
				nn.init.constant_(parameter, 0.0)
			elif "weight_ih" in name:
				nn.init.kaiming_normal_(parameter)
			elif "weight_hh" in name:
				nn.init.orthogonal_(parameter)

	def forward(self, x):
		batchsize = x.shape[0]

		x = self._linear_1(x)
		x = self._relu(x)

		lstm_out, (h_n, c_n) = self._lstm(x)

		x = h_n.permute(1, 0, 2).reshape(batchsize, -1)

		x = self._dropout(x)
		predictions = self._linear_2(x)

		return predictions[:,-1]






















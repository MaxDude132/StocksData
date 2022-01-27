import time
import numpy
import torch

from torch import nn

from db_operations import stocks_db_operations
from neural_network import neural_data, neural_classes, plotting_operations
from neural_network.config import config


class SymbolProcess:
	def __init__(self, symbol, symbol_class=neural_data.CurrencyData):
		self.symbol = symbol
		self.symbol_data = symbol_class(symbol)

		self._window_size = config["data"]["window_size"]
		self._train_split_size = config["data"]["train_split_size"]

		self._batch_size = config["training"]["batch_size"]
		self._device = config["training"]["device"]
		self._learning_rate = config["training"]["learning_rate"]
		self._scheduler_step_size = config["training"]["scheduler_step_size"]

		self._input_size = config["model"]["input_size"]
		self._lstm_size = config["model"]["lstm_size"]
		self._num_lstm_layers = config["model"]["num_lstm_layers"]
		self._dropout = config["model"]["dropout"]

		self.close_prices_normalized = self._normalize()
		self._data_x, self._data_x_unseen = self._prepare_data_x()
		self._data_y = self._prepare_data_y()

		self.split_index = int(self._data_y.shape[0] * self._train_split_size)

		self._data_x_train = self._data_x[:self.split_index]
		self._data_x_validate = self._data_x[self.split_index:]
		self._data_y_train = self._data_y[:self.split_index]
		self._data_y_validate = self._data_y[self.split_index:]

		self._dataset_train = neural_classes.Dataset(self._data_x_train, self._data_y_train)
		self._dataset_validate = neural_classes.Dataset(self._data_x_validate, self._data_y_validate)

		self._train_dataloader = torch.utils.data.DataLoader(self._dataset_train, batch_size=self._batch_size, shuffle=True)
		self._validate_dataloader = torch.utils.data.DataLoader(self._dataset_validate, batch_size=self._batch_size, shuffle=True)

		self._model = neural_classes.LSTMModel(input_size=self._input_size, hidden_layer_size=self._lstm_size, num_layers=self._num_lstm_layers, dropout=self._dropout)
		self._model = self._model.to(self._device)

		self._criterion = nn.MSELoss()
		self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._learning_rate, betas=(0.9, 0.98), eps=1e-9)
		self._scheduler = torch.optim.lr_scheduler.StepLR(self._optimizer, step_size=self._scheduler_step_size, gamma=0.1)

		for epoch in range(config["training"]["num_epoch"]):
			loss_train, lr_train = self._run_epoch(self._train_dataloader, is_training=True)
			loss_validation, lr_validation = self._run_epoch(self._validate_dataloader)
			self._scheduler.step()

			print('Epoch[{}/{}] | loss train:{:.6f}, test:{:.6f} | lr:{:.6f}'.format(epoch+1, config["training"]["num_epoch"], loss_train, loss_validation, lr_validation))

		self._train_dataloader = torch.utils.data.DataLoader(self._dataset_train, batch_size=self._batch_size, shuffle=False)
		self._validate_dataloader = torch.utils.data.DataLoader(self._dataset_validate, batch_size=self._batch_size, shuffle=False)

		self._model.eval()

		self._predicted_train = self._get_predicted(self._train_dataloader)
		self._predicted_validate = self._get_predicted(self._validate_dataloader)

		print(self.symbol_data.get_close_prices()[-10:])

		x = torch.tensor(self._data_x_unseen).float().to(self._device).unsqueeze(0).unsqueeze(2)
		self._prediction = self._model(x)
		self._prediction = self._prediction.cpu().detach().numpy()
		self._prediction_value = self.symbol_data.get_close_prices()[-1] * self._prediction + self.symbol_data.get_close_prices()[-1]
		print(self._prediction_value)

		self._data_x_unseen_2 = numpy.concatenate((self._data_x_unseen, self._prediction))
		x = torch.tensor(self._data_x_unseen_2).float().to(self._device).unsqueeze(0).unsqueeze(2)
		self._prediction2 = self._model(x)
		self._prediction2 = self._prediction2.cpu().detach().numpy()
		print(self._prediction2 * self._prediction_value + self._prediction_value)


		# self.plot_results()

	def plot_results(self):
		plot_range = 10
		dates = self.symbol_data.get_dates()[-plot_range+1:]

		to_plot_data_y_val_pred = numpy.zeros(plot_range)
		to_plot_data_y_test_pred = numpy.zeros(plot_range)

		to_plot_data_y_val_pred[:plot_range - 1] = self._normalizer.inverse_transform(self._data_y_validate)[-plot_range + 1:]
		to_plot_data_y_test_pred[plot_range - 1] = self._normalizer.inverse_transform(self._prediction)

		to_plot_data_y_val_pred = numpy.where(to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred)
		to_plot_data_y_test_pred = numpy.where(to_plot_data_y_test_pred == 0, None, to_plot_data_y_test_pred)

		dates.append("Tomorrow")

		# print(dates, to_plot_data_y_test_pred, to_plot_data_y_val_pred)

		plot = plotting_operations.Plot(self.symbol_data.get_plot_data(), [
			[dates, to_plot_data_y_val_pred, "Past Predicted Prices", config["plots"]["color_pred_val"]]
			# [dates, to_plot_data_y_test_pred, "Predicted Price for Next Day", config["plots"]["color_pred_test"]]
		], plot_range=plot_range)
		plot.show()

	def _get_plot_data(self, values, plot_range):
		to_plot_data = numpy.zeros(plot_range)
		to_plot_data = self._normalizer.inverse_transform(values)[plot_range + 1:]
		to_plot_data = numpy.where(to_plot_data == 0, None, to_plot_data)

		return to_plot_data


	def _get_predicted(self, dataloader):
		predicted = numpy.array([])

		for idx, (x, y) in enumerate(dataloader):
			x = x.to(self._device)
			out = self._model(x)
			out = out.cpu().detach().numpy()
			predicted = numpy.concatenate((predicted, out))

		return predicted

	def _prepare_data_x(self):
		n_row = self.close_prices_normalized.shape[0] - self._window_size + 1
		output = numpy.lib.stride_tricks.as_strided(self.close_prices_normalized, shape=(n_row, self._window_size), strides=(self.close_prices_normalized.strides[0], self.close_prices_normalized.strides[0]))
		return output[:-1], output[-1]

	def _prepare_data_y(self):
		return self.close_prices_normalized[self._window_size:]

	def _normalize(self):
		self._normalizer = neural_classes.Normalizer()
		return self._normalizer.fit_transform(self.symbol_data.get_close_prices())

	def _run_epoch(self, dataloader, is_training=False):
		epoch_loss = 0

		if is_training:
			self._model.train()
		else:
			self._model.eval()

		for idx, (x, y) in enumerate(dataloader):
			if is_training:
				self._optimizer.zero_grad()

			batchsize = x.shape[0]

			x = x.to(self._device)
			y = y.to(self._device)

			out = self._model(x)
			loss = self._criterion(out.contiguous(), y.contiguous())

			if is_training:
				loss.backward()
				self._optimizer.step()

			epoch_loss += (loss.detach().item() / batchsize)

		lr = self._scheduler.get_last_lr()[0]

		return epoch_loss, lr


class MainProcess:
	def __init__(self):
		self._db = stocks_db_operations.StocksDb.initialize_connection()

	def start_process(self):
		for currency_code in self._db.get_all_currency_codes():
			try:
				symbol_process = SymbolProcess(currency_code)
			except ValueError as e:
				print(e, currency_code)
				raise e

			break


if __name__ == "__main__":
	main_process = MainProcess()
	main_process.start_process()
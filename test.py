import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from neural_network import neural_data


class DataPreprocessing:
	def __init__(self, symbol):
		self.symbol = symbol
		self.symbol_data = neural_data.CurrencyData(symbol)

		self.pd_data = self.symbol_data.get_pd_data()
		self.next_days = self.symbol_data.get_next_days()

		self.matrice = self.pd_data.iloc[:, :].values
		self.variable = self.next_days.iloc[:, -1].values

		from sklearn.model_selection import train_test_split
		self._x_train, self._x_test, self._y_train, self._y_test = train_test_split(self.matrice, self.variable, test_size=0.2, shuffle=False)
		# self._x_train, self._x_test = self._get_standard_scale(self._x_train, self._x_test) # Doesn't seem to make a difference here...

		linear_regression_prediction = self.get_linear_regression_prediction(False)
		# print(linear_regression_prediction)
		# print(self.get_polynomial_regression_prediction(False))

		self.get_adjusted_r_squared(linear_regression_prediction)

	def _get_standard_scale(self, train, test):
		from sklearn.preprocessing import StandardScaler

		standard_scaler = StandardScaler()
		train = standard_scaler.fit_transform(train)
		test = standard_scaler.transform(test)

		return train, test

	def get_linear_regression_prediction(self, get_last_only=True):
		from sklearn.linear_model import LinearRegression

		regressor = LinearRegression()
		regressor.fit(self._x_train, self._y_train)

		if get_last_only:
			return regressor.predict(np.array([self._x_test[-1]]))[0]
		else:
			return regressor.predict(np.array(self._x_test))

	def get_polynomial_regression_prediction(self, get_last_only=True):
		from sklearn.linear_model import LinearRegression
		from sklearn.preprocessing import PolynomialFeatures

		polynomial_regressor = PolynomialFeatures(degree=3)
		x_polynomial = polynomial_regressor.fit_transform(self._x_train)

		regressor = LinearRegression()
		regressor.fit(x_polynomial, self._y_train)

		if get_last_only:
			return regressor.predict(np.array([polynomial_regressor.fit_transform(self._x_test)[-1]]))[0]
		else:
			return regressor.predict(np.array(polynomial_regressor.fit_transform(self._x_test)))


if __name__ == "__main__":
	data_preprocessing = DataPreprocessing("BTC")
	# print(data_preprocessing._x_train)
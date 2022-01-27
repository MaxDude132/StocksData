import datetime
import pandas as pd

from db_operations import stocks_db_operations
from db_operations import constants


class CurrencyData:
	def __init__(self, symbol):
		self.symbol = symbol
		self._db = stocks_db_operations.StocksDb.initialize_connection()

		self.all_data = self._get_symbol_data()

	def _get_symbol_data(self):
		return self._db.select(constants.TABLE_DIGITAL_CURRENCY_DAILY, where="currency_code = '{}'".format(self.symbol), order_by="date asc")

	def get_dates(self, start=None):
		return [data["date"] for data in self.all_data if start is None or datetime.datetime.strptime(start, "%Y-%m-%d").date() <= datetime.datetime.strptime(data["date"], "%Y-%m-%d").date()]

	def get_close_prices(self, start=None):
		return [float(data["close"]) for data in self.all_data if start is None or datetime.datetime.strptime(start, "%Y-%m-%d").date() <= datetime.datetime.strptime(data["date"], "%Y-%m-%d").date()]

	def get_pd_data(self):
		dataframe = pd.read_sql_query(self._db._build_select_query(constants.TABLE_DIGITAL_CURRENCY_DAILY, selection="open, high, low, volume, close", where="currency_code = '{}'".format(self.symbol), order_by="date asc"), self._db._connection)
		for series in dataframe:
			dataframe[series] = pd.to_numeric(dataframe[series])
		return dataframe

	def get_next_days(self):
		dataframe = pd.read_sql_query(self._db._build_select_query(constants.TABLE_DIGITAL_CURRENCY_DAILY, selection="close", where="currency_code = '{}'".format(self.symbol), order_by="date asc"), self._db._connection)
		close_series = pd.to_numeric(dataframe["close"])
		mean = close_series.mean()
		dataframe_2 = pd.DataFrame([pd.Series({"close": mean})])
		out_dataframe = pd.concat([dataframe, dataframe_2], ignore_index=True)
		return out_dataframe.drop(0)

	def get_plot_data(self, start=None, title="Daily close prices"):
		dates = self.get_dates(start)
		close_prices = self.get_close_prices(start)
		display_title = "{} for {}, from {} to {}".format(title, self.symbol, dates[0], dates[-1])
		return dates, close_prices, len(dates), display_title
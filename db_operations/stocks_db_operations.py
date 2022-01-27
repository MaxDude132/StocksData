import hashlib

from db_operations import constants
from db_operations import db_operations


class StocksDb(db_operations.OperationCtx):
	_instance = None

	def create_all_tables(self):
		self.create_table(constants.TABLE_DIGITAL_CURRENCIES, [
			"currency_code text UNIQUE",
			"currency_name text"
		])

		self.create_table(constants.TABLE_DIGITAL_CURRENCY_DAILY, [
			"currency_code text",
			"date text",
			"open text",
			"high text",
			"low text",
			"close text",
			"volume text",
			"market_cap text"
		])

		self.create_table(constants.TABLE_FILES, [
			"filename text UNIQUE",
			"md5 text"
		])

		self.create_table(constants.TABLE_INVALID_SYMBOLS, [
			"symbol text UNIQUE",
			"date_added text"
		])

	def set_table_digital_currencies(self):
		file_data = open(constants.DIGITAL_CURRENCIES_LIST_PATH_NAME, "r+").read()
		file_md5 = hashlib.md5(file_data.encode()).hexdigest()

		current_file_md5 = self.select(constants.TABLE_FILES, where="filename = '{}'".format(constants.DIGITAL_CURRENCIES_LIST))[0]

		if file_md5 == current_file_md5:
			# pass
			return

		lines = file_data.split('\n')[1:]
		formatted_lines = [line.split(',') for line in lines]

		for data_line in formatted_lines:
			if len(data_line) != 2 or len(self.select(constants.TABLE_DIGITAL_CURRENCIES, where="currency_code = '{}'".format(data_line[0]))) != 0:
				continue

	@staticmethod
	def initialize_connection():
		if not StocksDb._instance:
			StocksDb._instance = StocksDb(constants.DB_PATH_NAME)

		return StocksDb._instance

	def get_all_currency_codes(self):
		return [currency["currency_code"] for currency in self.select("t_digital_currency_daily", selection="distinct currency_code")]


if __name__ == "__main__":
	pass
import datetime
import requests

from db_operations import errors
from db_operations import constants
from db_operations import stocks_db_operations


class SetTable:
	def __init__(self, db):
		self._db = db
		self._request = None

	def _try_get_new_table_date(self):
		try:
			self._get_new_table_data()
		except (errors.RequestError, RecursionError) as e:
			pass
			# print(e)

	def _get_new_table_data(self):
		# print(self._request)
		request_response = requests.get(self._request)

		try:
			reponse_json = request_response.json()
		except requests.exceptions.JSONDecodeError:
			raise errors.RequestError("Could not get data for request {} with the following details: {}".format(self._request, reponse_json or ""))

		if 'Error Message' in reponse_json:
			raise errors.UnavailableSymbol("No data for request {} with the following details: {}".format(self._request, reponse_json))

		if not request_response or 'Meta Data' not in reponse_json or not request_response.content:
			raise errors.RequestError("Could not get data for request {} with the following details: {}".format(self._request, reponse_json or ""))

		data = request_response.json()

		self._handle_insertion(data)

	def _handle_insertion(self, ticker):
		self._insert_line(ticker)

	def _insert_line(self, ticker):
		pass

	def _timestamp_to_datetime(self, timestamp):
		formatted_timestamp = timestamp.split("+")[0].replace("T", "+").replace("-", "+").replace(":", "+").split("+")
		formatted_timestamp = [int(value) for value in formatted_timestamp]
		return datetime.datetime(*formatted_timestamp)


class SetTableDigitalCurrencyDaily(SetTable):
	def __init__(self, db):
		super(SetTableDigitalCurrencyDaily, self).__init__(db)
		self._request = constants.BASE_REQUEST_PATH.format(function="DIGITAL_CURRENCY_DAILY", symbol="{symbol}")
		self._prepare_all_requests()

	def _prepare_all_requests(self):
		initial_request = self._request
		currencies = self._db.select(constants.TABLE_DIGITAL_CURRENCIES)

		today = datetime.datetime.now().strftime("%Y-%m-%d")
		
		for currency in currencies:
			if len(self._db.select(constants.TABLE_DIGITAL_CURRENCY_DAILY, where="currency_code = '{}' and date = '{}'".format(currency["currency_code"], today))) != 0:
				continue

			if len(self._db.select(constants.TABLE_INVALID_SYMBOLS, where="symbol = '{}'".format(currency["currency_code"], today))) != 0:
				continue

			self._request = initial_request.format(symbol=currency["currency_code"])

			try:
				self._try_get_new_table_date()
			except errors.UnavailableSymbol as e:
				print("No data for ", currency["currency_code"])
				self._db.insert(constants.TABLE_INVALID_SYMBOLS, (currency["currency_code"], today))

	def _handle_insertion(self, currency):
		currency_code = currency["Meta Data"]["2. Digital Currency Code"]

		latest_date = list(currency["Time Series (Digital Currency Daily)"].keys())[0]
		if datetime.datetime.strptime(latest_date, "%Y-%m-%d").date() < datetime.date.today() - datetime.timedelta(days=1):
			print("Currency not in use anymore, removing from databse: {}".format(currency_code))
			self._db.delete(constants.TABLE_DIGITAL_CURRENCY_DAILY, where="currency_code = '{}'".format(currency_code))
			self._db.insert(constants.TABLE_INVALID_SYMBOLS, (currency_code, datetime.datetime.utcnow().strftime("%Y-%m-%d")))
			return

		for date, currency_day in currency["Time Series (Digital Currency Daily)"].items():
			currency_day["currency_code"] = currency_code
			currency_day["date"] = date

			if len(self._db.select(constants.TABLE_DIGITAL_CURRENCY_DAILY, where="currency_code = '{}' and date = '{}'".format(currency_code, date))) != 0:
				print("Updated data for", currency_code)
				return

			self._insert_line(currency_day)

		print("Added data for", currency_code)

	def _insert_line(self, currency):
		self._db.insert(constants.TABLE_DIGITAL_CURRENCY_DAILY, [
			currency["currency_code"],
			currency["date"],
			currency["1a. open (CAD)"],
			currency["2a. high (CAD)"],
			currency["3a. low (CAD)"],
			currency["4a. close (CAD)"],
			currency["5. volume"],
			currency["6. market cap (USD)"]
		])














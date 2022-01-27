import os
import pathlib


BASEPATH = pathlib.Path(__file__).parent.resolve()


DB_NAME = "stocks.db"
DB_PATH_NAME = os.path.join(BASEPATH, DB_NAME)


API_KEY = "BI1WOFPNXZXK7FPP"


TIME_BETWEEN_UPDATE_LOOPS = 60
TIME_BETWEEN_PLOT_LOOPS = 60 * 60 * 24

TIME_START_PLOT_LOOPS = "06:00:00"


BASE_REQUEST_PATH = "https://www.alphavantage.co/query?function={function}&symbol={symbol}&market=CAD&apikey=" + API_KEY


# All file dependencies
DIGITAL_CURRENCIES_LIST = "digital_currency_list.csv"
DIGITAL_CURRENCIES_LIST_PATH_NAME = os.path.join(BASEPATH, DIGITAL_CURRENCIES_LIST)


# All table names
TABLE_DIGITAL_CURRENCIES = "t_digital_currencies"
TABLE_DIGITAL_CURRENCY_DAILY = "t_digital_currency_daily"
TABLE_FILES = "t_files"
TABLE_INVALID_SYMBOLS = "t_invalid_symbols"
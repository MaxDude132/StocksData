	import time
import datetime
import multiprocessing

from db_operations import constants, stocks_db_operations, update_stocks_db
from neural_network import plotting_operations, neural_data


class CmdLog:
	def __init__(self):
		self._logging_thread = ""

	def _log(self, message):
		current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		current_utc_time = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
		print("{} :: UTC {} - {} - {}".format(current_time, current_utc_time, self._logging_thread, message))


class ProcessLoop(CmdLog):
	def __init__(self):
		self._db = stocks_db_operations.StocksDb.initialize_connection()
		self._main_db_process = MainDbProcess(self._db)
		self._main_plotting_process = MainPlottingProcess(self._db)

	def start_all_loops(self):
		loops_to_start = [self.start_update_loop, self.start_plotting_loop]

		processes = []

		for loop in loops_to_start:
			process = multiprocessing.Process(target=loop)
			processes.append(process)

		for process in processes:
			process.start()

		print("All loops have been launched.")

		for process in processes:
			process.join()

	def start_update_loop(self):
		while True:
			delta_time = self._update_loop()
			sleep_time = constants.TIME_BETWEEN_UPDATE_LOOPS - delta_time if constants.TIME_BETWEEN_UPDATE_LOOPS - delta_time > 0 else 0
			self._log("Update loop will sleep for {:0.2f} seconds, or {:0.2f} minutes.".format(sleep_time, sleep_time / 60))
			time.sleep(sleep_time)

	def _update_loop(self):
		start_time = time.time()

		self._logging_thread = "UPDATE_LOOP"
		self._log("__STARTING DB UPDATE__")

		print("DB_INFO - Lines of available data: ", len(self._db.select("t_digital_currency_daily")))
		print("DB_INFO - Data available for: ", len(self._db.select("t_digital_currency_daily", selection="distinct currency_code")))
		print("DB_INFO - Up to date data available for: ", len(self._db.select("t_digital_currency_daily", where="date = '{}'".format(datetime.datetime.now().strftime("%Y-%m-%d")))))
		print("DB_INFO - Data not available for: ", len(self._db.select("t_invalid_symbols")))

		self._log("Checking and updating dependencies...")
		self._main_db_process.set_dependencies()
		self._log("Dependencies updated!")

		self._log("Starting db update...")
		self._main_db_process.update_db()

		end_time = time.time()
		delta_time = end_time - start_time
		self._log("Db updated in {:0.2f} seconds".format(delta_time))
		return delta_time

	def start_plotting_loop(self):
		while True:
			delta_time = self._plotting_loop()
			sleep_time = constants.TIME_BETWEEN_PLOT_LOOPS - delta_time if constants.TIME_BETWEEN_PLOT_LOOPS - delta_time > 0 else 0
			self._log("Plotting loop will sleep for {:0.2f} minutes, or {:0.2f} hours.".format(sleep_time / 60, sleep_time / 60 / 60))
			time.sleep(sleep_time)

	def _plotting_loop(self):
		current_time = time.strftime("%H:%M:%S")

		while current_time != constants.TIME_START_PLOT_LOOPS:
			time.sleep(1)
			current_time = time.strftime("%H:%M:%S")

		start_time = time.time()

		self._logging_thread = "PLOTTING_LOOP"
		self._log("__STARTING PLOT GENERATION__")

		self._main_plotting_process.generate_all_plots("Plots")

		end_time = time.time()
		delta_time = end_time - start_time
		self._log("All plots have been generated in {:0.2f} seconds".format(delta_time))
		return delta_time


class MainDbProcess:
	def __init__(self, db):
		self._db = db

		# self._db.drop_table("t_invalid_symbols")
		# self._create_all_tables() # Use if you need to recreate the tables after losing data. Should be erased once in production.

	def update_db(self):
		update_stocks_db.SetTableDigitalCurrencyDaily(self._db)

	def set_dependencies(self):
		self._db.set_table_digital_currencies()

	def _create_all_tables(self):
		self._db.create_all_tables()


class MainPlottingProcess(CmdLog):
	def __init__(self, db):
		self._db = db
		self._logging_thread = "PLOTTING_LOOP"

	def generate_all_plots(self, out_dir):
		all_symbols = self._db.get_all_currency_codes()

		for symbol in all_symbols:
			currency_data = neural_data.CurrencyData(symbol)
			plot = plotting_operations.Plot(currency_data.get_plot_data())
			plot.save("{}/{}.png".format(out_dir, symbol))
			self._log("Plot generated for {}".format(symbol))


if __name__ == "__main__":
	process_loop = ProcessLoop()
	process_loop.start_all_loops()
import copy
import numpy
import matplotlib
import matplotlib.pyplot as pyplot

from neural_network.config import config


class Plot:
	def __init__(self, plot_data, additional_plots=()):
		self._dates, self._close_prices, self._length, self._display_title = plot_data
		self._additional_plots = additional_plots

		self._figure = pyplot.figure(figsize=(25, 12), dpi=80)
		self._figure.patch.set_facecolor((1.0, 1.0, 1.0))

		self._generate_plot()

	def _generate_plot(self):
		pyplot.plot(self._dates, self._close_prices, label="Actual prices", color=config["plots"]["color_actual"])

		for plot in self._additional_plots:
			self._add_plot(*plot)

		xticks = [self._dates[i] if ((i % config["plots"]["xticks_interval"] == 0 and (self._length - i) > config["plots"]["xticks_interval"]) or i == self._length - 1) else None for i in range(self._length)]
		x = numpy.arange(0, len(xticks))
		pyplot.xticks(x, xticks, rotation="75")
		pyplot.title(self._display_title)

		if self._additional_plots:
			pyplot.legend()
			
		pyplot.grid(b=None, which="major", axis="y", linestyle="--")

	def _add_plot(self, x, y, label, color):
		pyplot.plot(x, y, label=label, color=color)

	def show(self):
		pyplot.show()

	def save(self, filename):
		pyplot.savefig(filename, dpi=300)
		pyplot.close()
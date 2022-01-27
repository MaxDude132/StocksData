import sqlite3


class OperationCtx:
	def __init__(self, file):
		self._connection = sqlite3.connect(file)
		self._cursor = self._connection.cursor()

	def create_table(self, table_name, columns_list):
		query = "CREATE TABLE {} {}".format(table_name, self._get_values_separators(columns_list, values_decorator="{}"))
		query = query.format(*columns_list)
		return self._try_run_query(query)

	def drop_table(self, table_name):
		query = "DROP TABLE {}".format(table_name)
		return self._try_run_query(query)

	def insert(self, table_name, values):
		query = "INSERT INTO {} VALUES {}".format(table_name, self._get_values_separators(values))
		return self._try_run_query(query, values)

	def select(self, table_name, selection="*", where=None, order_by=None, limit=None):
		query = self._build_select_query(table_name, selection, where, order_by, limit)
		success = self._try_run_query(query)

		return self._return_select_as_dict_list()

	def delete(self, table_name, where=None):
		query = "DELETE FROM {}".format(table_name)
		if where:
			query += " WHERE {}".format(where)

		return self._try_run_query(query)

	def _return_select_as_dict_list(self):
		out_list = []

		try:
			headers = [header[0] for header in self._cursor.description]
		except TypeError:
			pass

		value = self._cursor.fetchone()

		while value:
			temp_dict = {}
			for i, header in enumerate(headers):
				temp_dict[header] = value[i]

			out_list.append(temp_dict)
			value = self._cursor.fetchone()

		return out_list


	def _build_select_query(self, table_name, selection="*", where=None, order_by=None, limit=None):
		query = "SELECT {} FROM {}".format(selection, table_name)

		if where:
			query += " WHERE {}".format(where)
		if order_by:
			query += " ORDER BY {}".format(order_by)
		if limit:
			query += " LIMIT {}".format(limit)

		return query

	def _try_run_query(self, query, values=None):
		try:
			if values:
				self._cursor.execute(query, values)
			else:
				self._cursor.execute(query)
			self._connection.commit()
			return 0
		except sqlite3.OperationalError as e:
			print(e)
			return -1
		except sqlite3.IntegrityError as e:
			print(query, values)
			raise e

	def _get_values_separators(self, columns_list, values_decorator="?"):
		values_separators = "{}, ".format(values_decorator) * len(columns_list)
		return "({})".format(values_separators[:-2])

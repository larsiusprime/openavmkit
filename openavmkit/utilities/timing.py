import time


class TimingData:
	_data = {}
	results = {}

	def __init__(self):
		self._data = {}
		self.results = {}

	def start(self, key):
		if key in self.results:
			self._data[key] = time.time() - self.results[key]
		else:
			self._data[key] = time.time()

	def stop(self, key):
		if key in self._data:
			result = time.time() - self._data[key]
			self.results[key] = result
			return result
		else:
			return -1

	def get(self, key):
		return self.results.get(key)


	def print(self):
		value = ""
		for key in self.results:

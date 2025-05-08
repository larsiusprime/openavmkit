import time


class TimingData:
	_data = {}
	results = {}
	_active_keys = set()

	def __init__(self):
		self._data = {}
		self.results = {}
		self._active_keys = set()

	def start(self, key):
		if key in self.results:
			self._data[key] = time.time() - self.results[key]
		else:
			self._data[key] = time.time()
		self._active_keys.add(key)

	def stop(self, key):
		if key in self._data:
			result = time.time() - self._data[key]
			self.results[key] = result
			if key in self._active_keys:
				self._active_keys.remove(key)
			return result
		else:
			if key in self._active_keys:
				self._active_keys.remove(key)
			return -1

	def get(self, key):
		return self.results.get(key)

	def is_running(self, key):
		return key in self._active_keys

	def print(self):
		value = ""
		for key in self.results:
			if value != "":
				value += "\n"
			value += (f"{key}: {self.results[key]:.2f} seconds")
		return value

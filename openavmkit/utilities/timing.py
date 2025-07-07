import time


class TimingData:
    """A simple timing class for quick & dirty performance profiling.

    This class holds multiple internal "stopwatches" which can each keep track of their own running time.
    Just instantiate a TimingData and call ``start()`` and ``stop()`` with a desired key.

    Attributes
    ----------
    results : dict[str:float]
        Raw timing results
    """
    _data = {}
    results = {}

    def __init__(self):
        self._data = {}
        self.results = {}

    def start(self, key):
        """Start a named stopwatch

        Parameters
        ----------
        key : str
            Name of the stopwatch to begin timing, or resume timing if it already exists.
        """
        if key in self.results:
            self._data[key] = time.time() - self.results[key]
        else:
            self._data[key] = time.time()

    def stop(self, key):
        """Stop a named stopwatch

        Parameters
        ----------
        key : str
            If this stopwatch has been started before, stops it if running and calculates how long it has run for.

        Returns
        -------
        float
            How long the stopwatch has been running for
        """
        if key in self._data:
            result = time.time() - self._data[key]
            self.results[key] = result
            return result
        else:
            return -1

    def get(self, key):
        """Get the running time of the indicated stopwatch

        Parameters
        ----------
        key : str
            The stopwatch to get timing data for

        Returns
        -------
        float
            How long the indicated stopwatch has run, if it exists
        """
        return self.results.get(key)

    def print(self):
        """Print the value of all stopwatches"""
        value = ""
        for key in self.results:
            if value != "":
                value += "\n"
            value += f"{key}: {self.results[key]:.2f} seconds"
        return value

# Usage

## Using the code modules

Here's how you can import and use the core modules directly in your own Python code.

For instance, here's a simple example that demonstrates how to calculate the Coefficient of Dispersion (COD) for a list of ratios:

```python
import openavmkit

ratios = [0.8, 0.9, 1.0, 1.1, 1.2]
cod = openavmkit.utilities.stats.calc_cod(ratios)
print(cod)
```

You can also specify the specific module you want to import:

```python
from openavmkit.utilities import stats

ratios = [0.8, 0.9, 1.0, 1.1, 1.2]
cod = stats.calc_cod(ratios)
```

Or even import specific functions directly:

```python
from openavmkit.utilities.stats import calc_cod

ratios = [0.8, 0.9, 1.0, 1.1, 1.2]
cod = calc_cod(ratios)
```
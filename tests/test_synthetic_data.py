from openavmkit.synthetic_data import generate_inflation_curve


def test_inflation_curve():
	print("")
	time_mult = generate_inflation_curve(
		start_year=2020,
		end_year=2024,
		annual_inflation_rate=0.05,
		annual_inflation_rate_stdev=0.01,
		seasonality_amplitude=0.20,
		monthly_noise=0.05,
		daily_noise=0.01
	)


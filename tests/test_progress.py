

def test_a_part_can_be_skipped_and_its_own_handlers_cant_swallow_it():
	from openavmkit import progress

	events = []
	skipped = {"distances"}

	def work():
		for i, feature in enumerate(["parks", "water"]):
			try:
				progress.report("distances", message=feature, sub=i / 2, part="distances")
			except Exception:  # a failed download is handled like this in the engine
				pass

	with progress.reporting(events.append, skip=lambda part: part in skipped):
		try:
			work()
			raise AssertionError("not skipped")
		except progress.Skipped as e:
			assert e.part == "distances"
		progress.report("elevation", sub=0.5, part="elevation")  # another part carries on
	assert [(e.step, e.sub, e.part) for e in events] == [("distances", 0.0, "distances"), ("elevation", 0.5, "elevation")]


def test_detail_says_what_the_current_step_is_doing():
	from openavmkit import progress
	from openavmkit.utilities.data import do_per_model_group
	import pandas as pd

	events = []
	df = pd.DataFrame({"key": ["a", "b"], "model_group": ["houses", "condos"], "x": [1, 2]})
	settings = {"modeling": {"model_groups": {"houses": {}, "condos": {}}}}
	with progress.reporting(events.append):
		progress.detail("before any step")  # nothing to add to: not reported
		progress.report("time_adjustment", 0.4, "Time adjustment")
		do_per_model_group(df, settings, lambda d, model_group: d, {})
	assert [(e.step, e.fraction, e.message, e.sub) for e in events] == [
		("time_adjustment", 0.4, "Time adjustment", None),
		("time_adjustment", 0.4, "Time adjustment: houses (1 of 2)", 0.0),
		("time_adjustment", 0.4, "Time adjustment: condos (2 of 2)", 0.5),
	]

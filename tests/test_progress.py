

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

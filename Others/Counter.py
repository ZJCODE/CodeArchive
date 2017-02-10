def Counter(List):
	count = dict({i: List.count(i) for i in List})
	return count

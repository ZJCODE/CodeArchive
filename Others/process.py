def extract_(s):
	s = s.strip()
	in_brackets = s[s.find('(')+1:s.find(')')]
	link_start = in_brackets.find('[')-6
	link_end= in_brackets.find(']')-1
	link = in_brackets[link_start:link_end]
	others = (in_brackets[:link_start]+in_brackets[link_end+4:]).strip().split(',')
	others.append(Links)
	return others


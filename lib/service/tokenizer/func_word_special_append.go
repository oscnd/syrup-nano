package tokenizer

import "slices"

func (r *Service) WordSpecialAppend(word string) {
	firstChar := rune(word[0])
	r.WordSpecialLookup[firstChar] = append(r.WordSpecialLookup[firstChar], word)
	slices.SortFunc(r.WordSpecialLookup[firstChar], func(a, b string) int {
		if len(a) != len(b) {
			return len(b) - len(a)
		}
		if a < b {
			return -1
		}
		if a > b {
			return 1
		}
		return 0
	})
}

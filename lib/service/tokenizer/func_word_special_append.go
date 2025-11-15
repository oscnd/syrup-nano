package tokenizer

import (
	"go.scnd.dev/open/syrup/nano/lib/type/tuple"
	"slices"
)

func (r *Service) WordSpecialAppend(word string) {
	firstChar := rune(word[0])
	specialWord := &tuple.SpecialWord{
		Text:  word,
		Words: []string{word},
	}
	r.WordSpecialLookup[firstChar] = append(r.WordSpecialLookup[firstChar], specialWord)
	slices.SortFunc(r.WordSpecialLookup[firstChar], func(a, b *tuple.SpecialWord) int {
		if len(a.Text) != len(b.Text) {
			return len(b.Text) - len(a.Text)
		}
		if a.Text < b.Text {
			return -1
		}
		if a.Text > b.Text {
			return 1
		}
		return 0
	})
}

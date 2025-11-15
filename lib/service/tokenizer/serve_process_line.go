package tokenizer

import (
	"strings"
	"unicode"

	"go.scnd.dev/open/syrup/nano/lib/type/enum"
	"go.scnd.dev/open/syrup/nano/lib/type/tuple"
	"go.scnd.dev/open/syrup/nano/lib/util"
)

func (r *Service) ProcessLine(line string) []*tuple.WordPair {
	var pairs []*tuple.WordPair
	var current []rune
	i := 0

	for i < len(line) {
		// check for word using WordLookupCheck utility (includes look-ahead functionality)
		specialWord := util.WordLookupCheck(line, i, r.WordLookup)
		if specialWord != nil {
			// case of accumulated characters before, add them as a word
			if len(current) > 0 {
				wordPair := r.ProcessWord(string(current))
				pairs = append(pairs, wordPair...)
				current = current[:0] // clear accumulated characters
			}

			// process each subword in the special word
			for _, subword := range specialWord.Words {
				wordPair := r.ProcessWord(subword)
				pairs = append(pairs, wordPair...)
			}
			i += len(specialWord.Text)
			continue
		}

		// check if character is uppercase (modifier)
		if unicode.IsUpper(rune(line[i])) {
			// case of accumulated characters before this modifier, add them as a word
			if len(current) > 0 {
				wordPair := r.ProcessWord(string(current))
				pairs = append(pairs, wordPair...)
				current = current[:0] // clear accumulated characters
			}

			// check for uppercase consecutiveness
			consecutiveUpper := false
			var j int
			for j = i + 1; j < len(line); j++ {
				// break on word special check
				if util.WordLookupCheck(line, j, r.WordLookup) != nil {
					consecutiveUpper = true
					break
				}

				// break on non-uppercase character
				if !unicode.IsUpper(rune(line[j])) {
					break
				}

				// last character is uppercase
				if j == len(line)-1 {
					consecutiveUpper = true
				}
			}

			if consecutiveUpper {
				pairs = append(pairs, &tuple.WordPair{
					Word:  string(enum.WordModifierNextUpper),
					Token: enum.WordModifier[enum.WordModifierNextUpper],
				})

				wordPair := r.ProcessWord(strings.ToLower(line[i:j]))
				pairs = append(pairs, wordPair...)
				i = j
				continue
			} else {
				pairs = append(pairs, &tuple.WordPair{
					Word:  string(enum.WordModifierNextCamel),
					Token: enum.WordModifier[enum.WordModifierNextCamel],
				})
				current = append(current, unicode.ToLower(rune(line[i])))
				i++
				continue
			}
		}

		// add character to current word
		current = append(current, unicode.ToLower(rune(line[i])))
		i++
	}

	// add any remaining characters as a word
	if len(current) > 0 {
		wordPair := r.ProcessWord(string(current))
		pairs = append(pairs, wordPair...)
	}

	return pairs
}

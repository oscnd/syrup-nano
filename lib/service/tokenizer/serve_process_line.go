package tokenizer

import (
	"fmt"
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
		// check for special word
		specialWord := util.WordSpecialCheck(line, i, r.WordSpecialLookup)
		if specialWord != "" {
			// case of accumulated characters before, add them as a word
			if len(current) > 0 {
				wordPair := r.ProcessWord(string(current))
				pairs = append(pairs, wordPair)
				current = current[:0] // clear accumulated characters
			}

			// add special word
			tokenNo, exists := r.WordSpecialToken[specialWord]
			if !exists {
				fmt.Printf("special word token not found for word: %s\n", specialWord)
			}
			pairs = append(pairs, &tuple.WordPair{
				Word:  specialWord,
				Token: tokenNo,
			})
			i += len(specialWord)
			goto nextIteration
		}

		// check if character is uppercase (modifier)
		if unicode.IsUpper(rune(line[i])) {
			// case of accumulated characters before this modifier, add them as a word
			if len(current) > 0 {
				wordPair := r.ProcessWord(string(current))
				pairs = append(pairs, wordPair)
				current = current[:0] // clear accumulated characters
			}

			// check for uppercase consecutiveness
			consecutiveUpper := false
			var j int
			for j = i; j < len(line); j++ {
				// break on special word check
				if util.WordSpecialCheck(line, j, r.WordSpecialLookup) != "" {
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

				wordPair := r.ProcessWord(line[i:j])
				pairs = append(pairs, wordPair)
				i = j
				goto nextIteration
			} else {
				pairs = append(pairs, &tuple.WordPair{
					Word:  string(enum.WordModifierNextCamel),
					Token: enum.WordModifier[enum.WordModifierNextCamel],
				})
			}
		}

		// add character to current word
		current = append(current, unicode.ToLower(rune(line[i])))
		i++

	nextIteration:
	}

	// add any remaining characters as a word
	if len(current) > 0 {
		wordPair := r.ProcessWord(string(current))
		pairs = append(pairs, wordPair)
	}

	return pairs
}

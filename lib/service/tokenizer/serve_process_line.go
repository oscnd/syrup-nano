package tokenizer

import (
	"unicode"

	"go.scnd.dev/open/syrup/nano/lib/type/tuple"
	"go.scnd.dev/open/syrup/nano/lib/util"
)

func (r *Service) ProcessLine(line string) []*tuple.WordPair {
	var pairs []*tuple.WordPair
	var current []rune
	i := 0

	for i < len(line) {
		char := rune(line[i])

		// check for word special lookup boundary
		if util.WordLookupCheck(line, i, r.WordSpecialLookup) != nil {
			// flush accumulated characters before this boundary
			if len(current) > 0 {
				wordPair := r.ProcessWord(string(current))
				pairs = append(pairs, wordPair...)
				current = current[:0]
			}
		}

		// check if character is uppercase (modifier) - check this FIRST
		if unicode.IsUpper(char) {
			// flush accumulated characters before this modifier
			if len(current) > 0 {
				wordPair := r.ProcessWord(string(current))
				pairs = append(pairs, wordPair...)
				current = current[:0]
			}

			// handle uppercase sequences
			modifier, consumed, isCamelCase := r.ProcessUppercaseSequence(line, i)
			pairs = append(pairs, modifier...)

			if isCamelCase {
				// for camel case, add the lowercase character to accumulator and let it be processed normally
				current = append(current, unicode.ToLower(char))
				i += consumed
			} else {
				// for consecutive uppercase, skip all consumed characters
				i += consumed
			}
			continue
		}

		// * check for word lookup only if accumulator is empty (to preserve words after camel case)
		if len(current) == 0 {
			if wordPairs, consumed := r.ProcessCharacterLookup(line, i); consumed > 0 {
				pairs = append(pairs, wordPairs...)
				i += consumed
				continue
			}
		}

		// accumulate lowercase character
		current = append(current, unicode.ToLower(char))
		i++
	}

	// flush any remaining accumulated characters
	if len(current) > 0 {
		wordPair := r.ProcessWord(string(current))
		pairs = append(pairs, wordPair...)
	}

	return pairs
}

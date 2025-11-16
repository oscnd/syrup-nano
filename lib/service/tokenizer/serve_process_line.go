package tokenizer

import (
	"unicode"

	"go.scnd.dev/open/syrup/nano/lib/type/tuple"
)

func (r *Service) ProcessLine(line string) []*tuple.WordPair {
	var pairs []*tuple.WordPair
	var current []rune
	i := 0

	for i < len(line) {
		char := rune(line[i])

		// check if character is uppercase (modifier) - check this FIRST
		if unicode.IsUpper(char) {
			// flush accumulated characters before this modifier
			if len(current) > 0 {
				wordPair := r.ProcessWord(string(current))
				pairs = append(pairs, wordPair...)
				current = current[:0]
			}

			// handle uppercase sequences
			modifier, consumed, _ := r.ProcessUppercaseSequence(line, i)
			pairs = append(pairs, modifier...)

			// skip consumed characters (camel case consumes 1, consecutive uppercase consumes more)
			i += consumed
			continue
		}

		// * check for word lookup
		if wordPairs, consumed := r.ProcessCharacterLookup(line, i); consumed > 0 {
			pairs = append(pairs, wordPairs...)
			i += consumed
			continue
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

package util

import (
	"go.scnd.dev/open/syrup/nano/lib/type/tuple"
)

func WordLookupCheck(line string, startIndex int, lookup map[rune][]*tuple.SpecialWord) *tuple.SpecialWord {
	if startIndex >= len(line) {
		return nil
	}

	currentChar := rune(line[startIndex])
	possibleWords, exists := lookup[currentChar]
	if !exists {
		return nil
	}

	// check each possible word that starts with this character
	for _, possibleWord := range possibleWords {
		endIndex := startIndex + len(possibleWord.Text)

		// limit line length check
		if endIndex > len(line) {
			continue
		}

		// extract the substring from current position to potential special length
		candidate := line[startIndex:endIndex]

		// if it matches, return the special word
		if candidate == possibleWord.Text {
			return possibleWord
		}
	}

	// no match found
	return nil
}

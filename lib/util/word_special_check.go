package util

import "go.scnd.dev/open/syrup/nano/lib/type/tuple"

func WordSpecialCheck(line string, startIndex int, lookup map[rune][]*tuple.SpecialWord) *tuple.SpecialWord {
	if startIndex >= len(line) {
		return nil
	}

	currentChar := rune(line[startIndex])
	possibleSpecials, exists := lookup[currentChar]
	if !exists {
		return nil
	}

	// check each possible special word that starts with this character
	for _, possibleSpecial := range possibleSpecials {
		endIndex := startIndex + len(possibleSpecial.Text)

		// make sure we don't go beyond the line length
		if endIndex > len(line) {
			continue
		}

		// extract the substring from current position to potential special length
		candidate := line[startIndex:endIndex]

		// if it matches, return the special word
		if candidate == possibleSpecial.Text {
			return possibleSpecial
		}
	}

	// no match found
	return nil
}

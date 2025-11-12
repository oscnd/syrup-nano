package util

func WordSpecialCheck(line string, startIndex int, lookup map[rune][]string) string {
	if startIndex >= len(line) {
		return ""
	}

	currentChar := rune(line[startIndex])
	possibleWords, exists := lookup[currentChar]
	if !exists {
		return ""
	}

	// check each possible word that starts with this character
	for _, possibleWord := range possibleWords {
		endIndex := startIndex + len(possibleWord)

		// make sure we don't go beyond the line length
		if endIndex > len(line) {
			continue
		}

		// extract the substring from current position to potential word length
		candidate := line[startIndex:endIndex]

		// if it matches, return the special word
		if candidate == possibleWord {
			return possibleWord
		}
	}

	// no match found
	return ""
}

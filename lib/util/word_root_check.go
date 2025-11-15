package util

import "go.scnd.dev/open/syrup/nano/lib/type/tuple"

func WordRootCheck(line string, startIndex int, lookup map[rune][]*tuple.CompoundWord) *tuple.CompoundWord {
	if startIndex >= len(line) {
		return nil
	}

	currentChar := rune(line[startIndex])
	possibleCompounds, exists := lookup[currentChar]
	if !exists {
		return nil
	}

	// check each possible compound word that starts with this character
	for _, possibleCompound := range possibleCompounds {
		endIndex := startIndex + len(possibleCompound.Compound)

		// make sure we don't go beyond the line length
		if endIndex > len(line) {
			continue
		}

		// extract the substring from current position to potential compound length
		candidate := line[startIndex:endIndex]

		// if it matches, return the compound word
		if candidate == possibleCompound.Compound {
			return possibleCompound
		}
	}

	// no match found
	return nil
}
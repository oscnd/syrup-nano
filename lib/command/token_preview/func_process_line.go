package main

import (
	"unicode"

	"go.scnd.dev/open/syrup/nano/lib/common/pogreb"
	"go.scnd.dev/open/syrup/nano/lib/type/enum"
)

func ProcessLine(pogreb *pogreb.Pogreb, line string) []string {
	var tokens []string
	var current []rune
	i := 0

	for i < len(line) {
		char := rune(line[i])

		// check for special word (similar logic from token_construct)
		if possibleWords, exists := wordSpecialLookup[string(char)]; exists {
			for _, possibleWord := range possibleWords {
				if i+len(possibleWord) > len(line) {
					continue
				}

				currentWord := make([]rune, 0)
				for j := 0; j < len(possibleWord); j++ {
					currentWord = append(currentWord, rune(line[i+j]))
				}

				if string(currentWord) == possibleWord {
					// add accumulated characters as a word
					if len(current) > 0 {
						token := ProcessWord(pogreb, string(current))
						tokens = append(tokens, token)
						current = current[:0] // clear accumulated characters
					}

					// add special word
					token := ProcessWord(pogreb, possibleWord)
					tokens = append(tokens, token)
					i += len(possibleWord)
					goto nextIteration
				}
			}
		}

		// check if character is uppercase (modifier)
		if unicode.IsUpper(char) {
			// add accumulated characters as a word
			if len(current) > 0 {
				token := ProcessWord(pogreb, string(current))
				tokens = append(tokens, token)
				current = current[:0] // clear accumulated characters
			}

			// add the modifier as a separate token
			tokens = append(tokens, enum.WordModifier[enum.WordModifierNextCamel])

			// add modifier
			char = unicode.ToLower(char)
		}

		// add character to current word
		current = append(current, char)
		i++

	nextIteration:
	}

	// add any remaining characters as a word
	if len(current) > 0 {
		token := ProcessWord(pogreb, string(current))
		tokens = append(tokens, token)
	}

	return tokens
}

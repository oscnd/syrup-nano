package main

import (
	"fmt"
	"unicode"

	"go.scnd.dev/open/syrup/nano/lib/common/pogreb"
	"go.scnd.dev/open/syrup/nano/lib/type/enum"
	"go.scnd.dev/open/syrup/nano/lib/type/tuple"
)

func ProcessLine(pogreb *pogreb.Pogreb, line string) []tuple.WordPair {
	var pairs []tuple.WordPair
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
					// case of accumulated characters before this special word, add them as a word
					if len(current) > 0 {
						wordPair := ProcessWord(pogreb, string(current))
						pairs = append(pairs, wordPair)
						current = current[:0] // clear accumulated characters
					}

					// add special word
					tokenNo, exists := wordSpecialToken[possibleWord]
					if !exists {
						fmt.Printf("special word token not found for word: %s\n", possibleWord)
					}
					pairs = append(pairs, tuple.WordPair{
						Word:  "#" + possibleWord,
						Token: tokenNo,
					})
					i += len(possibleWord)
					goto nextIteration
				}
			}
		}

		// check if character is uppercase (modifier)
		if unicode.IsUpper(char) {
			// case of accumulated characters before this modifier, add them as a word
			if len(current) > 0 {
				wordPair := ProcessWord(pogreb, string(current))
				pairs = append(pairs, wordPair)
				current = current[:0] // clear accumulated characters
			}

			// add modifier token
			modifierToken := enum.WordModifier[enum.WordModifierNextCamel]
			pairs = append(pairs, tuple.WordPair{
				Word:  string(enum.WordModifierNextCamel),
				Token: modifierToken,
			})
		}

		// add character to current word
		current = append(current, unicode.ToLower(rune(line[i])))
		i++

	nextIteration:
	}

	// add any remaining characters as a word
	if len(current) > 0 {
		wordPair := ProcessWord(pogreb, string(current))
		pairs = append(pairs, wordPair)
	}

	return pairs
}

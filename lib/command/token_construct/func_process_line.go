package main

import (
	"fmt"
	"unicode"

	"go.scnd.dev/open/syrup/nano/lib/common/pogreb"
	"go.scnd.dev/open/syrup/nano/lib/type/enum"
)

func ProcessLine(pogreb *pogreb.Pogreb, line string) []any {
	var values []any
	var current []rune
	i := 0

	for i < len(line) {
		char := rune(line[i])

		// check for special word
		if possibleWords, exists := wordSpecialLookup[string(char)]; exists {
			// look ahead for possible special tokens matching
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
						values = append(values, string(current))
						current = current[:0] // clear accumulated characters
					}

					// found a matching special token
					value, err := pogreb.WordMapper.Get([]byte(possibleWord))
					if err != nil || value == nil {
						fmt.Printf("error retrieving special token %s: %v", possibleWord, err)
					}
					tokenNo, _ := MapperPayloadExtract(value)
					values = append(values, tokenNo)
					i += len(possibleWord)
					goto nextIteration
				}
			}
		}

		// check if character is uppercase
		if unicode.IsUpper(rune(line[i])) {
			// case of accumulated characters before this modifier, add them as a word
			if len(current) > 0 {
				values = append(values, string(current))
				current = current[:0] // clear accumulated characters
			}

			// add modifier
			values = append(values, enum.WordModifier[enum.WordModifierNextCamel])

			// TODO: handle next upper
		}

		// add character to current word
		current = append(current, unicode.ToLower(rune(line[i])))
		i++

	nextIteration:
	}

	// add any remaining characters as a word
	if len(current) > 0 {
		values = append(values, string(current))
	}

	return values
}

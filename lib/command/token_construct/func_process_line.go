package main

import (
	"fmt"
	"unicode"

	"go.scnd.dev/open/syrup/nano/lib/common/pogreb"
)

func ProcessLine(pogreb *pogreb.Pogreb, line string) []any {
	var values []any
	var current []rune
	i := 0

	for i < len(line) {
		char := rune(line[i])

		// check for special word
		if _, isSpecial := wordSpecialMapper[string(char)]; isSpecial {
			// case of accumulated characters before this special word, add them as a word
			if len(current) > 0 {
				values = append(values, string(current))
				current = current[:0] // clear accumulated characters
			}

			// look ahead for multi-character special tokens
			if possibleWords, exists := wordSpecialLookup[string(char)]; exists {
				for _, possibleWord := range possibleWords {
					if i+len(possibleWord) > len(line) {
						continue
					}

					currentWord := []rune{char}
					for j := 1; j < len(possibleWord); j++ {
						currentWord = append(currentWord, rune(line[i+j]))
					}

					if string(currentWord) == possibleWord {
						// found a multi-character special token
						value, err := pogreb.WordMapper.Get([]byte(possibleWord))
						if err != nil {
							fmt.Printf("error retrieving special token %s: %v", possibleWord, err)
						}
						if value != nil {
							tokenNo, _ := MapperPayloadExtract(value)
							values = append(values, tokenNo)
							i += len(possibleWord)
							goto nextIteration
						}
					}
				}
			}

			// * single character special token
			if value, err := pogreb.WordMapper.Get([]byte{byte(char)}); err == nil && value != nil {
				tokenNo, _ := MapperPayloadExtract(value)
				values = append(values, tokenNo)
			}
			i++
		} else {
			// check if character is uppercase
			if unicode.IsUpper(rune(line[i])) {
				// case of accumulated characters before this modifier, add them as a word
				if len(current) > 0 {
					values = append(values, string(current))
					current = current[:0] // clear accumulated characters
				}

				// add modifier
				values = append(values, wordModifier["#nextUpper#"])
			}

			// add character to current word
			current = append(current, unicode.ToLower(rune(line[i])))
			i++
		}
	nextIteration:
	}

	// add any remaining characters as a word
	if len(current) > 0 {
		values = append(values, string(current))
	}

	return values
}

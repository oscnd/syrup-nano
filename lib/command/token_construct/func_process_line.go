package main

import (
	"fmt"
	"strings"
	"unicode"

	"go.scnd.dev/open/syrup/nano/lib/common/pogreb"
	"go.scnd.dev/open/syrup/nano/lib/type/enum"
	"go.scnd.dev/open/syrup/nano/lib/util"
)

func ProcessLine(pogreb *pogreb.Pogreb, line string) []any {
	var values []any
	var current []rune
	i := 0

	for i < len(line) {
		// check for special word using WordSpecialCheck utility
		specialWord := util.WordSpecialCheck(line, i, WordSpecialLookup)
		if specialWord != "" {
			// case of accumulated characters before, add them as a word
			if len(current) > 0 {
				values = append(values, string(current))
				current = current[:0] // clear accumulated characters
			}

			// found a matching special token
			value, err := pogreb.WordMapper.Get([]byte(specialWord))
			if err != nil || value == nil {
				fmt.Printf("error retrieving special token %s: %v\n", specialWord, err)
			}
			_, tokenNo, _ := util.MapperPayloadExtract(value)
			values = append(values, tokenNo)
			i += len(specialWord)
			goto nextIteration
		}

		// check if character is uppercase
		if unicode.IsUpper(rune(line[i])) {
			// case of accumulated characters before this modifier, add them as a word
			if len(current) > 0 {
				values = append(values, string(current))
				current = current[:0] // clear accumulated characters
			}

			// check for uppercase consecutiveness
			consecutiveUpper := false
			var j int
			for j = i; j < len(line); j++ {
				// break on special word check
				if util.WordSpecialCheck(line, j, WordSpecialLookup) != "" {
					consecutiveUpper = true
					break
				}

				// break on non-uppercase character
				if !unicode.IsUpper(rune(line[j])) {
					break
				}

				// last character is uppercase
				if j == len(line)-1 {
					consecutiveUpper = true
				}
			}

			if consecutiveUpper {
				values = append(values, enum.WordModifier[enum.WordModifierNextUpper])
				values = append(values, strings.ToLower(line[i:j]))
				i = j
				goto nextIteration
			} else {
				values = append(values, enum.WordModifier[enum.WordModifierNextCamel])
			}
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

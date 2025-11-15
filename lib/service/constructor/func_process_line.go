package constructor

import (
	"fmt"
	"strings"
	"unicode"

	"go.scnd.dev/open/syrup/nano/lib/type/enum"
	"go.scnd.dev/open/syrup/nano/lib/util"
)

func (r *Service) ProcessLine(line string) []any {
	var values []any
	var current []rune
	i := 0

	for i < len(line) {
		// check for special word using WordSpecialCheck utility
		specialWord := util.WordSpecialCheck(line, i, r.WordSpecialLookup)
		if specialWord != nil {
			// case of accumulated characters before, add them as a word
			if len(current) > 0 {
				values = append(values, string(current))
				current = current[:0] // clear accumulated characters
			}

			// process each subword in the special word
			for _, subword := range specialWord.Words {
				value, err := r.pogreb.WordMapper.Get([]byte(subword))
				if err != nil || value == nil {
					fmt.Printf("error retrieving subword token %s: %v\n", subword, err)
				}
				_, tokenNo, _ := util.MapperPayloadExtract(value)
				values = append(values, tokenNo)
			}
			i += len(specialWord.Text)
			continue
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
				// break on word special check
				if util.WordSpecialCheck(line, j, r.WordSpecialLookup) != nil {
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
				continue
			} else {
				values = append(values, enum.WordModifier[enum.WordModifierNextCamel])
			}
		}

		// add character to current word
		current = append(current, unicode.ToLower(rune(line[i])))
		i++
	}

	// add any remaining characters as a word
	if len(current) > 0 {
		values = append(values, string(current))
	}

	return values
}

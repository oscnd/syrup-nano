package tokenizer

import (
	"strings"
	"unicode"

	"go.scnd.dev/open/syrup/nano/lib/type/enum"
	"go.scnd.dev/open/syrup/nano/lib/type/tuple"
	"go.scnd.dev/open/syrup/nano/lib/util"
)

func (r *Service) ProcessUppercaseSequence(line string, startIdx int) ([]*tuple.WordPair, int, bool) {
	var pairs []*tuple.WordPair

	// check for consecutive uppercase letters
	consecutiveUpper := false
	j := startIdx + 1

	for j < len(line) {
		// break on special word check
		if util.WordLookupCheck(line, j, r.WordLookup) != nil {
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
			j++
			break
		}

		j++
	}

	if consecutiveUpper {
		// multiple consecutive uppercase letters
		pairs = append(pairs, &tuple.WordPair{
			Word:  string(enum.WordModifierNextUpper),
			Token: enum.WordModifier[enum.WordModifierNextUpper],
		})

		// process the uppercase sequence as lowercase
		wordPair := r.ProcessWord(strings.ToLower(line[startIdx:j]))
		pairs = append(pairs, wordPair...)
		return pairs, j - startIdx, false
	}

	// single uppercase letter (camel case)
	pairs = append(pairs, &tuple.WordPair{
		Word:  string(enum.WordModifierNextCamel),
		Token: enum.WordModifier[enum.WordModifierNextCamel],
	})

	// also return the lowercase character as a wordpair
	lowerChar := strings.ToLower(string(line[startIdx]))
	pairs = append(pairs, &tuple.WordPair{
		Word:  lowerChar,
		Token: 0,
	})

	// return modifier and character, consume 1 character
	return pairs, 1, true
}

package constructor

import (
	"fmt"

	"go.scnd.dev/open/syrup/nano/lib/type/enum"
	"go.scnd.dev/open/syrup/nano/lib/util"
)

func (r *Service) ProcessWord(word string) {
	// * generalize word
	special := false
	if possibleSpecialWords, ok := r.WordSpecialLookup[rune(word[0])]; ok {
		for _, possibleSpecial := range possibleSpecialWords {
			if word == possibleSpecial.Text {
				special = true
			}
		}
	}

	if !special {
		var modifier enum.WordSuffixType
		word, modifier = r.ProcessWordGeneralize(word)
		if modifier != "" {
			if false {
				fmt.Printf("found modifier for word %s: %s\n", word, modifier)
			}
		}
	}

	// * get word from pogreb
	value, err := r.pogreb.WordMapper.Get([]byte(word))
	if err != nil {
		fmt.Printf("error getting word %s: %v\n", word, err)
		return
	}
	if value == nil {
		// * case generalize word
		for _, suffix := range enum.WordSuffix {
			if suffix.Check(word) {
				continue
			}

			// Try the full word with this suffix
			fullWord := suffix.FullWord(word)
			if fullValue, fullErr := r.pogreb.WordMapper.Get([]byte(fullWord)); fullErr == nil && fullValue != nil {
				// extract token no
				special, tokenNo, _ := util.MapperPayloadExtract(fullValue)
				if special {
					continue
				}

				// remove from word mapper
				if err := r.pogreb.WordMapper.Delete([]byte(fullWord)); err != nil {
					fmt.Printf("error removing suffixed word %s: %v\n", fullWord, err)
				} else {
					fmt.Printf("removed suffixed word %s (token no: %d) for base word %s\n", fullWord, tokenNo, word)
				}

				// remove from token mapper
				if delErr := r.pogreb.TokenMapper.Delete(util.Uint64ToBytes(tokenNo)); delErr != nil {
					fmt.Printf("error removing token %d for word %s: %v\n", tokenNo, fullWord, delErr)
				}
			}
		}

		r.no++
		tokenNo := r.no
		count := uint64(1)

		if err := r.pogreb.WordMapper.Put([]byte(word), util.MapperPayloadBuild(special, tokenNo, count)); err != nil {
			fmt.Printf("error inserting word %s: %v\n", word, err)
			return
		}

		if err := r.pogreb.TokenMapper.Put(util.Uint64ToBytes(tokenNo), []byte(word)); err != nil {
			fmt.Printf("error inserting word %s: %v\n", word, err)
			return
		}

		fmt.Printf("new word added: %s (token no: %d)\n", word, tokenNo)
		return
	}

	// case word exists
	special, tokenNo, count := util.MapperPayloadExtract(value)
	count++
	if err := r.pogreb.WordMapper.Put([]byte(word), util.MapperPayloadBuild(special, tokenNo, count)); err != nil {
		fmt.Printf("error updating word %s: %v\n", word, err)
		return
	}
}

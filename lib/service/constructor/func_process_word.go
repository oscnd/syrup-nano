package constructor

import (
	"fmt"
	"time"

	"github.com/bsthun/gut"
	"go.scnd.dev/open/syrup/nano/lib/type/enum"
	"go.scnd.dev/open/syrup/nano/lib/type/tuple"
	"go.scnd.dev/open/syrup/nano/lib/util"
)

func (r *Service) ProcessWord(word string) {
	// * check special word
	var specialWord *tuple.SpecialWord
	if possibleSpecialWords, ok := r.WordSpecialLookup[rune(word[0])]; ok {
		for _, possibleSpecialWord := range possibleSpecialWords {
			if word == possibleSpecialWord.Text {
				if len(possibleSpecialWord.Words) > 1 {
					for _, w := range possibleSpecialWord.Words {
						r.ProcessWord(w)
					}
					return
				}
				specialWord = possibleSpecialWord
			}
		}
	}

	// * generalize word
	if specialWord == nil {
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
	if value != nil {
		// * case of word exists
		special, tokenNo, count := util.MapperPayloadExtract(value)
		count++
		if err := r.pogreb.WordMapper.Put([]byte(word), util.MapperPayloadBuild(special, tokenNo, count)); err != nil {
			fmt.Printf("error updating word %s: %v\n", word, err)
		}
		return
	}

	// * try to generalize word
	for _, suffix := range enum.WordSuffix {
		// try the full word with this suffix
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
				if false {
					fmt.Printf("removed suffixed word %s (token no: %d) for base word %s\n", fullWord, tokenNo, word)
				}
			}

			// remove from token mapper
			if err := r.pogreb.TokenMapper.Delete(util.Uint64ToBytes(tokenNo)); err != nil {
				fmt.Printf("error removing token %d for word %s: %v\n", tokenNo, fullWord, err)
			}
		}
	}

	r.No++
	tokenNo := r.No

	if specialWord != nil {
		fmt.Printf("adding special word %s as token no: %d\n", word, tokenNo)
	}

	if err := r.pogreb.WordMapper.Put([]byte(word), util.MapperPayloadBuild(specialWord != nil, tokenNo, uint64(1))); err != nil {
		fmt.Printf("error inserting word %s: %v\n", word, err)
		return
	}

	if err := r.pogreb.TokenMapper.Put(util.Uint64ToBytes(tokenNo), []byte(word)); err != nil {
		fmt.Printf("error inserting word %s: %v\n", word, err)
		return
	}

	if time.Now().After(r.LastLogged.Add(10 * time.Second)) {
		gut.Debug(fmt.Sprintf("added to token no: %d", tokenNo))
		r.LastLogged = gut.Ptr(time.Now())
	}
}

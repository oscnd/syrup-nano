package main

import (
	"fmt"
	"strings"

	"go.scnd.dev/open/syrup/nano/lib/common/pogreb"
)

func ProcessWordGeneralize(pogreb *pogreb.Pogreb, word string) ([]byte, WordModifierType) {
	if strings.HasSuffix(word, "ies") {
		if value, err := pogreb.WordMapper.Get([]byte(string(word[:len(word)-3]) + "y")); err != nil {
			fmt.Printf("progreb error on generalize -ies %s: %v\n", word, err)
		} else if value != nil {
			return value, WordModifierNextPlural
		}
	}

	if strings.HasSuffix(word, "es") {
		if value, err := pogreb.WordMapper.Get([]byte(word[:len(word)-2])); err != nil {
			fmt.Printf("progreb error on generalize -es %s: %v\n", word, err)
		} else if value != nil {
			return value, WordModifierNextPlural
		}
	}

	if strings.HasSuffix(word, "s") {
		if value, err := pogreb.WordMapper.Get([]byte(word[:len(word)-1])); err != nil {
			fmt.Printf("progreb error on generalize -s %s: %v\n", word, err)
		} else if value != nil {
			return value, WordModifierNextPlural
		}
	}

	if strings.HasSuffix(word, "ed") {
		if value, err := pogreb.WordMapper.Get([]byte(word[:len(word)-2])); err != nil {
			fmt.Printf("progreb error on generalize -ed %s: %v\n", word, err)
		} else if value != nil {
			return value, WordModifierNextEd
		}
	}

	if strings.HasSuffix(word, "er") {
		if value, err := pogreb.WordMapper.Get([]byte(word[:len(word)-2])); err != nil {
			fmt.Printf("progreb error on generalize -er %s: %v\n", word, err)
		} else if value != nil {
			return value, WordModifierNextEr
		}
	}

	if strings.HasSuffix(word, "ing") {
		if value, err := pogreb.WordMapper.Get([]byte(word[:len(word)-3])); err != nil {
			fmt.Printf("progreb error on generalize -ing %s: %v\n", word, err)
		} else if value != nil {
			return value, WordModifierNextIng
		}
	}

	if strings.HasSuffix(word, "ly") {
		if value, err := pogreb.WordMapper.Get([]byte(word[:len(word)-2])); err != nil {
			fmt.Printf("progreb error on generalize -ly %s: %v\n", word, err)
		} else if value != nil {
			return value, WordModifierNextLy
		}
	}

	return nil, ""
}

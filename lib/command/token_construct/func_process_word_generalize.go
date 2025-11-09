package main

import (
	"go.scnd.dev/open/syrup/nano/lib/common/pogreb"
	"go.scnd.dev/open/syrup/nano/lib/type/enum"
)

func ProcessWordGeneralize(pogreb *pogreb.Pogreb, word string) (string, enum.WordSuffixType) {
	if len(word) < 4 {
		return word, ""
	}

	// Iterate through suffix mappings to find matches
	for suffixKey, suffixBlock := range enum.WordSuffix {
		if suffixBlock.Check(word) {
			return suffixBlock.BaseWord(word), suffixKey
		}
	}

	return word, ""
}

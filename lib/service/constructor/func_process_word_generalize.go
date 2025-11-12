package constructor

import (
	"go.scnd.dev/open/syrup/nano/lib/type/enum"
)

func (r *Service) ProcessWordGeneralize(word string) (string, enum.WordSuffixType) {
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

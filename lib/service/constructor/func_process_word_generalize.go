package constructor

import (
	"go.scnd.dev/open/syrup/nano/lib/type/enum"
)

func (r *Service) ProcessWordGeneralize(word string) (string, enum.WordSuffixType) {
	// Iterate through suffix mappings to find matches
	for {
		block := new(enum.WordSuffixBlock)
		for _, suffixBlock := range enum.WordSuffix {
			if suffixBlock.Check(word) {
				block = suffixBlock
				break
			}
		}
		if block.Suffix == "" {
			return word, ""
		}
		word = block.BaseWord(word)
	}

	return word, ""
}

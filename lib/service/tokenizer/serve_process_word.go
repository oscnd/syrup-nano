package tokenizer

import (
	"fmt"

	"go.scnd.dev/open/syrup/nano/lib/type/enum"
	"go.scnd.dev/open/syrup/nano/lib/type/tuple"
	"go.scnd.dev/open/syrup/nano/lib/util"
)

func (r *Service) ProcessWord(word string) []*tuple.WordPair {
	// try to get word from pogreb directly
	value, err := r.pogreb.WordMapper.Get([]byte(word))
	if err == nil && value != nil {
		_, tokenNo, _ := util.MapperPayloadExtract(value)
		return []*tuple.WordPair{
			{
				Word:  word,
				Token: tokenNo,
			},
		}
	}

	// try to generalize word with suffix
	if generalizedPair, _ := r.ProcessWordGeneralize(word); generalizedPair != nil {
		return generalizedPair
	}

	// word not found
	return []*tuple.WordPair{
		{
			Word:  word,
			Token: 0,
		},
	}
}

func (r *Service) ProcessWordGeneralize(word string) ([]*tuple.WordPair, enum.WordSuffixType) {
	if len(word) < 4 {
		return nil, ""
	}

	// iterate through suffix mappings to find matches
	wordPairs := make([]*tuple.WordPair, 0)
	for {
		block := new(enum.WordSuffixBlock)
		for suffixKey, suffixBlock := range enum.WordSuffix {
			if suffixBlock.Check(word) {
				// assign suffix block
				block = suffixBlock
				word = block.BaseWord(word)

				// get base word from pogreb
				value, err := r.pogreb.WordMapper.Get([]byte(word))
				if err != nil {
					fmt.Printf("error retrieving base word from pogreb: %v\n", err)
				}
				if value == nil {
					continue
				}

				_, tokenNo, _ := util.MapperPayloadExtract(value)
				return append([]*tuple.WordPair{
					{
						Word:  word,
						Token: tokenNo,
					},
					{
						Word:  string(suffixKey),
						Token: suffixBlock.TokenNo,
					},
				}, wordPairs...), suffixKey
			}
		}
		if block.Suffix == "" {
			break
		}
	}

	return nil, ""
}

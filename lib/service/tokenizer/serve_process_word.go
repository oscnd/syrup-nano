package tokenizer

import (
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
	return []*tuple.WordPair{}
}

func (r *Service) ProcessWordGeneralize(word string) ([]*tuple.WordPair, enum.WordSuffixType) {
	if len(word) < 4 {
		return nil, ""
	}

	// Iterate through suffix mappings to find matches
	for suffixKey, suffixBlock := range enum.WordSuffix {
		if suffixBlock.Check(word) {
			baseWord := suffixBlock.BaseWord(word)
			if value, err := r.pogreb.WordMapper.Get([]byte(baseWord)); err == nil && value != nil {
				_, tokenNo, _ := util.MapperPayloadExtract(value)
				return []*tuple.WordPair{
					{
						Word:  baseWord,
						Token: tokenNo,
					},
					{
						Word:  string(suffixKey),
						Token: suffixBlock.TokenNo,
					},
				}, suffixKey
			}
		}
	}

	return nil, ""
}

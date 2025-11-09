package tokenizer

import (
	"go.scnd.dev/open/syrup/nano/lib/type/tuple"
	"go.scnd.dev/open/syrup/nano/lib/util"
)

func (r *Service) ProcessWord(word string) *tuple.WordPair {
	// try to get word from pogreb
	value, err := r.pogreb.WordMapper.Get([]byte(word))
	if err != nil || value == nil {
		// word not found, return WordPair with Token = 0
		return &tuple.WordPair{
			Word:  word,
			Token: 0,
		}
	}

	// extract token number
	tokenNo, _ := util.MapperPayloadExtract(value)
	return &tuple.WordPair{
		Word:  word,
		Token: tokenNo,
	}
}

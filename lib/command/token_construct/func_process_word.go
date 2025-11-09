package main

import (
	"fmt"

	"go.scnd.dev/open/syrup/nano/lib/common/pogreb"
	"go.scnd.dev/open/syrup/nano/lib/util"
)

func ProcessWord(pogreb *pogreb.Pogreb, no *uint64, word string) {
	// * construct variable
	var value []byte
	var err error

	// * generalize word
	if v, modifier := ProcessWordGeneralize(pogreb, word); v != nil {
		value = v
		fmt.Printf("found modifier for word %s: %s\n", word, modifier)
		goto update
	}

	// * get word from pogreb
	value, err = pogreb.WordMapper.Get([]byte(word))
	if err != nil || value == nil {
		// * case word does not exist
		*no++
		tokenNo := *no
		count := uint64(1)

		if err := pogreb.WordMapper.Put([]byte(word), util.MapperPayloadBuild(tokenNo, count)); err != nil {
			fmt.Printf("error inserting word %s: %v\n", word, err)
			return
		}

		fmt.Printf("new word added: %s (tokenNo: %d)\n", word, tokenNo)
		return
	}

update:
	// case word exists
	tokenNo, count := util.MapperPayloadExtract(value)
	count++
	if err := pogreb.WordMapper.Put([]byte(word), util.MapperPayloadBuild(tokenNo, count)); err != nil {
		fmt.Printf("error updating word %s: %v\n", word, err)
		return
	}
}

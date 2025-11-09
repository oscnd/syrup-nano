package main

import (
	"fmt"

	"go.scnd.dev/open/syrup/nano/lib/common/pogreb"
)

func ProcessWord(pogreb *pogreb.Pogreb, no *uint64, word string) {
	// * get word from pogreb
	value, err := pogreb.WordMapper.Get([]byte(word))
	if err != nil || value == nil {
		// * case word does not exist
		*no++
		tokenNo := *no
		count := uint64(1)

		if err := pogreb.WordMapper.Put([]byte(word), MapperPayloadBuild(tokenNo, count)); err != nil {
			fmt.Printf("error inserting word %s: %v\n", word, err)
			return
		}

		fmt.Printf("new word added: %s (tokenNo: %d)\n", word, tokenNo)
		return
	}

	// case word exists
	tokenNo, count := MapperPayloadExtract(value)
	count++
	if err := pogreb.WordMapper.Put([]byte(word), MapperPayloadBuild(tokenNo, count)); err != nil {
		fmt.Printf("error updating word %s: %v\n", word, err)
		return
	}
}

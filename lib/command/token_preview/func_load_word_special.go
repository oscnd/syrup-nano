package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"go.scnd.dev/open/syrup/nano/lib/common/pogreb"
	"go.scnd.dev/open/syrup/nano/lib/type/tuple"
	"go.scnd.dev/open/syrup/nano/lib/util"
)

var wordSpecialLookup = make(map[string][]string)
var wordSpecialToken = make(map[string]uint64)

func LoadWordSpecial(pogreb *pogreb.Pogreb) {
	// glob jsonl files for special words
	pattern := "dataset/tokenizer/word_*.jsonl"
	matches, err := filepath.Glob(pattern)
	if err != nil {
		fmt.Printf("glob error: %v\n", err)
		return
	}

	// process each file
	for _, filePath := range matches {
		loadWordSpecialFromFile(pogreb, filePath)
	}
}

func loadWordSpecialFromFile(pogreb *pogreb.Pogreb, filePath string) {
	file, err := os.Open(filePath)
	if err != nil {
		fmt.Printf("error opening file %s: %v\n", filePath, err)
		return
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Bytes()

		word := new(tuple.Word)
		if err := json.Unmarshal(line, word); err != nil {
			fmt.Printf("error unmarshaling line in file %s: %v\n", filePath, err)
			continue
		}

		// set word special lookup
		if len(word.Word) > 0 {
			firstChar := string(word.Word[0])
			wordSpecialLookup[firstChar] = append(wordSpecialLookup[firstChar], word.Word)

			// load and fetch pogreb for token and store in map
			value, err := pogreb.WordMapper.Get([]byte(word.Word))
			if err != nil || value == nil {
				fmt.Printf("special word %s not found in pogreb\n", word.Word)
				wordSpecialToken[word.Word] = 0
			} else {
				tokenNo, _ := util.MapperPayloadExtract(value)
				wordSpecialToken[word.Word] = tokenNo
			}
		}
	}

	if err := scanner.Err(); err != nil {
		fmt.Printf("error reading file %s: %v\n", filePath, err)
	}
}

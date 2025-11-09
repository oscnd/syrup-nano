package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"go.scnd.dev/open/syrup/nano/lib/common/pogreb"
	"go.scnd.dev/open/syrup/nano/lib/type/enum"
	"go.scnd.dev/open/syrup/nano/lib/type/tuple"
)

var wordSpecialMapper = make(map[string]struct{})
var wordSpecialLookup = make(map[string][]string)

func ConstructWordSpecial(pogreb *pogreb.Pogreb, no *uint64) {
	// * glob jsonl files
	pattern := "dataset/tokenizer/word_*.jsonl"
	matches, err := filepath.Glob(pattern)
	if err != nil {
		fmt.Printf("glob error: %v\n", err)
		return
	}

	// * process word modifier
	for key := range enum.WordModifier {
		ProcessWord(pogreb, no, string(key))
		enum.WordModifier[key] = *no
	}

	// * process word special
	for _, filePath := range matches {
		ConstructWordSpecialFile(pogreb, no, filePath)
	}
}

func ConstructWordSpecialFile(pogreb *pogreb.Pogreb, no *uint64, filePath string) {
	// * open the file
	file, err := os.Open(filePath)
	if err != nil {
		fmt.Printf("error opening file %s: %v\n", filePath, err)
		return
	}
	defer file.Close()

	// * read line by line
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Bytes()

		word := new(tuple.Word)
		if err := json.Unmarshal(line, word); err != nil {
			fmt.Printf("error unmarshaling line in file %s: %v\n", filePath, err)
			continue
		}

		// * set word special mapper
		wordSpecialMapper[word.Word] = struct{}{}

		// * set word special lookup
		firstChar := string(word.Word[0])
		wordSpecialLookup[firstChar] = append(wordSpecialLookup[firstChar], word.Word)

		// * process the word
		ProcessWord(pogreb, no, word.Word)
	}

	if err := scanner.Err(); err != nil {
		fmt.Printf("error reading file %s: %v\n", filePath, err)
	}
}

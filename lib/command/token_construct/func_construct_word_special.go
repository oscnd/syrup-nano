package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"

	"go.scnd.dev/open/syrup/nano/lib/common/pogreb"
	"go.scnd.dev/open/syrup/nano/lib/type/enum"
	"go.scnd.dev/open/syrup/nano/lib/type/tuple"
)

var WordSpecialLookup = make(map[rune][]string)

func ConstructWordSpecialAppend(word string) {
	firstChar := rune(word[0])
	WordSpecialLookup[firstChar] = append(WordSpecialLookup[firstChar], word)
	slices.SortFunc(WordSpecialLookup[firstChar], func(a, b string) int {
		if len(a) != len(b) {
			return len(a) - len(b)
		}
		if a < b {
			return -1
		}
		if a > b {
			return 1
		}
		return 0
	})
}

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

	// * process word suffix
	for key := range enum.WordSuffix {
		ProcessWord(pogreb, no, string(key))
		enum.WordSuffix[key].TokenNo = *no
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

		// * set word special lookup
		ConstructWordSpecialAppend(word.Word)

		// * process the word
		ProcessWord(pogreb, no, word.Word)
	}

	if err := scanner.Err(); err != nil {
		fmt.Printf("error reading file %s: %v\n", filePath, err)
	}
}

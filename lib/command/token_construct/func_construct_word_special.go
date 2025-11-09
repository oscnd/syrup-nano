package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"go.scnd.dev/open/syrup/nano/lib/common/pogreb"
	"go.scnd.dev/open/syrup/nano/lib/type/tuple"
)

func ConstructWordSpecial(pogreb *pogreb.Pogreb, no *uint64) {
	// * glob jsonl files
	pattern := "dataset/tokenizer/word_*.jsonl"
	matches, err := filepath.Glob(pattern)
	if err != nil {
		fmt.Printf("Error globbing files: %v\n", err)
		return
	}

	for _, filePath := range matches {
		ConstructWordSpecialFile(pogreb, no, filePath)
	}
}

func ConstructWordSpecialFile(pogreb *pogreb.Pogreb, no *uint64, filePath string) {
	// * open the file
	file, err := os.Open(filePath)
	if err != nil {
		fmt.Printf("Error opening file %s: %v\n", filePath, err)
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

		// * process the word
		ProcessWord(pogreb, no, word.Word)
	}

	if err := scanner.Err(); err != nil {
		fmt.Printf("Error reading file %s: %v\n", filePath, err)
	}
}

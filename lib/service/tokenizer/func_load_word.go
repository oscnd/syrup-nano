package tokenizer

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"go.scnd.dev/open/syrup/nano/lib/type/tuple"
	"go.scnd.dev/open/syrup/nano/lib/util"
)

func (r *Service) LoadWord() {
	// Load special words from configuration files
	matches, err := filepath.Glob(*r.config.WordSpecialDict)
	if err != nil {
		fmt.Printf("glob error: %v\n", err)
		return
	}

	// Process each special word file
	for _, filePath := range matches {
		r.LoadWordSpecialFromFile(filePath)
	}

	// Load all words from pogreb WordMapper into WordToken map
	fmt.Printf("Loading full word-to-token map from pogreb...\n")
	it := r.pogreb.WordMapper.Items()
	count := 0
	for {
		key, value, err := it.Next()
		if err != nil {
			break
		}

		word := string(key)
		_, tokenNo, _ := util.MapperPayloadExtract(value)
		r.WordToken[word] = tokenNo

		// Also add to lookup for potential matching
		if len(word) > 0 {
			r.WordAppend(word)
		}

		count++
		if count%10000 == 0 {
			fmt.Printf("Loaded %d words...\n", count)
		}
	}

	fmt.Printf("Finished loading %d words into WordToken map\n", count)
}

func (r *Service) LoadWordSpecialFromFile(filePath string) {
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

		// set word lookup
		if len(word.Word) > 0 {
			// append to word lookup list
			r.WordAppend(word.Word)

			// load and fetch pogreb for token and store in map
			value, err := r.pogreb.WordMapper.Get([]byte(word.Word))
			if err != nil || value == nil {
				fmt.Printf("special word %s not found in pogreb\n", word.Word)
				r.WordToken[word.Word] = 0
			} else {
				_, tokenNo, _ := util.MapperPayloadExtract(value)
				r.WordToken[word.Word] = tokenNo
			}
		}
	}

	if err := scanner.Err(); err != nil {
		fmt.Printf("error reading file %s: %v\n", filePath, err)
	}
}

func (r *Service) WordAppend(word string) {
	firstChar := rune(word[0])
	r.WordLookup[firstChar] = append(r.WordLookup[firstChar], &tuple.SpecialWord{
		Text:  word,
		Words: []string{word},
	})
}

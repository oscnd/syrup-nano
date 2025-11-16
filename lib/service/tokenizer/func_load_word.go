package tokenizer

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"

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

	// * process each special word file
	for _, filePath := range matches {
		r.LoadWordSpecialFromFile(filePath)
	}

	// * load pogreb word mapper
	it := r.pogreb.WordMapper.Items()
	for {
		key, value, err := it.Next()
		if err != nil {
			break
		}

		word := string(key)
		_, tokenNo, _ := util.MapperPayloadExtract(value)
		r.WordToken[word] = tokenNo

		if len(word) > 0 {
			r.WordAppend(word)
		}
	}

	fmt.Printf("finished loading %d words into mapper\n", len(r.WordToken))
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

		word := new(tuple.SpecialWord)
		if err := json.Unmarshal(line, word); err != nil {
			fmt.Printf("error unmarshaling line in file %s: %v\n", filePath, err)
			continue
		}

		// set word lookup
		if len(word.Text) > 0 {
			// add to word special lookup
			firstChar := rune(word.Text[0])
			r.WordSpecialLookup[firstChar] = append(r.WordSpecialLookup[firstChar], word)
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

	slices.SortFunc(r.WordLookup[firstChar], tuple.SpecialWordCompare)
}

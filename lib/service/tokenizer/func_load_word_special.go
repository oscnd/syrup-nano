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

func (r *Service) LoadWordSpecial() {
	// find special words declaration
	matches, err := filepath.Glob(*r.config.WordSpecialDict)
	if err != nil {
		fmt.Printf("glob error: %v\n", err)
		return
	}

	// process each file
	for _, filePath := range matches {
		r.LoadWordSpecialFromFile(filePath)
	}
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

		// set word special lookup
		if len(word.Word) > 0 {
			firstChar := string(word.Word[0])
			r.WordSpecialLookup[firstChar] = append(r.WordSpecialLookup[firstChar], word.Word)

			// load and fetch pogreb for token and store in map
			value, err := r.pogreb.WordMapper.Get([]byte(word.Word))
			if err != nil || value == nil {
				fmt.Printf("special word %s not found in pogreb\n", word.Word)
				r.WordSpecialToken[word.Word] = 0
			} else {
				tokenNo, _ := util.MapperPayloadExtract(value)
				r.WordSpecialToken[word.Word] = tokenNo
			}
		}
	}

	if err := scanner.Err(); err != nil {
		fmt.Printf("error reading file %s: %v\n", filePath, err)
	}
}

package constructor

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"

	"go.scnd.dev/open/syrup/nano/lib/type/tuple"
)

func (r *Service) ConstructWordSpecial(pattern string) {
	// * glob jsonl files
	matches, err := filepath.Glob(pattern)
	if err != nil {
		fmt.Printf("glob error: %v\n", err)
		return
	}

	// * process word special
	for _, filePath := range matches {
		r.ConstructWordSpecialFile(filePath)
	}
}

func (r *Service) ConstructWordSpecialFile(filePath string) {
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

		special := new(tuple.SpecialWord)
		if err := json.Unmarshal(line, special); err != nil {
			fmt.Printf("error unmarshaling line in file %s: %v\n", filePath, err)
			continue
		}

		// * set word special lookup
		r.ConstructWordSpecialAppend(special.Text, special.Words)

		// * process word
		r.ProcessWord(special.Text, true)
	}

	if err := scanner.Err(); err != nil {
		fmt.Printf("error reading file %s: %v\n", filePath, err)
	}
}

func (r *Service) ConstructWordSpecialAppend(text string, words []string) {
	firstChar := rune(text[0])
	r.WordSpecialLookup[firstChar] = append(r.WordSpecialLookup[firstChar], &tuple.SpecialWord{
		Text:  text,
		Words: words,
	})
	slices.SortFunc(r.WordSpecialLookup[firstChar], tuple.SpecialWordCompare)
}

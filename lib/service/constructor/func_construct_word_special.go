package constructor

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"

	"go.scnd.dev/open/syrup/nano/lib/type/enum"
	"go.scnd.dev/open/syrup/nano/lib/type/tuple"
)

func (r *Service) ConstructWordSpecialAppend(word string) {
	firstChar := rune(word[0])
	r.WordSpecialLookup[firstChar] = append(r.WordSpecialLookup[firstChar], word)
	slices.SortFunc(r.WordSpecialLookup[firstChar], func(a, b string) int {
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

func (r *Service) ConstructWordSpecial(pattern string) {
	// * glob jsonl files
	matches, err := filepath.Glob(pattern)
	if err != nil {
		fmt.Printf("glob error: %v\n", err)
		return
	}

	// * process word modifier
	for key := range enum.WordModifier {
		r.ProcessWord(string(key))
		enum.WordModifier[key] = r.no
	}

	// * process word suffix
	for key := range enum.WordSuffix {
		r.ProcessWord(string(key))
		enum.WordSuffix[key].TokenNo = r.no
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

		word := new(tuple.Word)
		if err := json.Unmarshal(line, word); err != nil {
			fmt.Printf("error unmarshaling line in file %s: %v\n", filePath, err)
			continue
		}

		// * set word special lookup
		r.ConstructWordSpecialAppend(word.Word)

		// * process the word
		r.ProcessWord(word.Word)
	}

	if err := scanner.Err(); err != nil {
		fmt.Printf("error reading file %s: %v\n", filePath, err)
	}
}

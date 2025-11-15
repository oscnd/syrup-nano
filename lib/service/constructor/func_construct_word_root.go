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

func (r *Service) ConstructWordRoot(pattern string) {
	// * glob jsonl files
	matches, err := filepath.Glob(pattern)
	if err != nil {
		fmt.Printf("glob error: %v\n", err)
		return
	}

	// * process word special
	for _, filePath := range matches {
		r.ConstructWordRootFile(filePath)
	}
}

func (r *Service) ConstructWordRootFile(filePath string) {
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

		compound := new(tuple.CompoundWord)
		if err := json.Unmarshal(line, compound); err != nil {
			fmt.Printf("error unmarshaling line in file %s: %v\n", filePath, err)
			continue
		}

		// * set word special lookup
		r.ConstructWordRootAppend(compound.Compound, compound.Words)

		// * process subwords
		for _, word := range compound.Words {
			r.ProcessWord(word)
		}
	}

	if err := scanner.Err(); err != nil {
		fmt.Printf("error reading file %s: %v\n", filePath, err)
	}
}

func (r *Service) ConstructWordRootAppend(compound string, words []string) {
	firstChar := rune(compound[0])
	r.WordRootLookup[firstChar] = append(r.WordRootLookup[firstChar], &tuple.CompoundWord{
		Compound: compound,
		Words:    words,
	})
	slices.SortFunc(r.WordRootLookup[firstChar], func(a, b *tuple.CompoundWord) int {
		if len(a.Compound) != len(b.Compound) {
			return len(a.Compound) - len(b.Compound)
		}
		if a.Compound < b.Compound {
			return -1
		}
		if a.Compound > b.Compound {
			return 1
		}
		return 0
	})
}

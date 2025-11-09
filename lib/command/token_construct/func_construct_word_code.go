package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"go.scnd.dev/open/syrup/nano/lib/common/pogreb"
	"go.scnd.dev/open/syrup/nano/lib/type/tuple"
)

func ConstructWordCode(pogreb *pogreb.Pogreb, no *uint64) {
	// * glob jsonl files in download/code/
	pattern := "download/code/*.jsonl"
	matches, err := filepath.Glob(pattern)
	if err != nil {
		fmt.Printf("glob error: %v\n", err)
		return
	}

	// * process each code file
	for _, filePath := range matches {
		ConstructWordCodeFile(pogreb, no, filePath)
	}
}

func ConstructWordCodeFile(pogreb *pogreb.Pogreb, no *uint64, filePath string) {
	// * open the file
	file, err := os.Open(filePath)
	if err != nil {
		fmt.Printf("error opening file %s: %v\n", filePath, err)
		return
	}
	defer file.Close()

	// * read line by line
	scanner := bufio.NewScanner(file)
	for lineNo := 1; scanner.Scan(); lineNo++ {
		line := scanner.Bytes()

		code := new(tuple.Code)
		if err := json.Unmarshal(line, code); err != nil {
			fmt.Printf("error unmarshaling line %d in file %s: %v\n", lineNo, filePath, err)
			continue
		}

		// * process the content using ProcessLine
		contentLines := strings.Split(code.Content, "\n")
		for contentLineNo, contentLine := range contentLines {
			// * process line
			values := ProcessLine(pogreb, contentLine)

			for _, value := range values {
				switch value.(type) {
				case uint64:
					continue
				case string:
					word := value.(string)
					ProcessWord(pogreb, no, word)
				}
			}

			// * optional: log processing
			fmt.Printf("processed code: %s:%d, extracted %d words\n", code.Path, contentLineNo, len(values))
		}
	}

	if err := scanner.Err(); err != nil {
		fmt.Printf("error reading file %s: %v\n", filePath, err)
	}
}

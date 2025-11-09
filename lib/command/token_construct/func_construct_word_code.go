package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
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

	// * set buffer
	buf := make([]byte, 0, 1024*1024)
	scanner.Buffer(buf, 1024*1024)

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

			// * preflight test
			for _, value := range values {
				word, ok := value.(string)
				if !ok {
					continue
				}

				if matched, _ := regexp.MatchString("^[a-z]+$", word); !matched {
					fmt.Printf("invalid word '%s' extracted from %s:%d, %s:%d\n", word, filepath.Base(filePath), lineNo, code.Path, contentLineNo)
					goto next
				}
			}

			// * store words
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
			if false {
				fmt.Printf("processed code: %s:%d, %s:%d, extracted %d words\n", filepath.Base(filePath), lineNo, code.Path, contentLineNo, len(values))
			}
		next:
		}
	}

	if err := scanner.Err(); err != nil {
		fmt.Printf("error reading file %s: %v\n", filePath, err)
	}
}

package constructor

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"go.scnd.dev/open/syrup/nano/lib/type/tuple"
)

func (r *Service) ConstructFromGlob(pattern string) {
	// * glob jsonl files
	matches, err := filepath.Glob(pattern)
	if err != nil {
		fmt.Printf("glob error: %v\n", err)
		return
	}

	// * process each code file
	for _, filePath := range matches {
		r.ConstructFromFile(filePath)
	}
}

func (r *Service) ConstructFromFile(filePath string) {
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

		// * process the content using ConstructContent
		r.ConstructContent(code.Path, code.Content)
	}

	if err := scanner.Err(); err != nil {
		fmt.Printf("error reading file %s: %v\n", filePath, err)
	}
}

func (r *Service) ConstructContent(filename string, content string) {
	// * process the content using ProcessLine
	contentLines := strings.Split(content, "\n")
	for contentLineNo, contentLine := range contentLines {
		// * process line
		values := r.ProcessLine(contentLine)

		// * preflight test
		for _, value := range values {
			word, ok := value.(string)
			if !ok {
				continue
			}

			if matched, _ := regexp.MatchString("^[a-z]+$", word); !matched {
				if false {
					fmt.Printf("invalid word '%s' extracted from %s:%d\n", word, filename, contentLineNo)
				}
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
				r.ProcessWord(word, false)
			}
		}

		// * optional: log processing
		if false {
			fmt.Printf("processed content: %s:%d, extracted %d words\n", filename, contentLineNo, len(values))
		}
	next:
	}
}

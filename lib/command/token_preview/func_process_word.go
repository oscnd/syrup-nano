package main

import (
	"fmt"
	"strings"

	"github.com/bsthun/gut"
	"go.scnd.dev/open/syrup/nano/lib/common/pogreb"
	"go.scnd.dev/open/syrup/nano/lib/util"
)

func ProcessWord(pogreb *pogreb.Pogreb, word string) string {
	// try to get word from pogreb
	value, err := pogreb.WordMapper.Get([]byte(word))
	if err != nil || value == nil {
		// word not found, return truncated word with no token
		return FormatWordOutput(word, "")
	}

	// extract token number
	tokenNo, _ := util.MapperPayloadExtract(value)
	tokenStr := gut.Base62(tokenNo)

	return FormatWordOutput(word, tokenStr)
}

func FormatWordOutput(word, token string) string {
	// case of special character
	if word == " " {
		word = "#spc#"
	}

	if word == "\t" {
		word = "#tab#"
	}

	// truncate word to 6 characters (with 7 characters)
	if len(word) >= 10 {
		word = word[:9] + "…" // U+2026
	} else if len(word) < 10 {
		word = word + strings.Repeat(" ", 10-len(word))
	}

	if token == "" {
		token = strings.Repeat(" ", 11)
	}

	return fmt.Sprintf("%s %s", word, token)
}

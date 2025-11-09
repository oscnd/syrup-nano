package main

import (
	"fmt"
	"strings"

	"github.com/bsthun/gut"
	"go.scnd.dev/open/syrup/nano/lib/type/tuple"
)

func OutputToken(pairs []*tuple.WordPair) {
	tokenPerLine := 5

	for i, pair := range pairs {
		word := pair.Word

		// handle special characters
		if word == " " {
			word = "#spc#"
		}
		if word == "\t" {
			word = "#tab#"
		}
		if word == "\n" {
			word = "#nl#"
		}

		// truncate word
		if len(word) > 10 {
			word = word[:9] + "…" // U+2026
		} else if len(word) < 10 {
			word = word + strings.Repeat(" ", 10-len(word))
		}

		// handle token display
		token := "—" + strings.Repeat(" ", 10)
		if pair.Token != 0 {
			token = gut.Base62(pair.Token)
		}

		output := fmt.Sprintf("%s %s ", word, token)
		fmt.Print(output)

		// add newline
		if (i+1)%tokenPerLine == 0 {
			fmt.Println()
		}
	}

	// add final newline
	if len(pairs)%tokenPerLine != 0 {
		fmt.Println()
	}
}

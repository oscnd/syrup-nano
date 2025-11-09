package main

import (
	"fmt"
)

func OutputTokens(tokens []string) {
	tokenPerLine := 5

	for i, token := range tokens {
		fmt.Print(token)

		// add space between tokens
		if (i+1)%tokenPerLine != 0 && i != len(tokens)-1 {
			fmt.Print(" ")
		}

		// add newline after every 10 tokens
		if (i+1)%tokenPerLine == 0 {
			fmt.Println()
		}
	}

	// add final newline
	if len(tokens)%tokenPerLine != 0 {
		fmt.Println()
	}
}

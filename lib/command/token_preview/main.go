package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"

	"github.com/bsthun/gut"
	"go.scnd.dev/open/syrup/nano/lib/common/config"
	"go.scnd.dev/open/syrup/nano/lib/common/pogreb"
	"go.uber.org/fx"
)

func main() {
	// parse file flag
	filePath := flag.String("file", "", "Path to the file to process")
	flag.Parse()

	if *filePath == "" {
		gut.Fatal("File path is required. Use -file flag.", fmt.Errorf("missing file path"))
	}

	// check if file exists
	if _, err := os.Stat(*filePath); os.IsNotExist(err) {
		gut.Fatal("File does not exist: ", err)
	}

	// main fx application
	fx.New(
		fx.Provide(
			config.Init,
			pogreb.Init,
		),
		fx.Invoke(
			func(config *config.Config, pogreb *pogreb.Pogreb) {
				invoke(pogreb, *filePath)
			},
		),
	).Run()
}

func invoke(pogreb *pogreb.Pogreb, filePath string) {
	// load special word
	LoadSpecialWord()

	// open and read file
	file, err := os.Open(filePath)
	if err != nil {
		gut.Fatal("Error opening file: ", err)
	}
	defer file.Close()

	// process file line by line
	scanner := bufio.NewScanner(file)
	var tokens []string

	for scanner.Scan() {
		line := scanner.Text()
		lineTokens := ProcessLine(pogreb, line)
		tokens = append(tokens, lineTokens...)
	}

	if err := scanner.Err(); err != nil {
		gut.Fatal("Error reading file: ", err)
	}

	// output formatted tokens
	OutputTokens(tokens)
	os.Exit(0)
}

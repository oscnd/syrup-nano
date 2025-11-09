package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"

	"github.com/bsthun/gut"
	"go.scnd.dev/open/syrup/nano/lib/common/config"
	"go.scnd.dev/open/syrup/nano/lib/common/pogreb"
	"go.scnd.dev/open/syrup/nano/lib/service/tokenizer"
	"go.scnd.dev/open/syrup/nano/lib/type/tuple"
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
			tokenizer.Serve,
		),
		fx.Invoke(
			func(shutdowner fx.Shutdowner, pogreb *pogreb.Pogreb, tokenizer tokenizer.Server) {
				invoke(shutdowner, pogreb, tokenizer, *filePath)
			},
		),
	).Run()
}

func invoke(shutdowner fx.Shutdowner, pogreb *pogreb.Pogreb, tokenizer tokenizer.Server, filePath string) {
	// open and read file
	file, err := os.Open(filePath)
	if err != nil {
		gut.Fatal("Error opening file: ", err)
	}
	defer file.Close()

	// process file line by line
	scanner := bufio.NewScanner(file)
	var pairs []*tuple.WordPair

	endLinePair := tokenizer.ProcessWord("\n")
	for scanner.Scan() {
		line := scanner.Text()
		linePairs := tokenizer.ProcessLine(line)
		pairs = append(pairs, linePairs...)
		pairs = append(pairs, endLinePair)
	}

	if err := scanner.Err(); err != nil {
		gut.Fatal("Error reading file: ", err)
	}

	// output formatted token
	OutputToken(pairs)

	//_ = shutdowner.Shutdown()
	os.Exit(0)
}

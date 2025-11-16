package main

import (
	"flag"
	"fmt"

	"github.com/bsthun/gut"
	"go.scnd.dev/open/syrup/nano/lib/common/config"
	"go.scnd.dev/open/syrup/nano/lib/common/fxo"
	"go.scnd.dev/open/syrup/nano/lib/common/pogreb"
	"go.scnd.dev/open/syrup/nano/lib/service/tokenizer"
	"go.uber.org/fx"
)

func main() {
	// parse text flag
	text := flag.String("text", "", "Text to encode")
	flag.Parse()

	if *text == "" {
		gut.Fatal("Text is required. Use -text flag.", fmt.Errorf("missing text"))
	}

	// main fx application
	fx.New(
		fxo.Option(),
		fx.Provide(
			config.Init,
			pogreb.Init,
			tokenizer.Serve,
		),
		fx.Invoke(
			func(shutdowner fx.Shutdowner, tokenizer tokenizer.Server) {
				invoke(shutdowner, tokenizer, *text)
			},
		),
	).Run()
}

func invoke(shutdowner fx.Shutdowner, tokenizer tokenizer.Server, text string) {
	// process text
	pairs := tokenizer.ProcessLine(text)

	// output as Go slice format
	fmt.Println("[]*tuple.WordPair{")
	for i := 0; i < len(pairs); i++ {
		pair := pairs[i]
		// escape the word for Go string representation
		word := pair.Word
		if word == " " {
			word = " "
		}
		if i < len(pairs)-1 {
			fmt.Printf("	WordPair(%q, %d),\n", word, pair.Token)
		} else {
			fmt.Printf("	WordPair(%q, %d)\n", word, pair.Token)
		}
	}
	fmt.Println("}")

	_ = shutdowner.Shutdown()
}

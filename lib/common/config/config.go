package config

import (
	"os"

	"github.com/bsthun/gut"
	"gopkg.in/yaml.v3"
)

type Config struct {
	PogrebWordMapper  *string `yaml:"pogrebWordMapper"`
	PogrebTokenMapper *string `yaml:"pogrebTokenMapper"`
}

func Init() *Config {
	// * parse arguments
	path := os.Getenv("BACKEND_CONFIG_PATH")
	if path == "" {
		path = "config.yml"
	}

	// * declare struct
	config := new(Config)

	// * read config
	yml, err := os.ReadFile(path)
	if err != nil {
		gut.Fatal("Unable to read configuration file", err)
	}

	// * parse config
	if err := yaml.Unmarshal(yml, config); err != nil {
		gut.Fatal("Unable to parse configuration file", err)
	}

	// * validate config
	if err := gut.Validate(config); err != nil {
		gut.Fatal("Invalid configuration", err)
	}

	return config
}

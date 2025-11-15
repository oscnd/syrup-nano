package tuple

type Word struct {
	Word string `json:"word"`
}

type WordPair struct {
	Word  string `json:"word"`
	Token uint64 `json:"token"`
}

type CompoundWord struct {
	Compound string   `json:"compound"`
	Words    []string `json:"words"`
}

type Code struct {
	Path    string `json:"path"`
	Content string `json:"content"`
}

package tuple

type Word struct {
	Word string `json:"word"`
}

type WordPair struct {
	Word  string `json:"word"`
	Token uint64 `json:"token"`
}

type SpecialWord struct {
	Text  string   `json:"text"`
	Words []string `json:"words"`
}

func SpecialWordCompare(a, b *SpecialWord) int {
	if len(a.Text) != len(b.Text) {
		return len(b.Text) - len(a.Text)
	}
	if a.Text < b.Text {
		return -1
	}
	if a.Text > b.Text {
		return 1
	}
	return 0
}

type Code struct {
	Path    string `json:"path"`
	Content string `json:"content"`
}

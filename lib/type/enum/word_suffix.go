package enum

import (
	"strings"

	"go.scnd.dev/open/syrup/nano/lib/util"
)

type WordSuffixType string

type WordSuffixBlock struct {
	Suffix   string
	TokenNo  uint64
	Check    func(word string) bool
	BaseWord func(word string) string
	FullWord func(word string) string
}

var WordSuffix = map[WordSuffixType]*WordSuffixBlock{
	"#suffixS#": {
		Suffix:  "s",
		TokenNo: 0,
		Check: func(word string) bool {
			return len(word) > 3 &&
				word[len(word)-1] == 's' &&
				word[len(word)-2] != 's' &&
				word[len(word)-2] != 'y' &&
				!util.CharacterVowel(rune(word[len(word)-2]))
		},
		BaseWord: func(word string) string {
			return strings.TrimSuffix(word, "s")
		},
		FullWord: func(word string) string {
			return word + "s"
		},
	},
	"#suffixEs#": {
		Suffix:  "es",
		TokenNo: 0,
		Check: func(word string) bool {
			return len(word) > 4 &&
				(strings.HasSuffix(word, "ses") || strings.HasSuffix(word, "xes") ||
					strings.HasSuffix(word, "zes") || strings.HasSuffix(word, "ches") ||
					strings.HasSuffix(word, "shes"))
		},
		BaseWord: func(word string) string {
			if strings.HasSuffix(word, "ses") {
				return strings.TrimSuffix(word, "es")
			}
			return strings.TrimSuffix(word, "es")
		},
		FullWord: func(word string) string {
			lastChar := word[len(word)-1]
			if lastChar == 's' || lastChar == 'x' || lastChar == 'z' {
				return word + "es"
			}
			if len(word) >= 2 {
				lastTwo := word[len(word)-2:]
				if lastTwo == "ch" || lastTwo == "sh" {
					return word + "es"
				}
			}
			return word + "s"
		},
	},
	"#suffixIes#": {
		Suffix:  "ies",
		TokenNo: 0,
		Check: func(word string) bool {
			return len(word) > 5 && strings.HasSuffix(word, "ies")
		},
		BaseWord: func(word string) string {
			base := strings.TrimSuffix(word, "ies")
			return base + "y"
		},
		FullWord: func(word string) string {
			if len(word) > 0 && word[len(word)-1] == 'y' {
				return strings.TrimSuffix(word, "y") + "ies"
			}
			return word + "ies"
		},
	},
	"#suffixNess#": {
		Suffix:  "ness",
		TokenNo: 0,
		Check: func(word string) bool {
			return len(word) > 6 && strings.HasSuffix(word, "ness")
		},
		BaseWord: func(word string) string {
			return strings.TrimSuffix(word, "ness")
		},
		FullWord: func(word string) string {
			return word + "ness"
		},
	},
	"#suffixTion#": {
		Suffix:  "tion",
		TokenNo: 0,
		Check: func(word string) bool {
			return len(word) > 6 && strings.HasSuffix(word, "tion")
		},
		BaseWord: func(word string) string {
			return strings.TrimSuffix(word, "tion")
		},
		FullWord: func(word string) string {
			return word + "tion"
		},
	},
	"#suffixMent#": {
		Suffix:  "ment",
		TokenNo: 0,
		Check: func(word string) bool {
			return len(word) > 6 && strings.HasSuffix(word, "ment")
		},
		BaseWord: func(word string) string {
			return strings.TrimSuffix(word, "ment")
		},
		FullWord: func(word string) string {
			return word + "ment"
		},
	},
	"#suffixIty#": {
		Suffix:  "ity",
		TokenNo: 0,
		Check: func(word string) bool {
			return len(word) > 5 && strings.HasSuffix(word, "ity")
		},
		BaseWord: func(word string) string {
			return strings.TrimSuffix(word, "ity")
		},
		FullWord: func(word string) string {
			return word + "ity"
		},
	},
	"#suffixLy#": {
		Suffix:  "ly",
		TokenNo: 0,
		Check: func(word string) bool {
			return len(word) > 4 && strings.HasSuffix(word, "ly")
		},
		BaseWord: func(word string) string {
			return strings.TrimSuffix(word, "ly")
		},
		FullWord: func(word string) string {
			return word + "ly"
		},
	},
	"#suffixIng#": {
		Suffix:  "ing",
		TokenNo: 0,
		Check: func(word string) bool {
			return len(word) > 5 && strings.HasSuffix(word, "ing")
		},
		BaseWord: func(word string) string {
			return strings.TrimSuffix(word, "ing")
		},
		FullWord: func(word string) string {
			return word + "ing"
		},
	},
	"#suffixEd#": {
		Suffix:  "ed",
		TokenNo: 0,
		Check: func(word string) bool {
			return len(word) > 4 && strings.HasSuffix(word, "ed")
		},
		BaseWord: func(word string) string {
			return strings.TrimSuffix(word, "ed")
		},
		FullWord: func(word string) string {
			return word + "ed"
		},
	},
	"#suffixEr#": {
		Suffix:  "er",
		TokenNo: 0,
		Check: func(word string) bool {
			return len(word) > 4 && strings.HasSuffix(word, "er")
		},
		BaseWord: func(word string) string {
			return strings.TrimSuffix(word, "er")
		},
		FullWord: func(word string) string {
			return word + "er"
		},
	},
	"#suffixAble#": {
		Suffix:  "able",
		TokenNo: 0,
		Check: func(word string) bool {
			return len(word) > 6 && strings.HasSuffix(word, "able")
		},
		BaseWord: func(word string) string {
			return strings.TrimSuffix(word, "able")
		},
		FullWord: func(word string) string {
			return word + "able"
		},
	},
	"#suffixIble#": {
		Suffix:  "ible",
		TokenNo: 0,
		Check: func(word string) bool {
			return len(word) > 6 && strings.HasSuffix(word, "ible")
		},
		BaseWord: func(word string) string {
			return strings.TrimSuffix(word, "ible")
		},
		FullWord: func(word string) string {
			return word + "ible"
		},
	},
	"#suffixIve#": {
		Suffix:  "ive",
		TokenNo: 0,
		Check: func(word string) bool {
			return len(word) > 5 && strings.HasSuffix(word, "ive")
		},
		BaseWord: func(word string) string {
			return strings.TrimSuffix(word, "ive")
		},
		FullWord: func(word string) string {
			return word + "ive"
		},
	},
	"#suffixIze#": {
		Suffix:  "ize",
		TokenNo: 0,
		Check: func(word string) bool {
			return len(word) > 5 && strings.HasSuffix(word, "ize")
		},
		BaseWord: func(word string) string {
			return strings.TrimSuffix(word, "ize")
		},
		FullWord: func(word string) string {
			return word + "ize"
		},
	},
	"#suffixFul#": {
		Suffix:  "ful",
		TokenNo: 0,
		Check: func(word string) bool {
			return len(word) > 5 && strings.HasSuffix(word, "ful")
		},
		BaseWord: func(word string) string {
			return strings.TrimSuffix(word, "ful")
		},
		FullWord: func(word string) string {
			return word + "ful"
		},
	},
	"#suffixLess#": {
		Suffix:  "less",
		TokenNo: 0,
		Check: func(word string) bool {
			return len(word) > 6 && strings.HasSuffix(word, "less")
		},
		BaseWord: func(word string) string {
			return strings.TrimSuffix(word, "less")
		},
		FullWord: func(word string) string {
			return word + "less"
		},
	},
}

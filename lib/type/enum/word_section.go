package enum

import "fmt"

type WordSectionType string

const (
	WordSectionInstructionStart WordSectionType = "#sectionInstructionStart#"
	WordSectionInstructionEnd   WordSectionType = "#sectionInstructionEnd#"
	WordSectionInputStart       WordSectionType = "#sectionInputStart#"
	WordSectionInputEnd         WordSectionType = "#sectionInputEnd#"
	WordSectionThinkingStart    WordSectionType = "#sectionThinkingStart#"
	WordSectionThinkingEnd      WordSectionType = "#sectionThinkingEnd#"
	WordSectionOutputStart      WordSectionType = "#sectionOutputStart#"
	WordSectionOutputEnd        WordSectionType = "#sectionOutputEnd#"
)

var WordSection = map[WordSectionType]uint64{
	WordSectionInstructionStart: 0,
	WordSectionInstructionEnd:   1,
	WordSectionInputStart:       2,
	WordSectionInputEnd:         3,
	WordSectionThinkingStart:    4,
	WordSectionThinkingEnd:      5,
	WordSectionOutputStart:      6,
	WordSectionOutputEnd:        7,
}

func init() {
	for i := range 1024 {
		WordSection[WordSectionType(fmt.Sprintf("#sectionTemplate%3d", i))] = uint64(0)
	}
}

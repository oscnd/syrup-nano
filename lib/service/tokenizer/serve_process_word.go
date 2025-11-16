package tokenizer

import (
	"strings"
	"unicode"

	"go.scnd.dev/open/syrup/nano/lib/type/enum"
	"go.scnd.dev/open/syrup/nano/lib/type/tuple"
	"go.scnd.dev/open/syrup/nano/lib/util"
)

func (r *Service) ProcessWord(word string) []*tuple.WordPair {
	// try to get word from WordToken map directly
	if tokenNo, exists := r.WordToken[word]; exists {
		return []*tuple.WordPair{
			{
				Word:  word,
				Token: tokenNo,
			},
		}
	}

	// try to generalize word with suffix
	if generalizedPair, _ := r.ProcessWordGeneralize(word); generalizedPair != nil {
		return generalizedPair
	}

	// word not found
	return []*tuple.WordPair{
		{
			Word:  word,
			Token: 0,
		},
	}
}

func (r *Service) ProcessWordGeneralize(word string) ([]*tuple.WordPair, enum.WordSuffixType) {
	if len(word) < 4 {
		return nil, ""
	}

	// iterate through suffix mappings to find matches
	wordPairs := make([]*tuple.WordPair, 0)
	for {
		block := new(enum.WordSuffixBlock)
		for suffixKey, suffixBlock := range enum.WordSuffix {
			if suffixBlock.Check(word) {
				// assign suffix block
				block = suffixBlock
				word = block.BaseWord(word)

				// get base word from WordToken map
				tokenNo, exists := r.WordToken[word]
				if !exists {
					continue
				}
				return append([]*tuple.WordPair{
					{
						Word:  word,
						Token: tokenNo,
					},
					{
						Word:  string(suffixKey),
						Token: suffixBlock.TokenNo,
					},
				}, wordPairs...), suffixKey
			}
		}
		if block.Suffix == "" {
			break
		}
	}

	return nil, ""
}

// ProcessCharacterLookup finds the longest word match starting at the given index
func (r *Service) ProcessCharacterLookup(line string, startIdx int) ([]*tuple.WordPair, int) {
	// * find word boundary for regular word matching
	boundary := r.ProcessWordBoundary(line, startIdx)

	// * try to find the longest regular word match with suffix decomposition
	if boundary > 0 {
		// * try from longest to shortest to find the best match
		for length := boundary; length >= 1; length-- {
			substring := line[startIdx : startIdx+length]

			// * try suffix decomposition
			if pairs := r.ProcessWordSuffixMatch(substring); pairs != nil {
				return pairs, length
			}
		}
	}

	// * check for special words
	if specialWord := util.WordLookupCheck(line, startIdx, r.WordLookup); specialWord != nil {
		pairs := make([]*tuple.WordPair, 0)
		for _, subword := range specialWord.Words {
			wordPair := r.ProcessWord(subword)
			pairs = append(pairs, wordPair...)
		}
		return pairs, len(specialWord.Text)
	}

	// * return single character with token 0
	char := unicode.ToLower(rune(line[startIdx]))
	return []*tuple.WordPair{
		{
			Word:  string(char),
			Token: 0,
		},
	}, 1
}

// ProcessWordBoundary finds how far we can look ahead from startIdx
func (r *Service) ProcessWordBoundary(line string, startIdx int) int {
	maxLen := 0
	for i := startIdx; i < len(line); i++ {
		// * stop at uppercase
		if unicode.IsUpper(rune(line[i])) {
			break
		}

		maxLen++
		if maxLen >= 16 {
			break
		}
	}
	return maxLen
}

// ProcessWordSuffixMatch tries to match word as base or decompose into base + suffixes
func (r *Service) ProcessWordSuffixMatch(word string) []*tuple.WordPair {
	lowercaseWord := strings.ToLower(word)

	// try direct match first
	if tokenNo, exists := r.WordToken[lowercaseWord]; exists {
		return []*tuple.WordPair{
			{
				Word:  lowercaseWord,
				Token: tokenNo,
			},
		}
	}

	// try suffix decomposition to find base word + suffixes
	return r.ProcessWordSuffixDecompose(lowercaseWord)
}

// ProcessWordSuffixDecompose tries to decompose a word into base + suffix chain
func (r *Service) ProcessWordSuffixDecompose(word string) []*tuple.WordPair {
	if len(word) < 4 {
		return nil
	}

	var suffixPairs []*tuple.WordPair
	currentWord := word

	// * keep stripping suffixes until we find a base word
	for {
		foundSuffix := false

		// * try each suffix
		for suffixKey, suffixBlock := range enum.WordSuffix {
			if !suffixBlock.Check(currentWord) {
				continue
			}

			baseWord := suffixBlock.BaseWord(currentWord)

			// * check if base word exists in vocabulary
			if tokenNo, exists := r.WordToken[baseWord]; exists {
				// * base word found, return full decomposition
				result := []*tuple.WordPair{
					{
						Word:  baseWord,
						Token: tokenNo,
					},
					{
						Word:  string(suffixKey),
						Token: suffixBlock.TokenNo,
					},
				}
				result = append(result, suffixPairs...)
				return result
			}

			// * case of base word not found, continue stripping suffixes
			suffixPairs = append([]*tuple.WordPair{
				{
					Word:  string(suffixKey),
					Token: suffixBlock.TokenNo,
				},
			}, suffixPairs...)

			currentWord = baseWord
			foundSuffix = true
			break
		}

		// * no more suffixes found
		if !foundSuffix {
			break
		}

		// * prevent infinite loop
		if len(currentWord) < 4 {
			break
		}
	}

	return nil
}

package main

import (
	"flag"
	"fmt"
	"log"
	"sort"
	"strings"

	"go.scnd.dev/open/syrup/nano/lib/common/config"
	"go.scnd.dev/open/syrup/nano/lib/common/fxo"
	"go.scnd.dev/open/syrup/nano/lib/common/pogreb"
	"go.scnd.dev/open/syrup/nano/lib/service/constructor"
	"go.scnd.dev/open/syrup/nano/lib/util"
	"go.uber.org/fx"
)

// * configuration constants
const (
	targetWords   = 100000
	maxWordLength = 16
	minWordCount  = 20
)

// * word data structure for processing
type wordData struct {
	word      string
	tokenNo   uint64
	count     uint64
	isSpecial bool
}

// * subword extraction candidate
type subwordCandidate struct {
	text         string
	length       int
	parentWords  []string
	wordCount    int
	netReduction int
}

// * extraction step result
type extractionStep struct {
	subword      string
	parentWords  []string
	remainWords  []string
	wordsRemoved int
	wordsAdded   int
	netReduction int
}

func main() {
	var apply = flag.Bool("apply", false, "apply changes to database instead of dry run")
	flag.Parse()

	fx.New(
		fxo.Option(),
		fx.Provide(
			config.Init,
			pogreb.Init,
			constructor.Serve,
		),
		fx.Invoke(func(shutdowner fx.Shutdowner, pogreb *pogreb.Pogreb, constructor constructor.Server) {
			CompactTokens(pogreb, constructor, *apply)
			_ = shutdowner.Shutdown()
		}),
	).Run()
}

// * main token compaction function
func CompactTokens(pogreb *pogreb.Pogreb, constructor constructor.Server, apply bool) {
	// * load words, separating special and non-special
	specialWords, nonSpecialWords := loadAndSeparateWords(pogreb)

	fmt.Printf("    loaded %d total words (%d special, %d non-special)\n",
		len(specialWords)+len(nonSpecialWords), len(specialWords), len(nonSpecialWords))

	if len(nonSpecialWords) <= targetWords {
		fmt.Printf("non-special words already at target (%d words)\n", len(nonSpecialWords))
		// * only reassign non-special tokens, keep special tokens unchanged
		reassignNonSpecialTokensOnly(pogreb, specialWords, nonSpecialWords, apply)
		return
	}

	// * step 1: character truncation (non-special only)
	nonSpecialWords = applyCharacterTruncation(nonSpecialWords, pogreb, apply)

	// * step 2: remove long words (non-special only)
	nonSpecialWords = removeLongWords(nonSpecialWords, pogreb, apply)

	// * step 3: remove low frequency words (non-special only)
	nonSpecialWords = removeLowFrequencyWords(nonSpecialWords, pogreb, apply)

	// * step 4: subword extraction (non-special only)
	if len(nonSpecialWords) > targetWords {
		steps := performSubwordExtraction(nonSpecialWords, targetWords)
		if apply && len(steps) > 0 {
			applyExtractionSteps(pogreb, constructor, steps)
		}
		// * reload non-special words after extraction
		_, nonSpecialWords = loadAndSeparateWords(pogreb)
	}

	// * step 5: reassign only non-special tokens, keep special tokens unchanged
	reassignNonSpecialTokensOnly(pogreb, specialWords, nonSpecialWords, apply)
}

// * load all words from database, separating special and non-special
func loadAndSeparateWords(pogreb *pogreb.Pogreb) (specialWords []wordData, nonSpecialWords []wordData) {
	specialMap := make(map[string]wordData)
	nonSpecialMap := make(map[string]wordData)
	it := pogreb.WordMapper.Items()

	for {
		key, value, err := it.Next()
		if err != nil {
			break
		}

		word := string(key)
		isSpecial, tokenNo, count := util.MapperPayloadExtract(value)
		wd := wordData{
			word:      word,
			tokenNo:   tokenNo,
			count:     count,
			isSpecial: isSpecial,
		}

		if isSpecial {
			// * handle special words - preserve first occurrence, merge duplicates
			if existing, exists := specialMap[word]; exists {
				// * merge counts but preserve original token and special flag
				merged := wordData{
					word:      word,
					tokenNo:   existing.tokenNo,
					count:     existing.count + count,
					isSpecial: true,
				}
				specialMap[word] = merged
			} else {
				specialMap[word] = wd
			}
		} else {
			// * handle non-special words - keep highest count
			if existing, exists := nonSpecialMap[word]; exists {
				if wd.count > existing.count {
					nonSpecialMap[word] = wd
				}
			} else {
				nonSpecialMap[word] = wd
			}
		}
	}

	// * convert maps to slices
	for _, wd := range specialMap {
		specialWords = append(specialWords, wd)
	}
	for _, wd := range nonSpecialMap {
		nonSpecialWords = append(nonSpecialWords, wd)
	}

	fmt.Printf("loaded %d special words and %d non-special words\n", len(specialWords), len(nonSpecialWords))
	return specialWords, nonSpecialWords
}

// * legacy function for compatibility
func loadAllWords(pogreb *pogreb.Pogreb) []wordData {
	specialWords, nonSpecialWords := loadAndSeparateWords(pogreb)
	return append(specialWords, nonSpecialWords...)
}

// * truncate consecutive characters (3+ -> 1)
func applyCharacterTruncation(words []wordData, pogreb *pogreb.Pogreb, apply bool) []wordData {
	truncationMap := make(map[string]string)
	var result []wordData

	// * identify words needing truncation
	for _, wd := range words {
		// * skip special words
		if wd.isSpecial {
			result = append(result, wd)
			continue
		}

		truncated := truncateConsecutiveChars(wd.word)
		if truncated != wd.word {
			truncationMap[wd.word] = truncated
		} else {
			result = append(result, wd)
		}
	}

	fmt.Printf("found %d words with 3+ consecutive characters\n", len(truncationMap))

	if !apply {
		printTruncationDryRun(truncationMap)
	}

	// * add truncated words
	for original, truncated := range truncationMap {
		// * find the original word data to preserve special flag and count
		var originalData wordData
		found := false
		for _, wd := range words {
			if wd.word == original {
				originalData = wd
				found = true
				break
			}
		}

		if found {
			result = append(result, wordData{
				word:      truncated,
				tokenNo:   0, // * will be reassigned later
				count:     originalData.count,
				isSpecial: originalData.isSpecial,
			})
		} else {
			// * fallback
			result = append(result, wordData{
				word:      truncated,
				tokenNo:   0,
				count:     0,
				isSpecial: false,
			})
		}
	}

	// * apply changes to database
	if apply {
		applyTruncationToDatabase(pogreb, truncationMap)
	}

	return result
}

// * truncate consecutive characters to single
func truncateConsecutiveChars(word string) string {
	if len(word) < 3 {
		return word
	}

	var result []rune
	chars := []rune(word)

	for i := 0; i < len(chars); {
		j := i + 1
		for j < len(chars) && chars[j] == chars[i] {
			j++
		}

		if j-i >= 3 {
			result = append(result, chars[i])
		} else {
			for k := i; k < j; k++ {
				result = append(result, chars[k])
			}
		}
		i = j
	}

	return string(result)
}

// * print truncation preview for dry run
func printTruncationDryRun(truncationMap map[string]string) {
	fmt.Printf("\n=== character truncation (dry run) ===\n")
	count := 0
	for orig, trunc := range truncationMap {
		if count >= 10 {
			fmt.Printf("  ... and %d more\n", len(truncationMap)-10)
			break
		}
		fmt.Printf("  '%s' -> '%s'\n", orig, trunc)
		count++
	}
	fmt.Printf("\n")
}

// * apply truncation changes to database
func applyTruncationToDatabase(pogreb *pogreb.Pogreb, truncationMap map[string]string) {
	for original, truncated := range truncationMap {
		// * get original word data
		value, err := pogreb.WordMapper.Get([]byte(original))
		if err != nil || value == nil {
			continue
		}

		_, tokenNo, count := util.MapperPayloadExtract(value)

		// * delete original word
		pogreb.WordMapper.Delete([]byte(original))
		pogreb.TokenMapper.Delete(util.Uint64ToBytes(tokenNo))

		// * check if truncated version already exists
		existing, _ := pogreb.WordMapper.Get([]byte(truncated))
		if existing != nil {
			// * merge counts
			_, existingTokenNo, existingCount := util.MapperPayloadExtract(existing)
			newPayload := util.MapperPayloadBuild(false, existingTokenNo, existingCount+count)
			pogreb.WordMapper.Put([]byte(truncated), newPayload)
		} else {
			// * create new truncated word with original token (safe since original was deleted)
			newPayload := util.MapperPayloadBuild(false, tokenNo, count)
			pogreb.WordMapper.Put([]byte(truncated), newPayload)
			pogreb.TokenMapper.Put(util.Uint64ToBytes(tokenNo), []byte(truncated))
		}
	}
}

// * remove words longer than max length
func removeLongWords(words []wordData, pogreb *pogreb.Pogreb, apply bool) []wordData {
	var filteredWords []wordData
	var longWords []wordData

	// * identify long words
	for _, wd := range words {
		// * skip special words
		if wd.isSpecial {
			filteredWords = append(filteredWords, wd)
			continue
		}

		if len(wd.word) > maxWordLength {
			longWords = append(longWords, wd)
		} else {
			filteredWords = append(filteredWords, wd)
		}
	}

	fmt.Printf("found %d words longer than %d characters\n", len(longWords), maxWordLength)

	if !apply {
		printLongWordsDryRun(longWords)
	}

	// * apply changes to database
	if apply {
		removeWordsFromDatabase(pogreb, longWords)
	}

	return filteredWords
}

// * print long words preview for dry run
func printLongWordsDryRun(longWords []wordData) {
	fmt.Printf("\n=== long words removal (dry run) ===\n")
	fmt.Printf("words to be removed (> %d chars):\n", maxWordLength)

	count := 0
	for _, wd := range longWords {
		if count >= 20 {
			fmt.Printf("  ... and %d more\n", len(longWords)-20)
			break
		}
		fmt.Printf("  '%s' (len=%d)\n", wd.word, len(wd.word))
		count++
	}
	fmt.Printf("\n")
}

// * remove words with low frequency
func removeLowFrequencyWords(words []wordData, pogreb *pogreb.Pogreb, apply bool) []wordData {
	var filteredWords []wordData
	var lowFreqWords []wordData

	// * identify low frequency words
	for _, wd := range words {
		// * skip special words entirely - no modifications allowed
		if wd.isSpecial {
			filteredWords = append(filteredWords, wd)
			continue
		}

		if wd.count < uint64(minWordCount) {
			lowFreqWords = append(lowFreqWords, wd)
		} else {
			filteredWords = append(filteredWords, wd)
		}
	}

	fmt.Printf("found %d words with count < %d\n", len(lowFreqWords), minWordCount)

	if !apply {
		printLowFreqWordsDryRun(lowFreqWords)
	}

	// * apply changes to database
	if apply {
		removeWordsFromDatabase(pogreb, lowFreqWords)
	}

	return filteredWords
}

// * print low frequency words preview for dry run
func printLowFreqWordsDryRun(lowFreqWords []wordData) {
	fmt.Printf("\n=== low frequency words removal (dry run) ===\n")
	fmt.Printf("words to be removed (count < %d):\n", minWordCount)

	count := 0
	for _, wd := range lowFreqWords {
		if count >= 20 {
			fmt.Printf("  ... and %d more\n", len(lowFreqWords)-20)
			break
		}
		fmt.Printf("  '%s' (count=%d)\n", wd.word, wd.count)
		count++
	}
	fmt.Printf("\n")
}

// * remove words from database completely
func removeWordsFromDatabase(pogreb *pogreb.Pogreb, words []wordData) {
	fmt.Printf("removing %d words from database...\n", len(words))

	removedCount := 0
	for _, wd := range words {
		// * delete from word mapper
		err := pogreb.WordMapper.Delete([]byte(wd.word))
		if err != nil {
			log.Printf("warning: failed to delete word '%s': %v", wd.word, err)
			continue
		}

		// * delete from token mapper
		if err := pogreb.TokenMapper.Delete(util.Uint64ToBytes(wd.tokenNo)); err != nil {
			log.Printf("warning: failed to delete token mapping %d: %v", wd.tokenNo, err)
		}

		removedCount++
	}

	fmt.Printf("successfully removed %d words from database\n", removedCount)
}

// * perform subword extraction to reduce vocabulary
func performSubwordExtraction(words []wordData, targetCount int) []extractionStep {
	// * convert to map for easier processing
	wordMap := make(map[string]bool)
	for _, wd := range words {
		wordMap[wd.word] = true
	}

	var steps []extractionStep
	currentWordCount := len(wordMap)

	// * process batches: 7-8, 5-6, 3-4, 2
	batches := []struct{ min, max int }{
		{7, 8}, {5, 6}, {3, 4}, {2, 2},
	}

	for _, batch := range batches {
		if currentWordCount <= targetCount {
			break
		}

		fmt.Printf("\n=== batch %d: length %d-%d subwords ===\n",
			getBatchNumber(batch), batch.min, batch.max)
		fmt.Printf("current vocabulary: %d words (target: %d)\n", currentWordCount, targetCount)

		// * find subword candidates
		candidates := findSubwordCandidates(wordMap, batch.min, batch.max)
		if len(candidates) == 0 {
			fmt.Printf("no candidates found in this batch\n")
			continue
		}

		fmt.Printf("found %d potential subwords in this batch\n", len(candidates))

		// * sort by net reduction
		sort.Slice(candidates, func(i, j int) bool {
			return candidates[i].netReduction > candidates[j].netReduction
		})

		// * process beneficial candidates
		batchWordsProcessed := 0
		for _, candidate := range candidates {
			if currentWordCount <= targetCount {
				fmt.Printf("target reached. stopping.\n")
				break
			}

			if candidate.netReduction <= 0 {
				continue
			}

			// * create extraction step
			step := extractionStep{
				subword:      candidate.text,
				parentWords:  candidate.parentWords[:candidate.wordCount],
				remainWords:  calculateRemainWords(candidate.parentWords[:candidate.wordCount], candidate.text),
				wordsRemoved: candidate.wordCount,
				wordsAdded:   1 + len(uniqueStrings(calculateRemainWords(candidate.parentWords[:candidate.wordCount], candidate.text))),
				netReduction: candidate.netReduction,
			}

			steps = append(steps, step)
			currentWordCount -= step.netReduction
			batchWordsProcessed++

			fmt.Printf("  extracted '%s' (len=%d): removed %d words, added %d words, net reduction: %d (remaining: %d)\n",
				step.subword, len(step.subword), step.wordsRemoved, step.wordsAdded, step.netReduction, currentWordCount)

			// * update word map
			wordMap = updateWordMap(wordMap, &step)
		}

		if batchWordsProcessed > 0 {
			fmt.Printf("batch completed: %d extractions, %d words reduced\n", batchWordsProcessed, len(steps))
		}
	}

	return steps
}

// * get batch number for display
func getBatchNumber(batch struct{ min, max int }) int {
	switch {
	case batch.min == 7:
		return 1
	case batch.min == 5:
		return 2
	case batch.min == 3:
		return 3
	case batch.min == 2:
		return 4
	default:
		return 0
	}
}

// * find subword candidates for extraction
func findSubwordCandidates(words map[string]bool, minLength, maxLength int) []subwordCandidate {
	subwordMap := make(map[string]*subwordCandidate)
	wordList := make([]string, 0, len(words))
	for word := range words {
		wordList = append(wordList, word)
	}

	// * find all possible subwords
	for _, word := range wordList {
		if len(word) <= minLength {
			continue
		}

		for length := minLength; length <= maxLength && length <= len(word)-1; length++ {
			for i := 0; i <= len(word)-length; i++ {
				subword := word[i : i+length]

				if candidate, exists := subwordMap[subword]; exists {
					// * add word to parent list if not already there
					found := false
					for _, parent := range candidate.parentWords {
						if parent == word {
							found = true
							break
						}
					}
					if !found {
						candidate.parentWords = append(candidate.parentWords, word)
						candidate.wordCount++
					}
				} else {
					subwordMap[subword] = &subwordCandidate{
						text:        subword,
						length:      len(subword),
						parentWords: []string{word},
						wordCount:   1,
					}
				}
			}
		}
	}

	// * convert to candidates and calculate net reduction
	var candidates []subwordCandidate
	for _, candidate := range subwordMap {
		if candidate.wordCount >= 2 { // * must be shared by at least 2 words
			remainWords := calculateRemainWords(candidate.parentWords, candidate.text)
			uniqueRemains := uniqueStrings(remainWords)

			wordsRemoved := candidate.wordCount
			wordsAdded := 1 + len(uniqueRemains)
			netReduction := wordsRemoved - wordsAdded

			if netReduction > 0 {
				candidate.netReduction = netReduction
				candidates = append(candidates, *candidate)
			}
		}
	}

	return candidates
}

// * calculate remaining words after subword extraction
func calculateRemainWords(parentWords []string, subword string) []string {
	var remains []string
	for _, word := range parentWords {
		parts := strings.Split(word, subword)
		for _, part := range parts {
			if part != "" {
				remains = append(remains, part)
			}
		}
	}
	return remains
}

// * get unique strings from slice
func uniqueStrings(slice []string) []string {
	unique := make(map[string]bool)
	var result []string
	for _, s := range slice {
		if !unique[s] {
			unique[s] = true
			result = append(result, s)
		}
	}
	return result
}

// * update word map after extraction
func updateWordMap(wordMap map[string]bool, step *extractionStep) map[string]bool {
	// * remove parent words
	for _, word := range step.parentWords {
		delete(wordMap, word)
	}

	// * add subword
	wordMap[step.subword] = true

	// * add remaining words
	for _, remain := range step.remainWords {
		if remain != "" {
			wordMap[remain] = true
		}
	}

	return wordMap
}

// * apply extraction steps to database
func applyExtractionSteps(pogreb *pogreb.Pogreb, constructor constructor.Server, steps []extractionStep) {
	fmt.Printf("\n=== applying extraction steps ===\n")

	currentCount := len(loadAllWords(pogreb))
	totalReduction := 0

	for i, step := range steps {
		fmt.Printf("applying step %d: extract '%s' (len=%d)\n", i+1, step.subword, len(step.subword))

		// * split words by subword
		removedCount := 0
		for _, word := range step.parentWords {
			err := splitWordBySubword(pogreb, constructor, word, step.subword)
			if err != nil {
				log.Printf("error splitting word '%s': %v", word, err)
			} else {
				removedCount++
			}
		}

		currentCount -= step.netReduction
		totalReduction += step.netReduction

		fmt.Printf("split %d words, net reduction: %d (remaining: %d)\n",
			removedCount, step.netReduction, currentCount)

		if currentCount <= targetWords {
			break
		}
	}

	fmt.Printf("\nextraction complete:\n")
	fmt.Printf("total reduction: %d (%.1f%%)\n", totalReduction,
		float64(totalReduction)/float64(len(loadAllWords(pogreb)))*100)
	fmt.Printf("final word count: %d\n", currentCount)
}

// * split word by subword in database
func splitWordBySubword(pogreb *pogreb.Pogreb, constructor constructor.Server, word, subword string) error {
	// * get original word data
	value, err := pogreb.WordMapper.Get([]byte(word))
	if err != nil || value == nil {
		return fmt.Errorf("word not found: %s", word)
	}

	_, tokenNo, count := util.MapperPayloadExtract(value)

	// * delete old mappings
	pogreb.TokenMapper.Delete(util.Uint64ToBytes(tokenNo))
	pogreb.WordMapper.Delete([]byte(word))

	// * get the current highest token and use next available
	highestToken := constructor.GetNum()
	nextAvailableToken := highestToken + 1

	// * split by subword and create new words
	parts := strings.Split(word, subword)
	for i, part := range parts {
		if part == "" {
			continue
		}

		// * check if part already exists
		existing, _ := pogreb.WordMapper.Get([]byte(part))
		if existing != nil {
			// * merge counts
			_, existingTokenNo, existingCount := util.MapperPayloadExtract(existing)
			newPayload := util.MapperPayloadBuild(false, existingTokenNo, existingCount+count)
			pogreb.WordMapper.Put([]byte(part), newPayload)
		} else {
			// * create new word with sequential token from highest+1
			newTokenNo := nextAvailableToken + uint64(i)
			newPayload := util.MapperPayloadBuild(false, newTokenNo, count)
			pogreb.WordMapper.Put([]byte(part), newPayload)
			pogreb.TokenMapper.Put(util.Uint64ToBytes(newTokenNo), []byte(part))
		}
	}

	return nil
}

// * reassign sequential token numbers to eliminate gaps (legacy function)
func reassignSequentialTokens(pogreb *pogreb.Pogreb, apply bool) {
	specialWords, nonSpecialWords := loadAndSeparateWords(pogreb)
	reassignNonSpecialTokensOnly(pogreb, specialWords, nonSpecialWords, apply)
}

// * reassign only non-special tokens, keeping special tokens unchanged
func reassignNonSpecialTokensOnly(pogreb *pogreb.Pogreb, specialWords []wordData, nonSpecialWords []wordData, apply bool) {
	fmt.Printf("\n=== reassigning non-special tokens (preserving %d special tokens) ===\n", len(specialWords))

	// * sort non-special words by original token order
	sort.Slice(nonSpecialWords, func(i, j int) bool {
		return nonSpecialWords[i].tokenNo < nonSpecialWords[j].tokenNo
	})

	// * find next available token number after highest special token
	maxSpecialToken := uint64(0)
	for _, wd := range specialWords {
		if wd.tokenNo > maxSpecialToken {
			maxSpecialToken = wd.tokenNo
		}
	}
	nextAvailableToken := maxSpecialToken

	fmt.Printf("preserving %d special tokens with highest token: %d\n", len(specialWords), maxSpecialToken)
	fmt.Printf("reassigning %d non-special tokens starting from %d\n", len(nonSpecialWords), nextAvailableToken+1)

	if !apply {
		fmt.Printf("dry run: would reassign %d non-special words to positions %d-%d\n",
			len(nonSpecialWords), nextAvailableToken+1, nextAvailableToken+uint64(len(nonSpecialWords)))
		return
	}

	// * delete existing non-special mappings only
	fmt.Printf("clearing non-special mappings...\n")
	for _, wd := range nonSpecialWords {
		pogreb.TokenMapper.Delete(util.Uint64ToBytes(wd.tokenNo))
		pogreb.WordMapper.Delete([]byte(wd.word))
	}

	// * reassign non-special tokens sequentially
	fmt.Printf("reassigning non-special tokens sequentially...\n")
	successCount := 0
	for i, wd := range nonSpecialWords {
		newTokenNo := nextAvailableToken + uint64(i+1)

		// * update word mapper (keep isSpecial=false)
		newPayload := util.MapperPayloadBuild(false, newTokenNo, wd.count)
		if err := pogreb.WordMapper.Put([]byte(wd.word), newPayload); err != nil {
			log.Printf("error updating non-special word '%s': %v", wd.word, err)
			continue
		}

		// * update token mapper
		if err := pogreb.TokenMapper.Put(util.Uint64ToBytes(newTokenNo), []byte(wd.word)); err != nil {
			log.Printf("error updating token mapping %d -> %s: %v", newTokenNo, wd.word, err)
			continue
		}

		successCount++

		// * progress indicator
		if (i+1)%1000 == 0 {
			fmt.Printf("progress: %d/%d non-special words reassigned...\n", i+1, len(nonSpecialWords))
		}
	}

	// * verify special tokens are preserved
	fmt.Printf("\nnon-special token reassignment complete:\n")
	fmt.Printf("- successfully reassigned %d non-special words\n", successCount)
	fmt.Printf("- preserved %d special tokens unchanged\n", len(specialWords))

	// * verify some special tokens
	for _, wd := range specialWords {
		if wd.word == "." || wd.word == "..." || len(wd.word) == 1 && !((wd.word[0] >= 'a' && wd.word[0] <= 'z') || (wd.word[0] >= 'A' && wd.word[0] <= 'Z')) {
			fmt.Printf("- preserved special word '%s' with token %d\n", wd.word, wd.tokenNo)
		}
	}
}

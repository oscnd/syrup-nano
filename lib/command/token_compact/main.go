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
	"go.scnd.dev/open/syrup/nano/lib/util"
	"go.uber.org/fx"
)

const (
	targetWords   = 100000
	maxWordLength = 16
	minWordCount  = 10
)

// SubwordCandidate represents a potential subword for extraction
type SubwordCandidate struct {
	Text         string
	Length       int
	ParentWords  []string
	WordCount    int
	NetReduction int // Words removed - (new subword + new remains)
	Benefit      int
}

// ExtractionStep represents one subword extraction step
type ExtractionStep struct {
	Subword      string
	ParentWords  []string
	RemainWords  []string
	WordsRemoved int
	WordsAdded   int
	NetReduction int
}

func main() {
	var apply = flag.Bool("apply", false, "apply changes to database instead of dry run")
	flag.Parse()

	fx.New(
		fxo.Option(),
		fx.Provide(
			config.Init,
			pogreb.Init,
		),
		fx.Invoke(func(shutdowner fx.Shutdowner, pogreb *pogreb.Pogreb) {
			CompactTokens(pogreb, *apply)
			_ = shutdowner.Shutdown()
		}),
	).Run()
}

func CompactTokens(p *pogreb.Pogreb, apply bool) {
	// Load all tokens
	tokens := loadAllTokens(p)
	fmt.Printf("Loaded %d tokens\n", len(tokens))

	if len(tokens) <= targetWords {
		fmt.Printf("Already at target (%d tokens)\n", len(tokens))
		return
	}

	// Step 1: Apply character truncation
	truncatedTokens, truncationCount := applyCharacterTruncation(tokens, p, apply)
	fmt.Printf("Character truncation reduced %d tokens\n", truncationCount)

	// Step 2: Remove long words
	filteredTokens, longWordCount := removeLongWords(truncatedTokens, p, apply)
	fmt.Printf("Removed %d words longer than %d characters\n", longWordCount, maxWordLength)

	// Step 3: Remove low-frequency words
	finalTokens, lowFreqCount := removeLowFrequencyWords(filteredTokens, p, apply)
	fmt.Printf("Removed %d words with count < %d\n", lowFreqCount, minWordCount)

	currentWordCount := len(finalTokens)
	if currentWordCount <= targetWords {
		fmt.Printf("Reached target after initial cleanup (%d tokens)\n", currentWordCount)
		return
	}

	// Step 4: Iterative subword extraction and re-analysis
	steps := iterativeSubwordExtraction(finalTokens, currentWordCount)

	// Step 5: Apply extraction steps
	if apply {
		applyExtractionSteps(p, finalTokens, steps)
	} else {
		printExtractionSteps(finalTokens, steps)
	}
}

func loadAllTokens(p *pogreb.Pogreb) []string {
	var tokens []string
	it := p.WordMapper.Items()
	for {
		key, _, err := it.Next()
		if err != nil {
			break
		}
		tokens = append(tokens, string(key))
	}
	return tokens
}

func applyCharacterTruncation(tokens []string, p *pogreb.Pogreb, apply bool) ([]string, int) {
	truncationMap := make(map[string]string)
	var result []string

	for _, word := range tokens {
		truncated := truncateConsecutiveChars(word)
		if truncated != word {
			truncationMap[word] = truncated
		} else {
			result = append(result, word)
		}
	}

	fmt.Printf("Found %d words with 3+ consecutive characters\n", len(truncationMap))

	if !apply {
		printTruncationDryRun(truncationMap)
	}

	for _, truncated := range truncationMap {
		result = append(result, truncated)
	}

	if apply {
		applyTruncationToDatabase(p, truncationMap)
	}

	return result, len(truncationMap)
}

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

func printTruncationDryRun(truncationMap map[string]string) {
	fmt.Printf("\n=== CHARACTER TRUNCATION (DRY RUN) ===\n")
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

func applyTruncationToDatabase(p *pogreb.Pogreb, truncationMap map[string]string) {
	for original, truncated := range truncationMap {
		value, err := p.WordMapper.Get([]byte(original))
		if err != nil || value == nil {
			continue
		}

		_, tokenNo, count := util.MapperPayloadExtract(value)
		p.WordMapper.Delete([]byte(original))

		existing, _ := p.WordMapper.Get([]byte(truncated))
		if existing != nil {
			_, existingTokenNo, existingCount := util.MapperPayloadExtract(existing)
			newPayload := util.MapperPayloadBuild(false, existingTokenNo, existingCount+count)
			p.WordMapper.Put([]byte(truncated), newPayload)
		} else {
			payload := util.MapperPayloadBuild(false, tokenNo, count)
			p.WordMapper.Put([]byte(truncated), payload)
		}
	}
}

func removeLongWords(tokens []string, p *pogreb.Pogreb, apply bool) ([]string, int) {
	var filteredTokens []string
	var longWords []string

	for _, word := range tokens {
		if len(word) > maxWordLength {
			longWords = append(longWords, word)
		} else {
			filteredTokens = append(filteredTokens, word)
		}
	}

	fmt.Printf("Found %d words longer than %d characters\n", len(longWords), maxWordLength)

	if !apply {
		printLongWordsDryRun(longWords)
	}

	if apply {
		removeLongWordsFromDatabase(p, longWords)
	}

	return filteredTokens, len(longWords)
}

func printLongWordsDryRun(longWords []string) {
	fmt.Printf("\n=== LONG WORDS REMOVAL (DRY RUN) ===\n")
	fmt.Printf("Words to be removed (> %d chars):\n", maxWordLength)

	count := 0
	for _, word := range longWords {
		if count >= 20 {
			fmt.Printf("  ... and %d more\n", len(longWords)-20)
			break
		}
		fmt.Printf("  '%s' (len=%d)\n", word, len(word))
		count++
	}
	fmt.Printf("\n")
}

func removeLongWordsFromDatabase(p *pogreb.Pogreb, longWords []string) {
	fmt.Printf("Removing %d long words from database...\n", len(longWords))

	removedCount := 0
	for _, word := range longWords {
		err := p.WordMapper.Delete([]byte(word))
		if err != nil {
			log.Printf("Warning: failed to delete long word '%s': %v\n", word, err)
		} else {
			removedCount++
		}
	}

	fmt.Printf("Successfully removed %d long words from database\n", removedCount)
}

func removeLowFrequencyWords(tokens []string, p *pogreb.Pogreb, apply bool) ([]string, int) {
	var filteredTokens []string
	var lowFreqWords []string
	var lowFreqWordsWithCount []struct {
		word  string
		count int
	}

	for _, word := range tokens {
		value, err := p.WordMapper.Get([]byte(word))
		if err != nil || value == nil {
			continue
		}

		_, _, count := util.MapperPayloadExtract(value)
		if count < uint64(minWordCount) {
			lowFreqWords = append(lowFreqWords, word)
			lowFreqWordsWithCount = append(lowFreqWordsWithCount, struct {
				word  string
				count int
			}{word, int(count)})
		} else {
			filteredTokens = append(filteredTokens, word)
		}
	}

	fmt.Printf("Found %d words with count < %d\n", len(lowFreqWords), minWordCount)

	if !apply {
		printLowFreqWordsDryRun(lowFreqWordsWithCount)
	}

	if apply {
		removeLowFreqWordsFromDatabase(p, lowFreqWords)
	}

	return filteredTokens, len(lowFreqWords)
}

func printLowFreqWordsDryRun(lowFreqWords []struct {
	word  string
	count int
}) {
	fmt.Printf("\n=== LOW FREQUENCY WORDS REMOVAL (DRY RUN) ===\n")
	fmt.Printf("Words to be removed (count < %d):\n", minWordCount)

	count := 0
	for _, item := range lowFreqWords {
		if count >= 20 {
			fmt.Printf("  ... and %d more\n", len(lowFreqWords)-20)
			break
		}
		fmt.Printf("  '%s' (count=%d)\n", item.word, item.count)
		count++
	}
	fmt.Printf("\n")
}

func removeLowFreqWordsFromDatabase(p *pogreb.Pogreb, lowFreqWords []string) {
	fmt.Printf("Removing %d low-frequency words from database...\n", len(lowFreqWords))

	removedCount := 0
	for _, word := range lowFreqWords {
		err := p.WordMapper.Delete([]byte(word))
		if err != nil {
			log.Printf("Warning: failed to delete low-frequency word '%s': %v\n", word, err)
		} else {
			removedCount++
		}
	}

	fmt.Printf("Successfully removed %d low-frequency words from database\n", removedCount)
}

func iterativeSubwordExtraction(initialTokens []string, initialCount int) []ExtractionStep {
	var steps []ExtractionStep
	currentTokens := make(map[string]bool)
	for _, token := range initialTokens {
		currentTokens[token] = true
	}

	currentWordCount := len(currentTokens)
	batchNum := 0

	// Process batches sequentially: 7-8, then 5-6, then 3-4, then 2
	batches := []struct{ min, max int }{
		{7, 8}, {5, 6}, {3, 4}, {2, 2},
	}

	for currentWordCount > targetWords && batchNum < len(batches) {
		batch := batches[batchNum]
		fmt.Printf("\n=== BATCH %d: LENGTH %d-%d SUBWORDS ===\n", batchNum+1, batch.min, batch.max)
		fmt.Printf("Current vocabulary: %d words (Target: %d)\n", currentWordCount, targetWords)

		// Find all subwords in this batch
		candidates := findSubwordCandidates(currentTokens, batch.min, batch.max)
		if len(candidates) == 0 {
			fmt.Printf("No candidates found in this batch\n")
			batchNum++
			continue
		}

		fmt.Printf("Found %d potential subwords in this batch\n", len(candidates))

		// Sort by net reduction (best first)
		sort.Slice(candidates, func(i, j int) bool {
			return candidates[i].NetReduction > candidates[j].NetReduction
		})

		// Process all beneficial candidates in this batch
		batchWordsProcessed := 0
		for _, candidate := range candidates {
			if currentWordCount <= targetWords {
				fmt.Printf("Target reached. Stopping.\n")
				break
			}

			if candidate.NetReduction <= 0 {
				continue // Skip non-beneficial extractions
			}

			// Don't exceed target
			if currentWordCount-candidate.NetReduction < targetWords {
				// Adjust to hit target exactly
				maxRemovable := currentWordCount - targetWords
				if maxRemovable < candidate.NetReduction {
					// Calculate how many words we can remove
					maxWordsToRemove := (maxRemovable * candidate.WordCount) / candidate.NetReduction
					if maxWordsToRemove < 1 {
						continue
					}
					candidate.WordCount = maxWordsToRemove
					candidate.NetReduction = maxWordsToRemove - (1 + len(uniqueStrings(calculateRemainWords(candidate.ParentWords[:maxWordsToRemove], candidate.Text))))
				} else {
					continue
				}
			}

			// Create extraction step
			step := ExtractionStep{
				Subword:      candidate.Text,
				ParentWords:  candidate.ParentWords[:candidate.WordCount],
				RemainWords:  calculateRemainWords(candidate.ParentWords[:candidate.WordCount], candidate.Text),
				WordsRemoved: candidate.WordCount,
				WordsAdded:   1 + len(uniqueStrings(calculateRemainWords(candidate.ParentWords[:candidate.WordCount], candidate.Text))),
				NetReduction: candidate.NetReduction,
			}

			// Apply the extraction
			steps = append(steps, step)
			currentWordCount -= step.NetReduction
			batchWordsProcessed++

			fmt.Printf("  Extracted '%s' (len=%d): removed %d words, added %d words, net reduction: %d (remaining: %d)\n",
				step.Subword, len(step.Subword), step.WordsRemoved, step.WordsAdded, step.NetReduction, currentWordCount)

			// Update current tokens for next extraction
			currentTokens = updateTokenSet(currentTokens, &step)

			// Re-calculate remaining candidates as vocabulary has changed
			if batchWordsProcessed < len(candidates) {
				// Re-analyze remaining candidates with updated vocabulary
				remainingCandidates := candidates[batchWordsProcessed:]
				for i := range remainingCandidates {
					// Skip if this subword's parent words were affected
					affected := false
					for _, parent := range step.ParentWords {
						for _, remaining := range remainingCandidates[i].ParentWords {
							if remaining == parent {
								affected = true
								break
							}
						}
						if affected {
							break
						}
					}

					if !affected {
						// Re-calculate with current vocabulary
						splitableWords := findSplitableWordsInSet(currentTokens, remainingCandidates[i].Text)
						if len(splitableWords) >= 2 {
							remainWords := calculateRemainWords(splitableWords, remainingCandidates[i].Text)
							uniqueRemains := uniqueStrings(remainWords)

							wordsRemoved := len(splitableWords)
							wordsAdded := 1 + len(uniqueRemains)
							netReduction := wordsRemoved - wordsAdded

							if netReduction > 0 {
								remainingCandidates[i].WordCount = wordsRemoved
								remainingCandidates[i].NetReduction = netReduction
							} else {
								remainingCandidates[i].NetReduction = 0
							}
						} else {
							remainingCandidates[i].NetReduction = 0
						}
					}
				}

				// Re-sort and continue with updated candidates
				candidates = remainingCandidates
				sort.Slice(candidates, func(i, j int) bool {
					return candidates[i].NetReduction > candidates[j].NetReduction
				})
			}
		}

		if batchWordsProcessed > 0 {
			fmt.Printf("Batch %d completed: %d extractions, %d words reduced\n", batchNum+1, batchWordsProcessed, len(steps))
		}

		batchNum++
	}

	return steps
}

func findSubwordCandidates(tokens map[string]bool, minLength, maxLength int) []SubwordCandidate {
	subwordMap := make(map[string]*SubwordCandidate)
	tokenList := make([]string, 0, len(tokens))
	for token := range tokens {
		tokenList = append(tokenList, token)
	}

	// Find all possible subwords
	for _, word := range tokenList {
		if len(word) <= minLength {
			continue
		}

		for length := minLength; length <= maxLength && length <= len(word)-1; length++ {
			for i := 0; i <= len(word)-length; i++ {
				subword := word[i : i+length]

				if candidate, exists := subwordMap[subword]; exists {
					// Add this word to parent list if not already there
					found := false
					for _, parent := range candidate.ParentWords {
						if parent == word {
							found = true
							break
						}
					}
					if !found {
						candidate.ParentWords = append(candidate.ParentWords, word)
						candidate.WordCount++
					}
				} else {
					subwordMap[subword] = &SubwordCandidate{
						Text:        subword,
						Length:      len(subword),
						ParentWords: []string{word},
						WordCount:   1,
					}
				}
			}
		}
	}

	// Convert to candidates and calculate net reduction
	var candidates []SubwordCandidate
	for _, candidate := range subwordMap {
		if candidate.WordCount >= 2 { // Must be shared by at least 2 words
			// Calculate net reduction
			remainWords := calculateRemainWords(candidate.ParentWords, candidate.Text)
			uniqueRemains := uniqueStrings(remainWords)

			wordsRemoved := candidate.WordCount
			wordsAdded := 1 + len(uniqueRemains)
			netReduction := wordsRemoved - wordsAdded

			if netReduction > 0 {
				candidate.NetReduction = netReduction
				candidate.Benefit = netReduction * candidate.Length // Weight by length
				candidates = append(candidates, *candidate)
			}
		}
	}

	return candidates
}

func findSplitableWordsInSet(tokens map[string]bool, subword string) []string {
	var splitableWords []string

	for word := range tokens {
		if len(word) > len(subword) && strings.Contains(word, subword) {
			// Check if this word can be split by the subword
			parts := strings.Split(word, subword)
			// Ensure we have at least some content besides just the subword
			hasValidParts := false
			for _, part := range parts {
				if part != "" {
					hasValidParts = true
					break
				}
			}
			if hasValidParts {
				splitableWords = append(splitableWords, word)
			}
		}
	}

	return splitableWords
}

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

func updateTokenSet(currentTokens map[string]bool, step *ExtractionStep) map[string]bool {
	// Remove parent words
	for _, word := range step.ParentWords {
		delete(currentTokens, word)
	}

	// Add the new subword
	currentTokens[step.Subword] = true

	// Add remaining words
	for _, remain := range step.RemainWords {
		if remain != "" {
			currentTokens[remain] = true
		}
	}

	return currentTokens
}

func printExtractionSteps(initialTokens []string, steps []ExtractionStep) {
	fmt.Printf("\n=== EXTRACTION STEPS (DRY RUN) ===\n")
	fmt.Printf("Initial word count: %d\n", len(initialTokens))
	fmt.Printf("Target word count: %d\n\n", targetWords)

	currentCount := len(initialTokens)
	totalReduction := 0

	for i, step := range steps {
		currentCount -= step.NetReduction
		totalReduction += step.NetReduction

		fmt.Printf("Step %d: Extract '%s' (len=%d)\n", i+1, step.Subword, len(step.Subword))
		fmt.Printf("  Parent words: %s\n", strings.Join(step.ParentWords[:min(5, len(step.ParentWords))], ", "))
		if len(step.ParentWords) > 5 {
			fmt.Printf("    ... and %d more\n", len(step.ParentWords)-5)
		}
		fmt.Printf("  Words removed: %d\n", step.WordsRemoved)
		fmt.Printf("  Words added: %d (1 subword + %d remains)\n", step.WordsAdded, step.WordsAdded-1)
		fmt.Printf("  Net reduction: %d\n", step.NetReduction)
		fmt.Printf("  Running total: %d words\n", currentCount)
		fmt.Printf("\n")
	}

	fmt.Printf("=== FINAL SUMMARY ===\n")
	fmt.Printf("Total reduction: %d (%.1f%%)\n", totalReduction,
		float64(totalReduction)/float64(len(initialTokens))*100)
	fmt.Printf("Final word count: %d\n", currentCount)
	fmt.Printf("Target reached: %t\n", currentCount <= targetWords)
}

func applyExtractionSteps(p *pogreb.Pogreb, initialTokens []string, steps []ExtractionStep) {
	fmt.Printf("\n=== APPLYING EXTRACTION STEPS ===\n")

	currentCount := len(initialTokens)
	totalReduction := 0

	for i, step := range steps {
		fmt.Printf("Applying step %d: Extract '%s' (len=%d)\n", i+1, step.Subword, len(step.Subword))

		// Apply the extraction
		removedCount := 0
		for _, word := range step.ParentWords {
			err := splitWordBySubword(p, word, step.Subword)
			if err != nil {
				log.Printf("Error splitting word '%s': %v\n", word, err)
			} else {
				removedCount++
			}
		}

		currentCount -= step.NetReduction
		totalReduction += step.NetReduction

		fmt.Printf("Split %d words, net reduction: %d (remaining: %d)\n",
			removedCount, step.NetReduction, currentCount)

		if currentCount <= targetWords {
			break
		}
	}

	fmt.Printf("\nExtraction complete:\n")
	fmt.Printf("Total reduction: %d (%.1f%%)\n", totalReduction,
		float64(totalReduction)/float64(len(initialTokens))*100)
	fmt.Printf("Final word count: %d\n", currentCount)
}

func splitWordBySubword(p *pogreb.Pogreb, word, subword string) error {
	value, err := p.WordMapper.Get([]byte(word))
	if err != nil || value == nil {
		return fmt.Errorf("word not found: %s", word)
	}

	_, tokenNo, count := util.MapperPayloadExtract(value)
	err = p.WordMapper.Delete([]byte(word))
	if err != nil {
		return err
	}

	// Split by subword
	parts := strings.Split(word, subword)

	// Add each part to database
	for i, part := range parts {
		if part == "" {
			continue
		}

		existing, _ := p.WordMapper.Get([]byte(part))
		if existing != nil {
			_, existingTokenNo, existingCount := util.MapperPayloadExtract(existing)
			newPayload := util.MapperPayloadBuild(false, existingTokenNo, existingCount+count)
			p.WordMapper.Put([]byte(part), newPayload)
		} else {
			payload := util.MapperPayloadBuild(false, tokenNo+uint64(i), count)
			p.WordMapper.Put([]byte(part), payload)
		}
	}

	return nil
}

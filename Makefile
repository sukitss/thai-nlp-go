.DEFAULT_GOAL := help
GO      ?= go
PKG     := ./...
BIN     := thainlp
CMD     := ./cmd/thainlp

.PHONY: help build install test test-short test-race cover bench fuzz dict vet fmt tidy ci clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

build: ## Build the CLI to ./$(BIN)
	$(GO) build -o $(BIN) $(CMD)

install: ## go install the CLI
	$(GO) install $(CMD)

test: ## Run the full test suite (golden vs PyThaiNLP + 200k fuzz)
	$(GO) test $(PKG)

test-short: ## Run tests with the reduced fuzz count
	$(GO) test -short $(PKG)

test-race: ## Run tests with the race detector
	$(GO) test -race -short $(PKG)

cover: ## Generate coverage.html
	$(GO) test -coverprofile=coverage.out $(PKG)
	$(GO) tool cover -html=coverage.out -o coverage.html
	@echo "wrote coverage.html"

bench: ## Run benchmarks
	$(GO) test -run=^$$ -bench=. -benchmem $(PKG)

report: ## Generate docs/TEST-REPORT.md (test + benchmark snapshot for tracking)
	$(GO) run scripts/genreport.go

fuzz: ## Run the native fuzzer for 30s
	$(GO) test -run=^$$ -fuzz=FuzzSegment -fuzztime=30s ./tokenize/

dict: ## Rebuild the embedded flat-trie from dict/data/words_th.txt
	$(GO) run $(CMD) build -dict dict/data/words_th.txt -out dict/data/words_th.fdt

vet: ## go vet
	$(GO) vet $(PKG)

fmt: ## gofmt -w on all sources
	gofmt -w .

tidy: ## go mod tidy
	$(GO) mod tidy

ci: vet test ## What CI runs: vet + test

clean: ## Remove build artifacts
	rm -f $(BIN) coverage.out coverage.html

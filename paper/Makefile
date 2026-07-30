SHELL     := /bin/bash
TEXBIN    := /home/mabdel03/.local/bin
PY        := /orcd/home/002/mabdel03/conda_envs/asys_analysis/bin/python
export PATH := $(TEXBIN):$(PATH)

REPO      := $(realpath ..)
CACHE     := $(REPO)/analysis/cache
CSVS      := $(wildcard $(REPO)/analysis/tables/*/*.csv)
PARQUETS  := $(CACHE)/analysis_view_v1.parquet $(CACHE)/cells_dedup_v1.parquet \
             $(CACHE)/items_v1.parquet $(CACHE)/agents_v1.parquet
SVGS      := $(wildcard $(REPO)/docs/figures/exported/*.svg)
ATLAS     := $(wildcard $(REPO)/analysis/figures/*.png) \
             $(wildcard $(REPO)/analysis/figures/*/*.png)
TOOLS     := tools/paperdata.py tools/texlib.py
SRCS      := main.tex preamble.tex refs.bib \
             $(wildcard sections/*.tex) $(wildcard sections/appendix/*.tex)

REQ_STY   := booktabs longtable siunitx rotating subcaption adjustbox \
             multirow tabularx enumitem float microtype natbib hyperref geometry \
             amsmath amssymb graphicx caption xcolor mathptmx

.PHONY: all draft tables figures schematics atlas pdf clean distclean check logcheck

all: check tables figures schematics atlas pdf logcheck

check:
	@command -v pdflatex >/dev/null || { echo "FAIL: no pdflatex on PATH"; exit 1; }
	@pdflatex --version | head -1 | grep -q 'TeX Live 2026' \
	  || { echo "FAIL: wrong pdflatex ($$(command -v pdflatex)); expected TinyTeX TeX Live 2026"; exit 1; }
	@test -x $(PY) || { echo "FAIL: missing python at $(PY)"; exit 1; }
	@for p in $(PARQUETS); do test -f $$p || { echo "FAIL: missing cache $$p"; exit 1; }; done
	@for s in $(REQ_STY); do kpsewhich $$s.sty >/dev/null \
	  || { echo "FAIL: missing $$s.sty"; exit 1; }; done
	@echo "toolchain OK: $$(pdflatex --version | head -1)"

tables: .stamp-tables
.stamp-tables: tools/build_tables.py $(TOOLS) $(CSVS) $(PARQUETS)
	$(PY) tools/build_tables.py --out tables
	@touch $@

figures: .stamp-figures
.stamp-figures: tools/build_figures.py $(TOOLS) $(PARQUETS) $(CSVS)
	$(PY) tools/build_figures.py --out figures/panels
	@touch $@

schematics: .stamp-schematics
.stamp-schematics: tools/convert_schematics.py $(SVGS)
	$(PY) tools/convert_schematics.py --out figures/schematics --scale 3 --workdir _build
	@touch $@

atlas: .stamp-atlas
.stamp-atlas: tools/stage_atlas.py $(ATLAS)
	$(PY) tools/stage_atlas.py --src $(REPO)/analysis/figures \
	      --out figures/atlas --tex sections/appendix/E_atlas.tex
	@touch $@

pdf: $(SRCS) .stamp-tables .stamp-figures .stamp-schematics .stamp-atlas
	latexmk -pdf -bibtex main.tex
	@echo "==> $(CURDIR)/_build/main.pdf ($$(du -h _build/main.pdf | cut -f1))"

# Fast iteration: skips the ~2100-row stratified appendix and the 59-figure atlas.
draft: $(SRCS) .stamp-tables .stamp-figures
	latexmk -pdf -bibtex -jobname=draft \
	  -pretex='\def\draftbuild{1}' -usepretex main.tex
	@echo "==> $(CURDIR)/_build/draft.pdf"

logcheck:
	@$(PY) tools/check_log.py _build/main.log

clean:
	-latexmk -c main.tex
	rm -f .stamp-*

distclean:
	-latexmk -C main.tex
	rm -rf _build tables figures/panels figures/atlas figures/schematics \
	       sections/appendix/E_atlas.tex .stamp-*

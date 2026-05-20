# Minimal sphinx Makefile.  Use "make html" to build.
SPHINXOPTS  ?=
SPHINXBUILD ?= ../venv/bin/sphinx-build
SOURCEDIR   = .
BUILDDIR    = _build

.PHONY: help clean html

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

clean:
	rm -rf $(BUILDDIR)

html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	@echo
	@echo "HTML docs: $(BUILDDIR)/html/index.html"

all: journal.tex.pdf

journal.tex.pdf: journal.tex
	pdflatex -shell-escape journal.tex

.PHONY: clean
clean:
	$(RM) journal.pdf \
journal.aux \
journal.bbl \
journal.blg \
journal.log \
journal.out \
journal.pdf \
journal.tex~
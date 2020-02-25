all: chameleon.tex.pdf

chameleon.tex.pdf: chameleon.tex
	pdflatex -shell-escape chameleon.tex

.PHONY: clean
clean:
	$(RM) chameleon.pdf \
chameleon.aux \
chameleon.bbl \
chameleon.blg \
chameleon.log \
chameleon.out \
chameleon.pdf \
chameleon.tex~
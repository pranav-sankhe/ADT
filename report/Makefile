all: main.pdf

main.pdf: Thesis.tex
	pdflatex -shell-escape Thesis.tex

.PHONY: clean
clean:
	$(RM) main.pdf \
main.aux \
main.bbl \
main.blg \
main.log \
main.out \
main.pdf \
main.tex~
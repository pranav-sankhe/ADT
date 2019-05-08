all: main.pdf

main.pdf: main.tex
	pdflatex -shell-escape main.tex

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

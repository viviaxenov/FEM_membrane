DOC_NAME = lab_template
PLOT_NAMES = graph plot
PLOTS = $(addsuffix .png, $(PLOT_NAMES))
# SCRIPTS = $(addsuffix .py, $(PLOT_NAMES))

$(DOC_NAME).pdf : $(PLOTS)
	pdflatex ../tex/$(DOC_NAME).tex
$(PLOTS) :
	python ../script/$(basename $@).py 
clean :
	mv $(DOC_NAME).pdf -t ../
	 	

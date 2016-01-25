name = practical-cnn-reg
DST = vgg@login.robots.ox.ac.uk:WWW/share
DSTDOC = vgg@login.robots.ox.ac.uk:WWW/practicals/cnn-reg

.PHONY: prepack, pack, pack-data, pack-code, post, clean, distclean

pack-all: pack-data pack-code pack

code=\
exercise2.m \
setup.m \
showDeblurringResult.m \
xavier.m \
matconvnet

doc=\
doc/images \
doc/instructions.html

data=\
data/text_imdb.mat

tarflags=--exclude='.git*'

code:=$(addprefix $(CURDIR)/,$(code))
data:=$(addprefix $(CURDIR)/,$(data))
doc:=$(addprefix $(CURDIR)/,$(doc))
deps:=$(shell find $(code) $(doc) $(data) -type f | sed "s/ /\\\\ /g")

pack: $(TMPDIR)/$(name).tar.gz
pack-data: $(TMPDIR)/$(name)-data-only.tar.gz
pack-code: $(TMPDIR)/$(name)-code-only.tar.gz

$(TMPDIR)/$(name).tar.gz: $(deps)
	rm -rf $(TMPDIR)/$(name)
	mkdir -p $(TMPDIR)/$(name)/{doc,data}
	ln -sf $(data) $(TMPDIR)/$(name)/data/
	ln -sf $(doc) $(TMPDIR)/$(name)/doc/
	ln -sf $(code) $(TMPDIR)/$(name)/
	tar -C $(TMPDIR) -czvhf $(TMPDIR)/$(name).tar.gz $(tarflags) $(name)/

$(TMPDIR)/$(name)-data-only.tar.gz: $(deps)
	rm -rf $(TMPDIR)/$(name)
	mkdir -p $(TMPDIR)/$(name)/{doc,data}
	ln -sf $(data) $(TMPDIR)/$(name)/
	tar -C $(TMPDIR) -czvhf $(TMPDIR)/$(name)-data-only.tar.gz $(tarflags) $(name)/

$(TMPDIR)/$(name)-code-only.tar.gz: $(deps)
	rm -rf $(TMPDIR)/$(name)
	mkdir -p $(TMPDIR)/$(name)/{doc,data}
	ln -sf $(doc) $(TMPDIR)/$(name)/doc/
	ln -sf $(code) $(TMPDIR)/$(name)/
	tar -C $(TMPDIR) -czvhf $(TMPDIR)/$(name)-code-only.tar.gz $(tarflags) $(name)/

doc/instructions.html : doc/instructions.md doc/base.html extra/Makefile
	cp doc/base.html "$@"
	python -m markdown \
	  -x toc -x footnotes -x tables -x fenced_code -x attr_list \
	  "$<" >> "$@"
	cat doc/end.html >> "$@"

post-doc: doc/instructions.html
	rsync -rvt doc/images $(DSTDOC)/
	rsync -vt doc/instructions.html $(DSTDOC)/index.html

post: pack-all
	rsync -vt $(TMPDIR)/$(name).tar.gz $(DST)/
	rsync -vt $(TMPDIR)/$(name)-data-only.tar.gz $(DST)/
	rsync -vt $(TMPDIR)/$(name)-code-only.tar.gz $(DST)/

clean:
	find . -name '*~' -delete

distclean: clean
	rm -f $(TMPDIR)/$(name)*.tar.gz

# Preprocessing

papers_url=http://www.robots.ox.ac.uk/~vedaldi/assets/pubs
papers=\
blaschko10simultaneous.pdf \
chatfield11devil.pdf \
chatfield14return.pdf \
cimpoi14describing.pdf \
cimpoi15deep.pdf \
cimpoi16deep.pdf \
fulkerson08localizing.pdf \
fulkerson09class.pdf \
jaderberg14deep.pdf \
jaderberg14speeding.pdf \
jaderberg14synthetic.pdf \
jaderberg15reading.pdf \
jones07inertial.pdf \
juneja13blocks.pdf \
lempitsky11pylon.pdf \
lenc15rcnn.pdf \
lenc15understanding.pdf \
mahendran15understanding.pdf \
parizi15automatic.pdf \
parkhi11truth.pdf \
parkhi12cat.pdf \
parkhi12spotting.pdf \
parkhi14compact.pdf \
parkhi15deep.pdf \
pedersoli11coarse.pdf \
pedersoli14coarse.pdf \
rabinovich07objects.pdf \
simonyan12descriptor.pdf \
simonyan13deep.pdf \
simonyan13fisher.pdf \
simonyan14deep.pdf \
simonyan14learning.pdf \
sreekanth10generalized.pdf \
vedaldi05features.pdf \
vedaldi05kalmansac.pdf \
vedaldi05TRviewpoint.pdf \
vedaldi06local.pdf \
vedaldi06viewpoint.pdf \
vedaldi07boosting.pdf \
vedaldi07complexity.pdf \
vedaldi07moving-orig.pdf \
vedaldi07moving.pdf \
vedaldi07open.pdf \
vedaldi08invariant.pdf \
vedaldi08joint.pdf \
vedaldi08quick.pdf \
vedaldi08relaxed.pdf \
vedaldi09multiple.pdf \
vedaldi09structured.pdf \
vedaldi10efficient.pdf \
vedaldi10knowing.pdf \
vedaldi10vlfeat.pdf \
vedaldi10vlfeat-tutorial.pdf \
vedaldi11efficient.pdf \
vedaldi11learning.pdf \
vedaldi12self.pdf \
vedaldi12sparse.pdf \
vedaldi14understanding.pdf \
vedaldi15matconvnet.pdf

papers:=$(addprefix data/text/,$(papers))
papers_crops=$(papers:%.pdf=%.png)

preproc: $(papers) $(papers_crops)

data/text/%.pdf:
	wget "$(papers_url)/$(*).pdf" -O "$(@)"

%.png : %.pdf extra/Makefile
	convert \
	-verbose \
	-units PixelsPerInch \
	-density 300 \
	"$(<)"[2] \
	-colorspace 'rgb' \
	-flatten \
	-resize 'x1536' \
	-gravity center -crop 512x512+0+0 \
	"$(@)"

info:
	@echo Images: "$(papers)"

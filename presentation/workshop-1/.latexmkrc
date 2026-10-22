# The auriga theme uses fontspec, which needs LuaLaTeX. Overriding $pdflatex
# (rather than $pdf_mode) also covers callers that pass -pdf, e.g. LaTeX Workshop.
$pdflatex = 'lualatex %O %S';

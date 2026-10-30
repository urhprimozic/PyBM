# The auriga theme uses fontspec, which needs LuaLaTeX. Overriding $pdflatex
# (rather than $pdf_mode) also covers callers that pass -pdf, e.g. LaTeX Workshop.
$pdflatex = 'lualatex %O %S';

# minted: TeX Live's latexminted 0.6.0 crashes on Python 3.14. A copy with the fix (backported from 0.7.0) is in
# ~/.local/share/latexminted, linked from ~/.local/bin, which has to come first on the PATH lualatex sees.
$ENV{PATH} = "$ENV{HOME}/.local/bin:$ENV{PATH}";

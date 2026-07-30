$pdf_mode = 1;
$pdflatex = 'pdflatex -interaction=nonstopmode -file-line-error -synctex=1 %O %S';
$bibtex_use = 2;
$out_dir = '_build';
$aux_dir = '_build';

# ~20 longtables write their column widths to the .aux and request a rerun on the
# first pass; the default of 5 repeats is tight once landscape appendices are on.
$max_repeat = 8;

$clean_ext = 'synctex.gz run.xml bbl fdb_latexmk fls';

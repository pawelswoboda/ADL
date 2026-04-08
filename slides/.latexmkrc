$pdf_mode = 1;            # Use pdflatex
$pdflatex = 'pdflatex -interaction=nonstopmode -synctex=1 %O %S';

# Auto-recover from corrupt .bcf files
$biber = 'biber %O %S || (rm -f %R.bcf %R.bbl %R.aux %R.run.xml && false)';

$max_repeat = 5;

#!/bin/bash

if [ "$QUARTO_PROJECT_RENDER_ALL" -ne 1 ]; then
  exit 1
fi

cd examples
rm *.qmd

for file in *.R; do
  echo "Rendering $file"
  Rscript -e "rmarkdown::render('_template.Rmd', params = list(file = '$file'), output_file = '$file.qmd')"
done
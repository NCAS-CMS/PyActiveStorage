#!/bin/bash
for f in *.pu
do
  echo "processing $f"
  java -jar ~/Applications/plantuml.jar -tsvg $f
done

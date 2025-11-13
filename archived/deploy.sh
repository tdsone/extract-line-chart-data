if [ "$1" == "cd" ]
then
    modal deploy plextract/run_chartdete.py
elif [ "$1" == "lf" ]
then
    modal deploy plextract/run_lineformer.py
elif [ "$1" == "ocr" ]
then
    modal deploy plextract/run_ocr.py
else
    modal deploy plextract/run_chartdete.py
    modal deploy plextract/run_lineformer.py
    modal deploy plextract/run_ocr.py
fi

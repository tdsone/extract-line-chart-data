# Extract Line Chart Data
A repo that show how to automatically extract the data of a line chart.

The pipeline works as follows: 
1. Use ChartDete to detect chart elements, most importantly axis labels and the plot area. (`chartdete.py`)
2. OCR the numbers from the labels. (`ocr.py`)
3. Extract the coordinates of the lines in the line chart using LineFormer. (`lineformer.py`)
4. Correct the coordinates of the lines to be relative to the plot origin. (`correct_coordinates.py`)
5. Calculate the conversion from pixels to axis values. (`correct_coordinates.py`)
6. Convert the coordinates using the conversion parameter from step before. (`correct_coordinates.py`)
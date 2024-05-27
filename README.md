# Extract Line Chart Data
A repo that shows how to automatically extract the data of a line chart.

The pipeline works as follows: 
1. Use ChartDete to detect chart elements, most importantly axis labels and the plot area. (`chartdete`)
2. OCR the numbers from the labels. (`ocr.py`)
3. Extract the coordinates of the lines in the line chart using LineFormer. (`lineformer.py`)
4. Correct the coordinates of the lines to be relative to the plot origin. (`correct_coordinates.py`)
5. Calculate the conversion from pixels to axis values. (`correct_coordinates.py`)
6. Convert the coordinates using the conversion parameter from step before. (`correct_coordinates.py`)
   

## Example

### Input
![Example Input](example/input.png)

### Output
This chart was generated using matplotlib using the extracted data (`example/data.json`)
![Example Output](example/output.png)

## Resources
- [LineFormer](https://github.com/TheJaeLal/LineFormer)
- [ChartDete](https://github.com/pengyu965/ChartDete/)


# Contact
If you need help setting this up or would just like to use it, shoot me an email: mail@timonschneider.de
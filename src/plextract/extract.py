def extract():
    f"""
    Notes
    - If we pass multiple files, the extraction process should be resumable. 

    Arguments
    - if you pass a folder it will try to find all possible images recursively and 
    - if you pass an array of a single file it should only extract from that file
        - we should support png and jpeg
        - later, we could support pdf where we find the image in the PDF but then it would wrap around extract
    
    Return
    - we want to save the extracted data somewhere if the user wants that. by default we save it
    - we should also return the data, we could do so as a dictionary with each file pointing to the file where the data is saved
        - e.g. {
            "path-to-input-image": "path-to-folder-with-results" 
        }


    What comes after this? 
    1. Use case 1: train a model on the collected data
        - I would want to do a quality check; how to do so? 
            - Overlay the extracted lines with the image?
            - Display image and extraction side by side

    files = os.listdir("myfolder") 

    out_dict = extract(files)

    input_img, outout_folder = list(outdict.items())[0]

    import numpy as np

    data = np.load(Path(output_folder) / "data.npz")) -> this would be a two dimensional array dim 1 = series, dim 2: points

    data 
    [
        [[x0,y0], [x1, x2], ... ], 
        [ ... ], 

    ]

    but how to map this to the name of the line series? -> we need an annotation for each row 
    -> have another file which stores the metadata? i.e. 
    
    [
        {"title": ...., ""}
    ]

    is there another way to save the data in the same file that's commonly used? 

    """
    print("hi")
    pass 
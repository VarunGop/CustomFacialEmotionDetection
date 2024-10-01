This project was designed to create a generically trained facial recognition model and compare it to a model
trained on a single person to see if the model had any changes in confidence.

The fer2013 database was used to train the generic (control) model. For simplicity only the emotions of happiness, sadness and anger were studied. The big dataset was filtered using 
csvfilter.py. This model was then used with genericmodel.py to generate a .h5 model file: genmodel.h5


The custom set was made by first taking 50 pictures of each emotion in a single subject, for a total of 150 pictures. The picture amount was then artificially augmented using 
transformations to increase the training set, using imageaugmentation.py. These pictures were then filtered to match the format of fer2013. To do this images were ensured to contain a 
detectable face using a .xml cascade, resized to 48x48, converted to greyscale, tagged with the expressed emotion (given by the image name) and then converted to a .csv row entry 
with their pixel numbers as the entries. This was done using csvconvertsmartypants.py


These models were then tested using modeltest2.py. The file used the webcam to detect the emotion of a subject live, and displays confidence values, with the added ability to print to console for analysis.
# Diabetes-Retina-Images-Classification-
1 : Create A Virtual Environnement in your local Pc : 

	python3 -m venv Env_Name 
	
2 : Activate the virtual Environnement : 

	source /Path_To_Activate_File 
	
3 : Install The necessary dependencies using : 

	pip3 install -r Requirements.txt 
	
4 : In order to create a folder Containing all the images

python3 Labeling_Images.py 


and create a csv file containing the labels of each image 

PS : Change F_Images_Path , Path *  4 and the position  of the csv file based on your PC 

5 : In order to do some Process on the images   

python3 preprocess_images.py 

The idea is to preprocess the images in order to eliminate the black in the Images that 

will  clearly help  the model You will obtain of all the images Preproceed in an one folder 

PS :   Change the paths in the main file fast_image_resize ("/home/semi/Image_Classification/Full_Images/", "/home/semi/Image_Classification/Full_Preproceed/", output_size=(150, 150))


The first one is for the generated folder in 4) and the second path is the path of the folder that will contain 
the preprocced images 


6 : Lunch the trainning process now 

python3 train.py

PS : Change the path in main of Full proceced images and labels.csv 
and lunch the train.py 

PS make_prediction : is the function responsible on predicting the labels .. It can be deployed 
in the application 

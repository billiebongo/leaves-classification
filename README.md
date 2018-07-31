
#########################

Final Submission: preprocessing.pass results.submission_8_0test.csv

########################

open_cv data obtained with preprocessing.open_cv.py and folder of leaves images (1584 images) and output to all_image_output_v2.csv

preprocessing.mergedbycol.py separates the OpenCV outputnin all_image_output_v2 to merged_train_sl_pit1.csv and merged_test_sl_pit1.csv used for training models

preprocessing.classifiers.py build models on several classifiers to get an idea on which classifier to use

PCA data obtained with models.pca_leaves and output to PCA_labels_saved.csv

Final model is trained with models.train_model.py

scripts in leaves.scripts are just scripts to organize data to view the data more efficiently while analysing results from opencv and model training.

#threholding
import pandas as pd
import numpy as np

# Attempted thresholding to improve score by correcting log-loss, 
# did not work well due to low confidence for some text samples




def return_max(row, RF_row):

	c=np.where(row == np.max(row))

	if row[c[0]] > 0.79:
		print(row[c[0]])
		print("thresohold since more 78 confidence")
		max_row=np.zeros(99)
		print(np.max(row))
		max_row[c[0]]=1
		return max_row
	else:
		print(type(row)) #numpy.ndarray


		print(type(RF_row))
		#return RF_row
		return row

def threshold(test_prob, classes, RF_proba):

	#submission = pd.DataFrame(test_predictions, columns=classes)
	for i in range(len(test_prob)):
		max_row=return_max(test_prob[i], RF_proba[i])
		test_prob[i]=max_row

	return test_prob



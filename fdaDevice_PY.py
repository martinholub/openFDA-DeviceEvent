# Import modules
import numpy as np
import pandas as pd
import json
import requests
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans, MiniBatchKMeans
import logging
import sys
from time import time
from datetime import datetime, timedelta

from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt

## Global Variables
list_features = ['device_date_of_manufacturer', 
                 'date_of_event',
                 #'date_report',
                 #'date_received',
                 'previous_use_code',
                 #'remedial_action',
                 #'single_use_flag',
                 #'reprocessed_and_reused_flag',
                 #'reporter_occupation_code',
                 #'device.date_received',
                 #'device.generic_name' # this allows for empty string! 
                ]

list_device_names = ["pump",
                    "sensor",
                    "prosthesis",
                    "defibrilator",
                    "pacemaker",
                    "catheter",
                    "electrode",
                    #"wearable",
                     "stent",
                     "ray",
                     "ventilator",
                     "bed",
                     "implant",
                     "lens",
                     #"mds" # https://www.cancer.org/cancer/myelodysplastic-syndrome/about/what-is-mds.html
                     "dialysis",
                     "graft",
                    ]

fois_result = ['device_date_of_manufacturer',
               'date_of_event',
               'previous_use_code',
               'single_use_flag',
               'reprocessed_and_reused_flag',
               #'reporter_occupation_code'
              ]
fois_device = [#'generic_name', 
               'expiration_date_of_device', 
               #'device_age_text', 
               #'implant_flag', 
               #'date_removed_flag', 
               #'manufacturer_d_name', 
               #'manufacturer_d_state',
               'manufacturer_d_country',
               #'device_operator'
              ]
fois_patient = [#'sequence_number_outcome', # problematic to factorize as it contains lists
                #'sequence_number_treatment' # problematic to factorize as it contains lists
              ]
fois_mdrText = ['text',
                'text_type_code']
fois_openfda = ['device_name',
                'device_class',
                'medical_specialty_description']

# Columns that we want to translate into categories
factCols = ['previous_use_code',
            'single_use_flag',
            'reprocessed_and_reused_flag',
            #'reporter_occupation_code',
            #'implant_flag',
            #'manufacturer_d_name',
            'manufacturer_d_country',
            #'device_operator',
            #'sequence_number_outcome', # problematic to factorize as it contains lists
            #'sequence_number_treatment', # problematic to factorize as it contains lists
            'text_type_code_0',
            'text_type_code_1', 
            'text_type_code_2',
            'device_name',
            'medical_specialty_description']

time_period = 7
initial_date = "20150101"
final_date = datetime.strptime("20150301", "%Y%m%d")
fillDic = {'mdr_text_key': '', 'patient_sequence_number': '', 'text': np.nan, 'text_type_code': np.nan}
n_features = 10000
k = 5

## Functions definitions
def buildAndQuery(initial_date, final_date, time_period = 7):
	# Build base query
	baseurl = 'https://api.fda.gov/device/event.json?'
	skip = 0
	limit = 100

	apikey = ''
	with open('apikey.txt', 'r') as myfile:
		apikey = myfile.read().replace('\n', '')
	
	start_date = datetime.strptime(initial_date, "%Y%m%d")
	end_date = start_date +  timedelta(days=time_period)
	device_name = list_device_names[4]
	results = []

	while True:
		skip = 0
		
		while (skip<=5000):
			query = 'search=device.generic_name:'+device_name+'+AND+'
			# adding date range
			start = str(start_date.date()).replace("_", "")
			end = str(end_date.date()).replace("_", "")
			query = query+"date_of_event:[\""+start+"\""+"+TO+"+"\""+end+"\"]"
			
			# checking features for existence
			for x in list_features:
				query = query + "+AND+_exists_:" + x
				
			# # Possibility to furhter narrow down the search
			# for y in list_features_specific:
			#     query = query + "+AND+" + y

			
			q1 = baseurl + 'api_key=' + apikey + '&' + query + '&' + 'limit=' + str(limit) + '&' + 'skip=' + str(skip)
			dq1 = requests.get(q1)
			# dq1.json()['results']
			data = json.loads(dq1.text)

			if "results" in data:
				result = data['results']
				results = results + result
				print(len(results))
				skip = skip + limit
			else:
				break

		print("Week Done")
		start_date = end_date + timedelta(days=1)
		end_date = start_date +  timedelta(days=time_period)
		if start_date > final_date:
			break
			
		if end_date > final_date:
			end_date = final_date
	
	return results
	
def selectData(results, fois_result, fois_device, fois_patient, fois_mdrText, fois_openfda, fillDic):
	# device = data['results'][0]['device'][0]
	device = [x['device'][0] for x in results]
	# patient = data['results'][0]['patient'][0]
	patient = [x['patient'][0] for x in results]
	# mdrText = data['results'][0]['mdr_text'][0] # there may be more items in the list! 
	mdrText = [x['mdr_text'] for x in results]
	#mdrText = [y['text'] for y in [x['mdr_text'][0] for x in data['results']]]
	# openfda = data['results'][0]['device'][0]['openfda']
	openfda = [x['device'][0]['openfda'] for x in results]
	
	# Create sub dataframes for non-multiple columns
	df_results = pd.DataFrame(results, index = range(len(results)), columns = fois_result)
	df_openfda = pd.DataFrame(openfda, index = range(len(results)),columns = fois_openfda)
	df_device = pd.DataFrame(device, index = range(len(results)),columns = fois_device)
	df_patient = pd.DataFrame(patient, index = range(len(results)),columns = fois_patient)
	
	
	# Pull out all relevant text fields
	a = [x[0] if len(x) > 0 else fillDic for x in mdrText]
	b = [x[1] if len(x) > 1 else fillDic for x in mdrText]
	c = [x[2] if len(x) > 2 else fillDic for x in mdrText] 


	a = pd.DataFrame(a, index = range(len(results)),columns = fois_mdrText)
	b = pd.DataFrame(b, index = range(len(results)),columns = fois_mdrText)
	c = pd.DataFrame(c, index = range(len(results)),columns = fois_mdrText)
	
	# Rename duplicate columns:
	fois_mdrText_a = [x + '_0' for x in fois_mdrText]
	fois_mdrText_b = [x + '_1' for x in fois_mdrText]
	fois_mdrText_c = [x + '_2' for x in fois_mdrText]

	columns_a = dict(zip(fois_mdrText, fois_mdrText_a))
	columns_b = dict(zip(fois_mdrText, fois_mdrText_b))
	columns_c = dict(zip(fois_mdrText, fois_mdrText_c))

	a.rename(columns = columns_a, inplace = True)
	b.rename(columns = columns_b, inplace = True)
	c.rename(columns = columns_c, inplace = True)
	
	# Construct final data frame
	df_mdrText = pd.concat([a, b, c], axis = 1)

	# Concatenate into final dataframe
	df = pd.concat([df_results, df_device, df_patient, df_mdrText, df_openfda], axis = 1)
	
	# Determine age of device in days
	df['age_of_device_days'] = pd.to_datetime(df['date_of_event'], format='%Y%m%d') \
	- pd.to_datetime(df['device_date_of_manufacturer'], format='%Y%m%d')

	# Determine timedelta from manufacture to specified expiry date
	df['days_to_expiry'] = pd.to_datetime(df['expiration_date_of_device'], format='%Y%m%d') \
	- pd.to_datetime(df['device_date_of_manufacturer'], format='%Y%m%d')
	df = df.drop(['date_of_event','device_date_of_manufacturer', 'expiration_date_of_device'], axis = 1)
	
	return df
	
def factColsfun(df, factCols):
	# copy dataframe to preserve original;
	df1 = df.copy()

	# This also works but will not assign consistent labeling across multiple columns
	df1[factCols] = df1[factCols].apply(lambda x: pd.factorize(x)[0])
	df1[factCols] = df1[factCols].apply(lambda x: x.astype('category'))
	return df1

def txtVectfun(df1, n_features = 10000):
	# pull out relevant subset
	# here we use regular expression to match all column names starting with "text_#" (ie we pick 3 columns)
	df_text = df1[df1.columns[df1.columns.to_series().str.contains('^text_[0-9]$')]]
	
	# comvert text data into single dimensional iterable [list]

	# save structuring information for possible later use
	IDXS = df_text.index
	COLS = df_text.columns
	# join texts into one string, droping nans
	df_text_cat = df_text.apply(lambda x: '; '.join(x.dropna().values.tolist()), axis=1)
	text_np_cat = df_text_cat.values

	# text_np = df_text.values
	# text_np_cat = ['; '.join(map(str, x)) for x in text_np]
	
	# Do Tfidf vectorization
	t0 = time()
	vectorizer = TfidfVectorizer(max_df=0.5, max_features=n_features,
									 min_df=2, stop_words='english',
									 use_idf=True)
	X = vectorizer.fit_transform(text_np_cat)

	print("done in %fs" % (time() - t0))
	print("n_samples: %d, n_features: %d" % X.shape)
	
	return X
	
def clusterFailurefun(X, k = 5):
	
	#clustering
	km = MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=1,
							 init_size=1000, batch_size=1000, verbose=False)
	print("Clustering sparse data with %s" % km)
	t0 = time()
	km.fit(X)
	print("done in %0.3fs" % (time() - t0))
	
	return km.labels_

def dataTargetfun(df1, labels):

	df2 = df1.drop(df1.columns[df1.columns.to_series().str.contains('^text_*')], axis = 1)
	df2['mdr_text_class'] = labels
	
	
	# X
	df2['days_to_expiry'] = df2['days_to_expiry'].dt.days

	X = df2.ix[:, df2.columns != 'age_of_device_days'].values
	days = df2['age_of_device_days']

	# y
	y = days.dt.days.values
	
	return X, y

def visClustersfun(labels):

	uniqs, uniq_counts = np.unique(labels, return_counts = True)
	# dict(zip(uniqs,uniq_counts))
	
	width = 0.5       # the width of the bars

	fig, ax = plt.subplots()
	rects1 = ax.bar(uniqs, uniq_counts, width, color='r')

	# add some text for labels, title and axes ticks
	ax.set_ylabel('Counts')
	ax.set_xlabel('Classes')
	ax.set_title('Counts of class membership')
	ax.set_xticks(uniqs)
	plt.show()

def predictFailfun(X, y):
	# impute missing entries (due to missing expiry date) with mean
	imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
	X_imputed = imp.fit_transform(X)

	# Nomralize
	X_normalized = preprocessing.normalize(X_imputed, norm='l2')
	
	
	kf = KFold(n_splits=10, shuffle=True)
	for train_index, test_index in kf.split(X_normalized):
		#print("TRAIN:", train_index, "TEST:", test_index)
		X_train, X_test = X_normalized[train_index], X_normalized[test_index]
		y_train, y_test = y[train_index], y[test_index]
		regr = linear_model.LinearRegression()
		regr.fit(X_train, y_train)
		print("Mean squared error: %.2f" % np.mean((regr.predict(X_test) - y_test) ** 2))

if __name__ == "__main__":

	results = buildAndQuery(initial_date, final_date, time_period)
	
	df = selectData(results, fois_result, fois_device, fois_patient, fois_mdrText, fois_openfda, fillDic)
	
	df1 = factColsfun(df, factCols)
	
	X_text = txtVectfun(df1, n_features)
	
	labels = clusterFailurefun(X_text, k)
	
	X, y = dataTargetfun(df1, labels)
	
	predictFailfun(X, y)
	
	
	
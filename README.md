# Advanced-ML-portfolio

Intended to act as a portfolio of projects I worked for my Projects in Machine Learning class at Columbia University in the Spring 2021 semester.

In the first notebook, I used the UN's World Happiness Rankings country level data (https://worldhappiness.report/) to experiment with models that predicted happiness rankings well. The X variables used were GDP per capita, Social support, Healthy life expectancy, Freedom to make life choices, Generosity, and Perceptions of corruption. The Y variable was Happiness_level (Very Low to Very High).

The models I tried out were:

SVC
LogisticRegression
RandomForestClassifier
KNeighborsClassifier with Grid Search
My best performing model was SVC. However, even it had a low f1 score of 53.24%. This was because the total number of rows in the dataset was limited to the number of countries that exist (fewer than 200).


In notebook 2, we were provided with a COVID-19 Radiography Dataset which contained a mixture of 423 COVID-19, 1485 viral pneumonia, and 1579 normal chest X-ray images. Building a predictive model using this dataset was useful as it could enable doctors to tell whether or not a patient has COVID-19 just by looking at an X-ray scan of their chest. This can help speed up the time taken to diagnose a patient with COVID-19. It can also help differentiate cases of COVID-19 from other diseases such as viral pneumonia, which would require different medications and treatments.

After splitting the dataset, the training data consisted of 1200, 1345 and 1341 images of COVID-19, viral pneumonia, and normal patients respectively. The test data had 240, 269, and 269 images of COVID-19, viral pneumonia, and normal patients respectively. The models I tried out were:

Model 1: A Keras Sequential model consisting of Conv2D and MaxPooling2D layers
Model 2: A transfer learning model using VGG16
Model 3: Another transfer learning model using InceptionV3 and imagenet weights
The f1 scores for the best performing models using the three techniques were-

Model 1: 0.961777

Model 2: 0.983636

Model 3: 0.97971

In my third notebook, I used a dataset containing COVID-19 related tweets which were labelled as being real or fake depending on whether they contained misinformation. The data was obtained from this paper.

The training and test datasets contained 6420 and 2140 rows respectively. I posited that one use of building a predictive model using this dataset is for news organizations who might need to quickly fact-check tweets before reporting them. This would help build trust among the general public, by not allowing tweets containing misinformation to be spread in the news. Another reason for building predictive model using this dataset is that it can benefit researchers who might be interested in ascertaining which politician retweeted the most amount of misinformation.

The models I tried out were-

Model 0: 1 embedding layer and 1 dense layer but no layers meant for sequential data
Model 1: two Conv 1D layers
Model 2: one LSTM layer with dropout
Model 3: stacked LSTM's with dropout
Model 4: Bidirectional LSTM
Model 1 performed the best, with an f1 score of 0.9434. Model 2 performed second best, with an f1 score of 0.9424. A variation of Model 1 with different hyperparameters (32,098,786 trainable parameters, compared to my best performing model's 148,066) came third with an f1 score of 0.9419.

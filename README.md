# Sentiment Analysis &nbsp;&nbsp
### Objective 
The objective of this projet is to perform sentiment analysis on textual data to identify and classify the opinions expressed as positive,negative,or neutral.This helps in understanding user behavior,analyzing feedback,and extracting meaningful insights from unstructured text data.Sentiment analysis is used to analyze text data such as reviews,comments,or feedback and classify the sentiment into categories like positive,negative,or neutral.This process helps in understanding customer opinions,identifying trends,and supporting data-driven decision-making
### Sample Dataset

### Dataset Overview
+ The dataset used in project is Twitter dataset ehich contains textual data collected from Kaggle. it is primarily sentiment analysis.
+ The dataset contains a mix of positive ,neutral,and negative sentiments,making it suitable for training and evaluating sentiment classification models.
+ The dataset is used to:
   -Analyze public opinion through text data
   - Classify sentiments into positive,negative,or neutral
   - Build sentiment analysis models using machine learning techniques

 ### Tools and Technologies
 + Google Colab - Cloud-based environment for writing and executing python code
 + Python-Core programming language used for implementation
 + Pandas-Used for data manipulation,cleaning,and preprocessing.
 + NumPy-Used for numerical operations and handling arrays.
 + Regular Expressions(re)-Used for text preprocessing and cleaning operations
 + TF-IDF Vectorizer - Used to convert textual data into numerical format so that it can be used for machine learning
 + Logistic Regression(Scikit-learn)-Used as the machine learning model for sentiment classification
 + Evaluation Metrics (Scikit-learn)
         Accuracy Score-To measure model performance
         Classification Report - To evaluate precision,recall,and F1-score
         Confusion Matrix - To visualize prediction performance

  ### Steps in Sentiment Analysis
   The following steps were performance to bulid the sentiment analysis model

 1.Import Libraries
     + Pandas-Data handling
     + re-Text cleaning
     + Scikit-learn - Model building and evaluation

 2.Load Dataset
     + Loaded dataset using 'pd.read_csv()'
     + Display initial data

 3.Data Inspection
     + View dataset structure using df.info()and df.describe()function
   
 4.Data cleaning
     + Removed duplicate rows
     + Identified missing values
     + Removed rows with missing test or category

 5.text preprocessing
     + Converted text to lowercase
     + removed URLs,Special  characters and Extra spaces
   
 6.Feature Extraction (Text to Numbers)
     + Used TF-IDF Vectorizer to convert text into numerical vectors
     + Extracted features from cleaned set

 7.Define Target Variable
     + Set X- Text features and y - Sentiment labels ('category')

 8.Train-Test Split
    + Split data into Training (80%)and Testing(20%)

 9.Model Training
     + Used Logistic Regression

 10.Model Evaluation
      + Predicted sentiment on test data
      + Calcualted Accuracy Score Classification Report and Confusion Matrix

 11.Sample prediction
      + Tested model on new important
      + Predicted sentiment label
      + Mapped output to:Positive(1),Neutral(0),Negative(-1)

 12.Visualized
      + Plotted sentiment distribution using bar chart
      + visualized frequency of each sentiment category

   ### Outcome
      


<img width="1717" height="233" alt="image" src="https://github.com/user-attachments/assets/15cbb4ed-7a86-415a-96dd-4b96499366fc" />



   + The sentiment analysis model was successfully able to classify text data into different sentiment categories:Postive,Neutral,and Negative
   + As shown in the result,when a sample input sentence was provided to the trained model,it correctly predicted the sentiment as **Postive(1)**.This indicates that the model has learned meaningful patterns from the data and can effectively analyze new,unseen text.
   + The use of TF-IDF vectorizztion and Logistic Regression helped in achieving and predicting sentiments from textual data,making it useful for analyzing opinions and feedback

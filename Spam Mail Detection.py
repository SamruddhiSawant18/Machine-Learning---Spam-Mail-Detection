import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tkinter import *
from tkinter import messagebox

''' Loading the dataset '''

# loading the data from csv file to a pandas Dataframe
raw_mail_data = pd.read_csv('C:/Users/Reliance/Downloads/mail_data.csv')

''' Data preprocessing '''

# replace the null values with a null string
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),'')

# printing the first 5 rows of the dataframe
print("\nDisplaying the first 5 entries of the dataset :\n")
print(mail_data.head())

# checking the number of rows and columns in the dataframe
print("\nNumber of rows and columns in dataset : ")
print(mail_data.shape)

''' Categorizing the Data '''

# label spam mail as 0;  Important mail as 1;
mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1

# separating the data as texts and label

X = mail_data['Message']
Y = mail_data['Category']

print("\nText data[Mails]:\n")
print(X)
print("\nLabels of Data:[0 for spam and 1 for important]\n")
print(Y)

# Splitting the dataset into training data and Testing Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

print("\nShapes of the dataset:")
print("\nTotal Mails:\n")
print(X.shape)
print("\nTrained Mails:\n")
print(X_train.shape)
print("\nMails for Testing:\n")
print(X_test.shape)

# transform the text data to feature vectors that can be used as input to the Logistic regression

feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase='True')

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# convert Y_train and Y_test values as integers

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

print('\nDisplaying the trained values of Mails and their features :\n')
print(X_train)

print(X_train_features)

# Training the model using Logistic Regression
model = LogisticRegression()

# Training the Logistic Regression model with the training data
model.fit(X_train_features, Y_train)

''' Evaluating the trained model '''
# prediction on training data

prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)

print('\nAccuracy on training data : ', accuracy_on_training_data)

# prediction on test data

prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)

print('\nAccuracy on test data : ', accuracy_on_test_data)

def buttonClick() :
    #Building a predictive system
    input_mail = []
    input_mail = [Mail.get("1.0", END)]

    if Mail.get("1.0") == " ":
        messagebox.showerror("Missing Data", "Please enter a mail")
        return
    #input_mail = ["I've been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times"]
    #input_mail = ["Marvel Mobile Play the official Ultimate Spider-man game (£4.50) on ur mobile right now. Text SPIDER to 83338 for the game & we ll send u a FREE 8Ball wallpaper"]
    #input_mail = ["Hello All. Hope you all are good."]
    # convert text to feature vectors
    input_data_features = feature_extraction.transform(input_mail)

    # making prediction

    prediction = model.predict(input_data_features)
    print(prediction[0])
    if (prediction[0] == 0):
        classification = Label(root, text="", font=('helvetica', 15 , 'bold'), fg="red")
        classification.pack()
        messagebox.showerror("Spam","Detected to be Spam!")

    if (prediction[0] == 1):
        classification = Label(root, text="", font=('helvetica', 15, 'bold'), fg="green")
        classification.pack()
        messagebox.showinfo("Important","Detected to be Important!")
        #print(prediction)


''' Creating GUI'''
root = Tk()
root.title("SPAM MAIL DETECTOR")
root.geometry("440x550")
root.configure(bg='lavender')
font = ("Helvetica", 22, "bold")

# label for text message
Label0 = Label(root, text="Mail", font=("helvetica", 15, "bold"), bg='#2764bf', fg='#d8e9f4', width='150')
Label0.pack()

# using text widget for reading the message
Mail = Text(root, font=("helvetica", 11))
Mail.pack(fill=BOTH)

# button to send SMS
sendBtn = Button(root, text="Check Mail", command=buttonClick, activebackground="LightSkyBlue", bg='#2764bf', bd=10, font=('Arial', 12), fg='white', relief='groove')

sendBtn.pack()
root.mainloop()
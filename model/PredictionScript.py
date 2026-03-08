import pickle
import pandas as pd
modelObject = pickle.load(open(r"C:\Users\krish\Downloads\Project-1-cdgi\model\model.pkl","rb"))
# 'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus'

sample = pd.DataFrame([[2000,3,2,1,1,1,1,1,1,1,1,1]],columns=['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus'])
prediction =modelObject.predict(sample)
print(prediction)
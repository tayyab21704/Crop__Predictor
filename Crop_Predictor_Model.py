import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


df = pd.read_excel(r'C:\physicspbl\Copy of Crop_recommendation_konkan_maharashtra(1).xlsx')


X = df.drop('crop', axis=1)
y = df['crop']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=5,
                           scoring='accuracy',
                           n_jobs=-1)
grid_search.fit(X_train, y_train)


best_model = grid_search.best_estimator_


best_model.fit(X_train, y_train)


y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy with GridSearchCV:", accuracy)

h = float(input("Enter The Humidity of the target:"))
t = float(input("Enter The temperature of the target:"))
r = float(input("Enter The rainfall of the target:"))


new_data = {
    'humidity': [h],
    'temperature': [t],
    'rainfall': [r]
}


new_df = pd.DataFrame(new_data, columns=X_train.columns)
prediction = best_model.predict(new_df)
print(prediction[0])

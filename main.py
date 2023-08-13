import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
weather=pd.read_csv(r"C:\Users\Kowshik\Downloads\archive\weatherHistory.csv")
weather_temp=weather[["Humidity","Apparent Temperature (C)"]]
dummies=pd.get_dummies(weather['Summary'])
weather_temp2=pd.concat([weather_temp,dummies],axis=1)
X=weather_temp2
Y=weather_temp["Apparent Temperature (C)"]
x=X>=0
X_train,X_test,Y_train,Y_test=train_test_split(
    X,Y,test_size=0.3
)
model=LinearRegression()
model.fit(X_train,Y_train)
plt.scatter(X_train["Humidity"],Y_train)
plt.title("Temperature VS Humidity")
plt.xlabel("Humidity")
plt.ylabel("Temperature")
plt.show()
Y_predict=model.predict(X_test)
plt.scatter(X_test["Humidity"],Y_predict,color='red')
plt.title("Temperature VS Humidity")
plt.xlabel("Humidity")
plt.ylabel("Temperature")
plt.show()
plt.scatter(X_test["Humidity"],Y_test,color='green')
plt.title("Temperature VS Humidity")
plt.xlabel("Humidity")
plt.ylabel("Temperature")
plt.show()
test=pd.DataFrame(Y_test)
test["Y_predict"]=Y_predict
variance=explained_variance_score(Y_test,Y_predict, multioutput='uniform_average')
print(variance)


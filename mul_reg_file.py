import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def delete():
    os.system('cls' if os.name == 'nt' else 'clear')
delete()


path = r"C:\Users\B-UNIT\Desktop\ML\codes\datasets\\profit/housing_data_expanded_5000.csv"


data = pd.read_csv(path , header=None , names = ['size' , 'bedrooms' , 'price'], skipinitialspace=True  )


data['size'] = pd.to_numeric(data['size'] , errors='coerce')
data['bedrooms'] = pd.to_numeric(data['bedrooms'] , errors='coerce')
data['price'] = pd.to_numeric(data['price'] , errors='coerce')

# show the data
def display():

    print(f"{'-' * 50}\n")
    print(data.info())
    print(f"{'-' * 50}\n")

    print(data.describe())
    print(f"{'-' * 50}\n")
    print(data.nunique())

    print(f"{'-' * 50}\n")
    print(data.head(2))
    print(f"{'-' * 50}\n")

    

# display()


X_original = data[['size', 'bedrooms']]
y_original = data['price']



# predicting

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# function return train of first variable , test of first variable
#  train of second variable , test of second variable

x_train , x_test , y_train , y_test = train_test_split(X_original , y_original , test_size=0.2 , random_state=42)

lin_reg = LinearRegression()
lin_reg.fit( x_train , y_train )


# r2_score ( actual values , predicted values )   : <> بتقول هنسيب تنسيق للرقم الي قبلها
# print(f"Train R* = {r2_score( y_train , lin_reg.predict(x_train)): .4f}")
# print(f"test R* = {r2_score( y_test , lin_reg.predict(x_test)): .4f}")


#******************************
# drawing

x = np.linspace(x_test['size'].min(), x_test['size'].max(), 100)
x_line = pd.DataFrame({'size' : x , 'bedrooms' : np.median(x_test['bedrooms'])})
line = lin_reg.predict(x_line)

y_pred = lin_reg.predict(x_test)


fig , (ax1 , ax2) = plt.subplots( 1 , 2 ,figsize = ( 10 , 5))

# بيرسم نقط محور اكس يحتوي علي السعر الحقيقي محور واي السعر المتوقع
ax1.scatter(y_test , y_pred)

# رقم مش مصفوفه لازم نديله رقمين ل اكس بدايه ونهايه  y_test هنا 
# ونديله رقمين ل واي بدايه ونهايه
# انا عايز هنا ارسم خط السعر المتوقع ف هستعمل الواي 
# لان الاكس تحتوي علي الحجم وعدد الغرف
#  علشان هي تحتوي علي اصغر واكبر سعر موجود عندنا y_test استخدمنا 
ax1.plot([y_test.min() , y_test.max()], [y_test.min() , y_test.max()] , 'r-')
ax1.set_xlabel('actual price')
ax1.set_ylabel('predicted price')
ax1.set_title('Actual vs predicted')

ax2.scatter(x_test['size'] , y_test , label = 'Trainset')
ax2.plot( x , line , 'r-' , label = 'regression line')
ax2.legend(loc = 2)
ax2.set_xlabel('Size')
ax2.set_ylabel('Price')
ax2.set_title('Predicted Price vs  Size')


plt.show()


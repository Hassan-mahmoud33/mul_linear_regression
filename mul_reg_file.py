import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def delete():
    os.system('cls' if os.name == 'nt' else 'clear')
delete()


path = os.path.join('data' , 'mul_var_lin_reg.csv')

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

cols = data.shape[1]
X_original = data.iloc[ :  , 0 : cols - 1 ]
y_original = data.iloc[ : , cols - 1 : cols]

# print(X_original.head(2))
# print(y_original.head(2))


# predicting

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


x_train , x_test , y_train , y_test = train_test_split(X_original , y_original , test_size=0.2 , random_state=42)

lin_reg = LinearRegression()
lin_reg.fit( x_train , y_train )

predicted_price = lin_reg.predict( pd.DataFrame( [[2000 , 3]] , columns = ['size', 'bedrooms']))

print(f"predicted price for house's size equals 2000 M and 3 bedrooms = {predicted_price[0][0]: .2f}k")
print(f"Train R* = {r2_score( y_train , lin_reg.predict(x_train)): .4f}")
print(f"test R* = {r2_score( y_test , lin_reg.predict(x_test)): .4f}")


#******************************
# drawing

x = np.linspace(x_test['size'].min(), x_test['size'].max(), 100)
x_line = pd.DataFrame({'size' : x , 'bedrooms' : np.median(x_test['bedrooms'])})
line = lin_reg.predict(x_line)

y_pred = lin_reg.predict(x_test)

fig , ax = plt.subplots(figsize = ( 7 , 5))


ax.scatter(x_test['size'] , y_test , label = 'Trainset')
ax.plot( x , line , 'r-' , label = 'regression line')
ax.legend(loc = 2)
ax.set_xlabel('Size')
ax.set_ylabel('Price')
ax.set_title('Predicted Price vs  Size')


plt.show()


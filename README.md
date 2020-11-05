# Geopython

```py
###ipython magic commands
#display variable names
%who
# Display variable name, type and info
%whos

#help on a function
help(function name)
#Note that you'll need to press q to exit the help viewer.

#view directory.
print(dir(module or submodule name))

# List all available modules in python. Note: when running this command, you might first get several warnings related to deprecated packages etc.
help("modules")


#require input from a user
input()
#for example variable=input('what is your first name?')

#length
len()

type()

#convert data types
str()
int()
float()

# delete
del 

###LISTS
#lists in square brackets (varibale is called List below)
List = []
#index your list
print(List[0])
#modify list
List[0]= ####
del List[0]
##LIST Methods
#add to a list (string data only)
List.append()

List.count()
#find location of a name in List. Good to locate if you need to change something.
List.index()
List.reverse()
#sort list alphabetically. Capitals will be ordered before lower case. 
List.sort()

####LOOPS
for variable in collection:
        do things with variable

numbers = [0, 1, 2, 3]
for i in range(len(numbers)):
    print('Value of i:', i)
    print('Value of numbers[i] before addition:', numbers[i])
    numbers[i] = numbers[i] + i
    print('Value of numbers[i] after addition:', numbers[i])
    print('')


cities = ['Helsinki', 'Stockholm', 'Oslo', 'Reykjavik', 'Copenhagen']
countries = ['Finland', 'Sweden', 'Norway', 'Iceland', 'Denmark']
for i in range(len(cities)):
    print(cities[i], 'is the capital of', countries[i])

###CONDITIONALS! if, else, elif, and, or, 
#Note: later on we will also need the bitwise operators & for and, and | for or.

if (1 < 0) | (-1 < 0):
    print('At least one test is true')
else:
    print('neither is true!')

###Combing for and conditions.
temperatures = [0, 12, 17, 28, 30]

for temperature in temperatures:
    if temperature > 25:
        print(temperature, 'celsius degrees is hot')
    else:
        print(temperature, 'is not hot')

'''one can write notes over 
mutiple lines doing this'''

#Both examples A and B give the same output
#A
squares = []
for x in range(10):
    squares.append(x**2)
print(squares)
#B
squares = [x**2 for x in range(10)]
print(squares)
#exercise 3 problem 3 loop script
for i in range(n):
    if lats[i] > north_south_cutoff and lons[i] > east_west_cutoff:
        north_east.append(stations[i])
    elif lats[i] < north_south_cutoff and lons[i] > east_west_cutoff:
        south_east.append(stations[i])
    elif lats[i] < north_south_cutoff and lons[i] < east_west_cutoff:
        south_west.append(stations[i])
    elif lats[i] > north_south_cutoff and lons[i] < east_west_cutoff:
        north_west.append(stations[i])

print(north_east, north_west, south_east, south_west)

# .format() is a Python function that can be used to easily insert values inside a text-template such as below.
# .0f below is a specific operator that rounds the decimal values into whole numbers
# the 0 before f dictates the number of decimal places!
print("Northwest contains{share: .0f} % of all stations.".format(share=north_west_share))

###FUNCTIONS
# the following three functions were saved as the module 'temp_converter.py'
def kelvins_to_celsius(temp_kelvins):
    return temp_kelvins - 273.15

def kelvins_to_fahr(temp_kelvins):
    temp_celsius = kelvins_to_celsius(temp_kelvins)
    temp_fahr = celsius_to_fahr(temp_celsius)
    return temp_fahr
    
def celsius_to_fahr(temp):
    return 9/5 * temp + 32
    
from temp_converter import celsius_to_fahr

print("The freezing point of water in Fahrenheit is:", celsius_to_fahr(0))

from my_script import func1, func2, func3

import temp_converter as tc

print("The freezing point of water in Fahrenheit is:", tc.celsius_to_fahr(0))


def hello(name, age):
    return 'Hello, my name is ' + name + '. I am ' + str(age) + ' years old.'

output = hello(name='Dave', age=39)
print(output)

def function(name, age, x, y):
    return 'Hello, my name is ' + name + ' I am ' + str(age) + ' years old.\
My town is located at ' + str(y) + ' degrees north and ' + str(x) + ' degrees west.'

output_variable = function(name = "Shane", x = 23.78, y = 34.77, age = 37)
print(output_variable)



###advanced topic, functions with multiple parameters. help(temp_calculator). Includes a docstring after the definition of the function.
def temp_calculator(temp_k, convert_to):
    """
    Function for converting temperature in Kelvins to Celsius or Fahrenheit.

    Parameters
    ----------
    temp_k: <numerical>
        Temperature in Kelvins
    convert_to: <str>
        Target temperature that can be either Celsius ('C') or Fahrenheit ('F'). Supported values: 'C' | 'F'

    Returns
    -------
    <float>
        Converted temperature.
    """

    # Check if user wants the temperature in Celsius
    if convert_to == "C":
        # Convert the value to Celsius using the dedicated function for the task that we imported from another script
        converted_temp = kelvins_to_celsius(temp_kelvins=temp_k)
    elif convert_to == "F":
        # Convert the value to Fahrenheit using the dedicated function for the task that we imported from another script
        converted_temp = kelvins_to_fahr(temp_kelvins=temp_k)
    # Return the result
    return converted_temp

#Test the function
import temp_converter as tc
help(tc.temp_calculator)

###Loading modules (interchangeably called libaries and packages in other programming languages). 
#to view the functions in a module
print(dir(math))

import math
#we can use a function within the math library by typing the \
#name of the module first, a period, and then the name of function
math.sqrt(81)
#We can also rename modules when they are imported. This can be helpful when using modules with longer names.
import math as m
#Some modules have submodules that can also be imported without importing the entire module
# here we import the core submodule from the qgis module
import qgis.core
#for a list of functions in the core submodule)
print(dir(qgis.core))

#It is customary to import pandas as pd
import pandas as pd
#It is also possible to import only a single function from a module, rather than the entire module. \
#This is sometimes useful when needing only a small piece of a large module. 
#it has the drawback that the imported function could conflict with other built-in or imported function names
#You should only do this when you truly need to.
from math import sqrt

###installing packages using Conda
#You can first check which packages you have installed using the conda list command. It’s a good idea to search for /
#installation instructions for each package online.
conda list
#this may need to be in a terminal
conda install [packagename]
# visit this page for importing conda etc https://geo-python.github.io/site/course-info/installing-anacondas.html

import temp_functions

temp_classes = []
#this code works, and is shorter, but does not follow quidelines for marking, ie., variable names.
'''for temp in temp_data:
    temp_classes.append(temp_classifier(fahr_to_celsius(temp)))

print(temp_classes)  
'''
#or 
for temp_fahrenheit in temp_data:
    temp_celsius = fahr_to_celsius(temp_fahrenheit)
    temp_class = temp_classifier(temp_celsius)
    temp_classes.append(temp_class)

print(temp_classes)

# 1. How many 0 values exist in temp_classes -list?
zeros = temp_classes.count(0)
print(zeros)

###PANDAS
import pandas as pd
# Read the file using pandas
data = pd.read_csv('Kumpula-June-2016-w-metadata.txt')
#to skip the metadata or unwanted rows at the top of a file
data = pd.read_csv('Kumpula-June-2016-w-metadata.txt', skiprows=8)
#defining the type of seperator in the imported csv.
data = pd.read_csv(fp, sep=',')
#the panda head function will by default show the first 5 lines 
data.head()
#conversally tail will show the end of the data frama
data.tail()
# Print number of rows using len()-function
print(len(data))
# Print dataframe shape (number of rows and columns)
print(data.shape)
#Print column values
print(data.columns.values)
#Print index
print(data.index)
#Check length of the index (Eventually, the "length" of the DataFrame (the number of rows) is actually the length of the index:)
len(data.index)
# Print data types (We can check the data type of all the columns at once using)
print(data.dtypes)
#selecting columns
selection = data[['YEARMODA','TEMP']]
#mean(), median(), min(), max(), and std()
# Check mean value of a column
data['TEMP'].mean()
# Check mean value for all columns
data.mean()
# Get descriptive statistics
data.describe()
##create Pandas objects from python lists
# Create Pandas Series from a list
number_series = pd.Series([ 4, 5, 6, 7.0])
print(number_series)
#custom index
number_series = pd.Series([ 4, 5, 6, 7.0], index=['a','b','c','d'])
print(number_series)
#combine multiple lists into a pandas dataframe
new_data = pd.DataFrame(data = {"station_name" : stations, "lat" : lats, "lon" : lons})
new_data #you can print the pandas dataframe without using print()
#Often, you might start working with an empty data frame in stead of existing lists:
df = pd.DataFrame()
print(df)

fp = 'Kumpula-June-2016-w-metadata.txt'
data = pd.read_csv(fp, sep=',', skiprows=8)

# Select first five rows of dataframe using index values
selection = data[0:5]
# Select temp column values on rows 0-5
selection = data.loc[0:5, 'TEMP']
# Select columns temp and temp_celsius on rows 0-5
selection = data.loc[0:5, ['TEMP', 'TEMP_CELSIUS']]
#Sometimes it is enought to access a single value in a DataFrame. In this case, we can use DataFrame.at in stead of Data.Frame.loc
data.at[0, "TEMP"]
##using iloc
# data at row 1 column 2
data.iloc[0,1]
##filtering and updating using pandas Dataframe
# Select rows with temp celsius higher than 15 degrees
warm_temps = data.loc[data['TEMP_CELSIUS'] > 15]
print(warm_temps)
# Select rows with temp celsius higher than 15 degrees from late June 2016
warm_temps = data.loc[(data['TEMP_CELSIUS'] > 15) & (data['YEARMODA'] >= 20160615)]
print(warm_temps)
# Reset index. Drop or keep previous index is True or False
warm_temps = warm_temps.reset_index(drop=True)
print(warm_temps)
# Drop no data values based on the MIN column
warm_temps.dropna(subset=['MIN'])
#makeing those drops permanant
warm_temps = warm_temps.dropna(subset=['MIN'])
# Fill na values
warm_temps.fillna(-9999)
##data type conversions
.round(0).astype(int)
print("Rounded integer values:")
print(data['TEMP'].round(0).astype(int).head())

##unique values in a dataframe
unique = data['TEMP'].unique()
unique
# unique values as list
print(list(unique))
# Number of unique values
unique_temps = len(unique)
print("There were", unique_temps, "days with unique mean temperatures in June 2016.")

##sorting
# Sort dataframe, descending
data.sort_values(by='TEMP', ascending=False)

##writing data to a file
# define output filename
output_fp = "Kumpula_temps_June_2016.csv"
# Save dataframe to csv
data.to_csv(output_fp, sep=',')
#ave the temperature values from warm_temps DataFrame without the index and with only 1 decimal in the floating point numbers
output_fp2 = "Kumpula_temps_above15_June_2016.csv"
warm_temps.to_csv(output_fp2, sep=',', index=False, float_format="%.1f")

###more pandas
#We can either use the sep or delim_whitespace parameter; sep='\s+' or delim_whitespace=True but not both.

fp = r"data/029440.txt"
# Read data using varying amount of spaces as separator and specifying * characters as NoData values
data = pd.read_csv(fp, delim_whitespace=True, na_values=['*', '**', '***', '****', '*****', '******'])

#check names of columns
data.columns
# This time, we will read in only some of the columns using the usecols parameter. 
data = pd.read_csv(fp, delim_whitespace=True, usecols=['USAF','YR--MODAHRMN', 'DIR', 'SPD', 'GUS','TEMP', 'MAX', 'MIN'], na_values=['*', '**', '***', '****', '*****', '******'])
data.head()

#A dictionary is used here to store new and old column names
# Create the dictionary with old and new names
new_names = {'YR--MODAHRMN': 'TIME', 'SPD': 'SPEED', 'GUS': 'GUST'}
#use the rename function with parameters column to rename
data = data.rename(columns=new_names)
data.shape
data.dtypes
print(data.describe())
#after definining the function create column TEMP F
data["TEMP_C"] = fahr_to_celsius(data["TEMP_F"])
data.head()

##iterating over rows
iterrows()
# Iterate over the rows
for idx, row in data.iterrows():
    # Print the index value
    print('Index:', idx)
    
    # Print the row
    print('Temp F:', row["TEMP_F"], "\n")
    #break stops the iteration!
    break

# Create an empty column for the DataFrame where the values will be stored
new_column = "TEMP_C"
data[new_column] = None

# Iterate over the rows 
for idx, row in data.iterrows():
    # Convert the Fahrenheit to Celsius
    celsius = fahr_to_celsius(row['TEMP_F'])
    
    # Update the value of 'Celsius' column with the converted value
    data.at[idx, new_column] = celsius
#Reminder: .at or .loc?
#Here, you could also use data.loc[idx, new_column] = celsius to achieve the same result.
#If you only need to access a single value in a DataFrame, DataFrame.at is faster compared to DataFrame.loc which is designed for accessing groups of rows and columns.
#Check out more examples for using .at and .loc from lesson 5 materials.

#check top 10. Bottom 20 would be data.tail(20)
data.head(10)

##BUT, you can also apply a function to rows or columns in pandas!
data["TEMP_C"] = data["TEMP_F"].apply(fahr_to_celsius)
data.head()
#Should I use .iterrows() or .apply()?
#We are teaching the .iterrows() method because it helps to understand the structure 
#of a DataFrame and the process of looping trough DataFrame rows. However, using .apply() is often more efficient in terms of execution time. 

data["TIME"].head(10)

##How to aggregate?
#string slicing
#1) Convert the TIME column from int into str datatype.
#2) Slice the correct range of characters from the character string using pandas.Series.str.slice()
#1
# Convert to string
data['TIME_STR'] = data['TIME'].astype(str)
#2
# SLice the string
data['YEAR_MONTH'] = data['TIME_STR'].str.slice(start=0, stop=6)

##However, let's have a look at a more clever way of dealing with dates and times.

##Pandas datetime (.to_datetime)
#In pandas, we can convert dates and times into a new data type datetime using pandas.to_datetime function.
# Convert to datetime
data["DATE"] = pd.to_datetime(data["TIME_STR"])
#if you check the data type .dtype(), it is a pandas datetime
data["DATE"].head()
#Now, we can extract different time units based on the datetime-column using the pandas.Series.dt accessor:
data['DATE'].dt.year
#or create new columns
data['YEAR'] = data['DATE'].dt.year
data['MONTH'] = data['DATE'].dt.month

##Aggregating data in Pandas by grouping
groupby()
#Our practical task is to calculate the average temperatures for each month
#Let's group our data based on unique year and month combinations
grouped = data.groupby(["YEAR", "MONTH"])
#or this would have done the same on the other column
grouped = data.groupby('YEAR_MONTH')
# What is the type?
print("Type:\n", type(grouped))

# How many?
print("Length:\n", len(grouped))

# Answer: the length of the grouped object should be the same as
data["YEAR_MONTH"].nunique()
#601 unique values means 601 unique combinations, ie., groups

# Check the "names" of each group (uncomment the next row if you want to print out all the keys)
grouped.groups.keys()

##accessing data from groups
# Specify the time of the first hour (as text)
month = (2019, 4)

# Select the group
group1 = grouped.get_group(month)

# Let's see what we have
print(group1)

#Ahaa! As we can see, a single group contains a DataFrame with values only for that specific month. Let's check the DataType of this group:
type(group1)
#pandas.core.frame.DataFrame (!) A group is a dataframe!

##Now we can perform statistics on these groups
# Specify the columns that will be part of the calculation
mean_cols = ['DIR', 'SPEED', 'GUST', 'TEMP_F', 'TEMP_C', 'MONTH']

# Calculate the mean values all at one go
mean_values = group1[mean_cols].mean()

# Let's see what we have
print(mean_values)

##OR LOOP THROUGH THEM!
#For-loops and grouped objects:
#When iterating over the groups in our DataFrameGroupBy -object it is important to understand that a single group 
#in our DataFrameGroupBy actually contains not only the actual values, but also information about the key that was 
#used to do the grouping. Hence, when iterating over the data we need to assign the key and the values into separate variables
# Iterate over groups
for key, group in grouped:
    # Print key and group
    print("Key:\n", key)
    print("\nFirst rows of data in this group:\n", group.head())
    
    # Stop iteration with break command
    break

# Create an empty DataFrame for the aggregated values
monthly_data = pd.DataFrame()

# The columns that we want to aggregate
mean_cols = ['DIR', 'SPEED', 'GUST', 'TEMP_F', 'TEMP_C', "MONTH"]

# Iterate over the groups to calculate the mean.
for key, group in grouped:
    
   # Calculate mean
   mean_values = group[mean_cols].mean()

   # Add the ´key´ (i.e. the date+time information) into the aggregated values
   mean_values['YEAR_MONTH'] = key

   # Append the aggregated values into the DataFrame
   monthly_data = monthly_data.append(mean_values, ignore_index=True)

#BUT, you could also get all the means at once with
grouped.mean()

##pulling it all together, for example which years had warmest aprils
#select all data with april
aprils = data[data["MONTH"]==4]
#select the columns of interest
aprils = aprils[['STATION_NUMBER','TEMP_F', 'TEMP_C','YEAR', 'MONTH']]
#group
grouped = aprils.groupby(by=["YEAR", "MONTH"])
#calculate mean
monthly_mean = grouped.mean()
monthly_mean.head()
#sort to find highest temperatures
monthly_mean.sort_values(by="TEMP_C", ascending=False).head(10)

##Repeating the data analysis with larger dataset!!!
#glob module for importing multiple data setsid
import glob
file_list = glob.glob(r'data/0*txt')
for fp in file_list:

    # Read selected columns of  data using varying amount of spaces as separator and specifying * characters as NoData values
    data = pd.read_csv(fp, delim_whitespace=True, usecols=['USAF','YR--MODAHRMN', 'DIR', 'SPD', 'GUS','TEMP', 'MAX', 'MIN'], na_values=['*', '**', '***', '****', '*****', '******'])

    # Rename the columns
    new_names = {'USAF':'STATION_NUMBER','YR--MODAHRMN': 'TIME', 'SPD': 'SPEED', 'GUS': 'GUST', 'TEMP':'TEMP_F'}
    data = data.rename(columns=new_names)

    #Print info about the current input file:
    print("STATION NUMBER:", data.at[0,"STATION_NUMBER"])
    print("NUMBER OF OBSERVATIONS:", len(data))

    # Create column
    col_name = 'TEMP_C'
    data[col_name] = None

    # Convert tempetarues from Fahrenheits to Celsius
    data["TEMP_C"] = data["TEMP_F"].apply(fahr_to_celsius)

    # Convert TIME to string 
    data['TIME_STR'] = data['TIME'].astype(str)

    # Parse year and month
    data['MONTH'] = data['TIME_STR'].str.slice(start=5, stop=6).astype(int)
    data['YEAR'] = data['TIME_STR'].str.slice(start=0, stop=4).astype(int)

    # Extract observations for the months of April 
    aprils = data[data['MONTH']==4]

    # Take a subset of columns
    aprils = aprils[['STATION_NUMBER','TEMP_F', 'TEMP_C', 'YEAR', 'MONTH']]

    # Group by year and month
    grouped = aprils.groupby(by=["YEAR", "MONTH"])

    # Get mean values for each group
    monthly_mean = grouped.mean()

    # Print info
    print(monthly_mean.sort_values(by="TEMP_C", ascending=False).head(5))
    print("\n")


###assertion errors
assert <some test>, 'Error message to display'
#for example
def convert_kph_ms(speed):
    """Converts velocity (speed) in km/hr to m/s"""
    assert speed >= 0.0, 'Wind speed values must be positive or zero'
    assert speed <= 408.0, 'Wind speed exceeds fastest winds ever measured'
    return speed * 1000 / 3600

wind_speed_km = 409
wind_speed_ms = convert_kph_ms(wind_speed_km)

print('A wind speed of', wind_speed_km, 'km/hr is', wind_speed_ms, 'm/s.')

###Ploting in Pandas
#introducing parse_dates and index_col on import.
fp = r"data/029740.txt"
data = pd.read_csv(fp, delim_whitespace=True, 
                   na_values=['*', '**', '***', '****', '*****', '******'],
                   usecols=['YR--MODAHRMN', 'TEMP', 'MAX', 'MIN'],
                   parse_dates=['YR--MODAHRMN'], index_col='YR--MODAHRMN')
                   
##basic x y plot
#to run in Jupyter
%matplotlib inline
#then, the plot is defined to a variable, ax here, so it can be modified
ax = data.plot()
#using type we can see that matlibplot is being used by pandas
type(ax)
#select only data from TEMP column and then certain dates (using the index we set on import!)
oct1_temps = data['TEMP'].loc[data.index >= '201910011200']
ax = oct1_temps.plot()
#but even better (In this case, r tells the oct1_temps.plot() function to use red color for the lines and symbols, o tells it to show circles at the points, and -- says to use a dashed line.)
ax = oct1_temps.plot(style='ro--', title='Helsinki-Vantaa temperatures')
ax.set_xlabel('Date')
ax.set_ylabel('Temperature [°F]')
#plotting help
help(oct1_temps.plot)
##embiggening the plot requires importing a package and adjusting the default plot size accordingly
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [12, 6]
##selecting what data to show. date strings have to be converted to datatime objects. .text adds text at (x,y, 'write what you want to see here')
start_time = pd.to_datetime('201910011200')
end_time = pd.to_datetime('201910011500')
cold_time = pd.to_datetime('201910011205')

ax = oct1_temps.plot(style='ro--', title='Helsinki-Vantaa temperatures',
                     xlim=[start_time, end_time], ylim=[40.0, 46.0])
ax.set_xlabel('Date')
ax.set_ylabel('Temperature [°F]')
ax.text(cold_time, 42.0, '<- Coldest temperature in early afternoon')
#or modified again
start_time = pd.to_datetime('201910011800')
end_time = pd.to_datetime('201910012359')
warm_time = pd.to_datetime('201910012350')

ax = oct1_temps.plot(color = 'black', linestyle = 'dashed', title ='Evening temperatures on October 1, Helsinki-Vantaa',
                     xlim=[start_time, end_time])
ax.set_xlabel('Date')
ax.set_ylabel('Temperature [°F]')
ax.text(warm_time, 43.0, '<- Warmest temperature in evening')
#help(ax.plot)
##Bar Plots
oct1_afternoon = oct1_temps.loc[oct1_temps.index <= '201910011500']
ax = oct1_afternoon.plot(kind='bar', title='Helsinki-Vantaa temperatures',
                         ylim=[40, 46])
ax.set_xlabel('Date')
ax.set_ylabel('Temperature [°F]')
ax.text(0, 42.0, 'Coldest \ntemp \nv')

## Saving plots as image
plt.savefig('bar-plot.png')
plt.savefig('bar-plot-hi-res.pdf', dpi=600)

##Pandas-Bokeh. Parameters are called differently (eg. x and y axis, style) and ONLY Dataframe (.plot on a Pandas Serie does not work, eg. data['TEMP'].plot()
ax = sept29_oct1_df.plot(title='Helsinki-Vantaa temperatures',
                         xlabel='Date', ylabel='Temperature [°F]',
                         xlim=[start_time, end_time], ylim=[35.0, 60.0],
                         plot_data_points=True)

```
# AutoGIS
```py
#new coding shortcuts I picked up. 
print("Distance between the points is {0:.2f} decimal degrees".format(point_dist))
assert value1 > 0, "'value1' needs to be higher than 0! Found: %s" % value1
print("Object data type:", type(line))
print("Geometry type as text:", line.geom_type)

#helpful code snippets
valid = multi_poly.is_valid
print("Is polygon valid?: ", valid)
#round result to 2 dp.
print(round(area1, 2))
# NON-EDITABLE CODE CELL FOR TESTING YOUR SOLUTION

# List all functions we created
functions = [create_point_geom, create_line_geom, create_poly_geom, get_centroid,
            get_area, get_length]

print("I created functions for doing these tasks:\n")

for function in functions:
    #Print function name and docstring:
    print("-", function.__name__ +":", function.__doc__)



###Lesson One: Shapely and geometric objects

# Import necessary geometric objects from shapely module
from shapely.geometry import Point, LineString, Polygon
##POINT
# Create Point geometric object(s) with coordinates
point1 = Point(2.2, 4.2)
type(point1)
We can also access the geometry type of the object using Point.geom_type:
point1.geom_type

##Point attributes and functions
# Get the coordinates
point_coords = point1.coords
print(point_coords)
# What is the data type? (it is a CoordinateSequence)
type(point_coords)
# Get x and y coordinates
xy = point_coords.xy

# Get only x coordinates of Point1
x = point1.x

# Whatabout y coordinate?
y = point1.y

# Calculate the distance between point1 and point2
point_dist = point1.distance(point2)

print("Distance between the points is {0:.2f} decimal degrees".format(point_dist))
##LINESTRING
# Create a LineString from our Point objects
line = LineString([point1, point2, point3])

# It is also possible to produce the same outcome using coordinate tuples
line2 = LineString([(2.2, 4.2), (7.2, -25.1), (9.26, -2.456)])

print("Object data type:", type(line))
print("Geometry type as text:", line.geom_type)
##LineString attributes and functions
# Get x and y coordinates of the line
lxy = line.xy

# Extract x coordinates
line_xcoords = lxy[0]

# Extract y coordinates straight from the LineObject by referring to a array at index 1
line_ycoords = line.xy[1]
# Get the lenght of the line
l_length = line.length

# Get the centroid of the line
l_centroid = line.centroid
# Print the outputs
print("Length of our line: {0:.2f}".format(l_length))
print("Centroid of our line: ", l_centroid)
print("Type of the centroid:", type(l_centroid))

##POLYGON
# Create a Polygon from the coordinates
poly = Polygon([(2.2, 4.2), (7.2, -25.1), (9.26, -2.456)])

# It is also possible to produce the same outcome using a list of lists which contain the point coordinates.
# We can do this using the point objects we created before and a list comprehension:
# --> here, we pass a list of lists as input when creating the Polygon (the linst comprehension generates this list: [[2.2, 4.2], [7.2, -25.1], [9.26, -2.456]]):
poly2 = Polygon([[p.x, p.y] for p in [point1, point2, point3]])
print('poly:', poly)
print('poly2:', poly2)
#Notice that polygons have double parentheses because polygons can have holes in them.
##creating a polygon with a hole
# First we define our exterior
world_exterior = [(-180, 90), (-180, -90), (180, -90), (180, 90)]

# Let's create a single big hole where we leave ten decimal degrees at the boundaries of the world
# Notice: there could be multiple holes, thus we need to provide a list of holes
hole = [[(-170, 80), (-170, -80), (170, -80), (170, 80)]]
# World without a hole
world = Polygon(shell=world_exterior)

# Now we can construct our Polygon with the hole inside
world_has_a_hole = Polygon(shell=world_exterior, holes=hole)

##Polygon attributes and functions
# Get the centroid of the Polygon
world_centroid = world.centroid

# Get the area of the Polygon
world_area = world.area

# Get the bounds of the Polygon (i.e. bounding box)
world_bbox = world.bounds

# Get the exterior of the Polygon
world_ext = world.exterior

# Get the length of the exterior
world_ext_length = world_ext.length

valid = multi_poly.is_valid
print("Is polygon valid?: ", valid)

###IMPORTING VECTOR DATA
#PostGIS
import geopandas as gpd
import psycopg2

# Create connection to database with psycopg2 module (update params according your db)
conn, cursor = psycopg2.connect(dbname='my_postgis_database', user='my_usrname', password='my_pwd', 
                                host='123.22.432.16', port=5432)

# Specify sql query
sql = "SELECT * FROM MY_TABLE;"

# Read data from PostGIS
data = gpd.read_postgis(sql=sql, con=conn)

#Shapefile
import geopandas as gpd

# Read file from Shapefile
fp = "L2_data/Finland.shp"
data = gpd.read_file(fp)

# Write to Shapefile (just make a copy)
outfp = "L2_data/Finland_copy.shp"
data.to_file(outfp)

#Managing File Paths
import os

# Define path to folder
input_folder = r"L2_data/NLS/2018/L4/L41/L4132R.shp"

# Join folder path and filename 
fp = os.path.join(input_folder, "m_L4132R_p.shp")

# Print out the full file path
print(fp)

#refresh
print("Number of rows", len(data['CLASS']))
print("Number of classes", data['CLASS'].nunique())
print("Number of groups", data['GROUP'].nunique())

# Print all unique values in the column
print(data['CLASS'].unique())

#writng a shapefile (.shp is default)
# Select a class
selection = data.loc[data["CLASS"]==36200]

# Write those rows into a new file (the default output file format is Shapefile)
selection.to_file(output_filepath)

##Grouping the Geodataframe
# Group the data by class
grouped = data.groupby('CLASS')

# Let's see what we have. Each group is actually a seperate Geodataframe
grouped

#look at grouped keys (the same as the unique classes!
grouped.groups.keys()

#look at how many rows of data each group has
#Iterate over the group object
for key, group in grouped:

    # Let's check how many rows each group has:
    print('Terrain class:', key)
    print('Number of rows:', len(group), "\n")

##Saving multiple output data files (ie., each grouped geodataframe)
#string formatting
basename = "terrain"
key = 36200

# OPTION 1. Concatenating using the `+` operator:
out_fp = basename + "_" + str(key) + ".shp"

# OPTION 2. Positional formatting using `%` operator
out_fp = "%s_%s.shp" %(basename, key)

# OPTION 3. Positional formatting using `.format()`
out_fp = "{}_{}.shp".format(basename, key)

#create new folder for the outputsCollapsed
# Determine output directory
output_folder = r"L2_data/"

# Create a new folder called 'Results' 
result_folder = os.path.join(output_folder, 'Results')

# Check if the folder exists already
if not os.path.exists(result_folder):
    # If it does not exist, create one
    os.makedirs(result_folder)

# Iterate over the groups
for key, group in grouped:
    # Format the filename 
    output_name = "terrain_%s.shp" % str(key)

    # Print information about the process
    print("Saving file", os.path.basename(output_name))

    # Create an output path
    outpath = os.path.join(result_folder, output_name)

    # Export the data
    group.to_file(outpath)
    
#save to csv ie., text 
area_info = grouped.area.sum().round()

# Create an output path
area_info.to_csv("terrain_class_areas.csv", header=True)

##SET CRS

#my_geoseries.crs = "EPSG:4326"

##Reprojecting coordinate systems
# Let's make a backup copy of our data
data_wgs84 = data.copy()
#The function has two alternative parameters 1) crs and 2) epgs that can be used to make the coordinate transformation and re-project the data into the CRS that you want to use. 
# Reproject the data
data = data.to_crs(epsg=3035)
#plotting the two different coordinate systems
%matplotlib inline
import matplotlib.pyplot as plt

# Make subplots that are next to each other
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))

# Plot the data in WGS84 CRS
data_wgs84.plot(ax=ax1, facecolor='gray');

# Add title
ax1.set_title("WGS84");

# Plot the one with ETRS-LAEA projection
data.plot(ax=ax2, facecolor='blue');

# Add title
ax2.set_title("ETRS Lambert Azimuthal Equal Area projection");

# Remove empty white space around the plot
plt.tight_layout()
#Dealing with different CRS formats

### Import CRS class from pyproj
from pyproj import CRS

# Initialize the CRS class for epsg code 3035:
crs_object = CRS.from_epsg(3035)
crs_object

# Re-define the CRS of the input GeoDataFrame 
data.crs = CRS.from_epsg(3035).to_wkt()

# Retrive CRS information in WKT format
crs_wkt = crs_object.to_wkt()
print(crs_wkt)
#summary of CRS objects
# PROJ dictionary:
crs_dict = data_wgs84.crs

# pyproj CRS object:
crs_object = CRS(data_wgs84.crs)

# EPSG code (here, the input crs information is a bit vague so we need to lower the confidence threshold)
crs_epsg = CRS(data_wgs84.crs).to_epsg(min_confidence=25)

# PROJ string
crs_proj4 = CRS(data_wgs84.crs).to_proj4()

# Well-Known Text (WKT)
crs_wkt = CRS(data_wgs84.crs).to_wkt()

##Calculating Distances
# Create the point representing Helsinki (in WGS84)
hki_lon = 24.9417
hki_lat = 60.1666

# Create GeoDataFrame
helsinki = gpd.GeoDataFrame([[Point(hki_lon, hki_lat)]], geometry='geometry', crs={'init': 'epsg:4326'}, columns=['geometry'])

# Print 
print(helsinki)

##Create you own CRS with Helsinki as the centre.

# Define the projection using the coordinates of our Helsinki point (hki_lat, hki_lon) as the center point
# The .srs here returns the text presentation of the projection
aeqd = CRS(proj='aeqd', ellps='WGS84', datum='WGS84', lat_0=hki_lat, lon_0=hki_lon).srs

# Reproject to aeqd projection using Proj4-string
helsinki = helsinki.to_crs(crs=aeqd)

# Print the data
print(helsinki)

# Print the crs
print('\nCRS:\n', helsinki.crs)

    proj='aeqd' refers to projection specifier that we determine to be Azimuthal Equidistant ('aeqd')
    ellps='WGS84' refers to the reference ellipsoid that is a mathematically modelled (based on measurements) surface that approximates the true shape of the world. World Geodetic System (WGS) was established in 1984, hence the name.
    datum='WGS84' refers to the Geodetic datum that is a coordinate system constituted with a set of reference points that can be used to locate places on Earth.
    lat_0 is the latitude coordinate of the center point in the projection
    lon_0 is the longitude coordinate of the center point in the projection

##Calculating Distances

#.apply() is an alternative way than irrating through rows with iterrows()
#It is the recommendable way of iterating over rows in a Pandas geodataframe
def calculate_distance(row, dest_geom, src_col='geometry', target_col='distance'):
    """
    Calculates the distance between Point geometries.

    Parameters
    ----------
    dest_geom : shapely.Point
       A single Shapely Point geometry to which the distances will be calculated to.
    src_col : str
       A name of the column that has the Shapely Point objects from where the distances will be calculated from.
    target_col : str
       A name of the target column where the result will be stored.

    Returns
    -------

    Distance in kilometers that will be stored in 'target_col'.
    """

    # Calculate the distances
    dist = row[src_col].distance(dest_geom)

    # Convert into kilometers
    dist_km = dist / 1000

    # Assign the distance to the original data
    row[target_col] = dist_km
    return row
    
# Retrieve the geometry from Helsinki GeoDataFrame
helsinki_geom = helsinki.loc[0, 'geometry']
print(helsinki_geom)

# Calculate the distances using our custom function called 'calculate_distance'
europe_borders_aeqd = europe_borders_aeqd.apply(calculate_distance, dest_geom=helsinki_geom, src_col='centroid', target_col='dist_to_Hki', axis=1)
print(europe_borders_aeqd.head(10))

# Calculat the maximum and average distance
max_dist = europe_borders_aeqd['dist_to_Hki'].max()
mean_dist = europe_borders_aeqd['dist_to_Hki'].mean()

print("Maximum distance to Helsinki is %.0f km, and the mean distance is %.0f km." % (max_dist, mean_dist))

##CREATING NEW LAYERS
#Since geopandas takes advantage of Shapely geometric objects, it is possible to create spatial data from a scratch by passing Shapely’s geometric objects into the GeoDataFrame.
import geopandas as gpd
from shapely.geometry import Point, Polygon
from pyproj import CRS
newdata = gpd.GeoDataFrame()
# Create a new column called 'geometry' to the GeoDataFrame
newdata['geometry'] = None
# Coordinates of the Helsinki Senate square in decimal degrees
coordinates = [(24.950899, 60.169158), (24.953492, 60.169158), (24.953510, 60.170104), (24.950958, 60.169990)]
# Create a Shapely polygon from the coordinate-tuple list
poly = Polygon(coordinates)
# Insert the polygon into 'geometry' -column at row 0
newdata.at[0, 'geometry'] = poly
# Add a new column and insert data 
newdata.at[0, 'location'] = 'Senaatintori'
# Let's check the data
print(newdata)
# Set the GeoDataFrame's coordinate system to WGS84 (i.e. epsg code 4326)
newdata.crs = CRS.from_epsg(4326).to_wkt()

# Determine the output path for the Shapefile
outfp = "L2_data/Senaatintori.shp"
# Write the data into that Shapefile (default Shapefile)
newdata.to_file(outfp)
##Alternatives to iterrows for creating geometries from x and y columns in a DataFrame
# OPTION 2: apply a function

# Define a function for creating points from row values
def create_point(row):
    '''Returns a shapely point object based on values in x and y columns'''

    point = Point(row['x'], row['y'])

    return point

# Apply the function to each row
df['geometry'] = df.apply(create_point, axis=1)

#-----------------------------------------


# OPTION 3: apply a lambda function
# see: https://docs.python.org/3.5/tutorial/controlflow.html#lambda-expressions

df['geometry'] = df.apply(lambda row: Point(row['x'], row['y']), axis=1)

#-----------------------------------------

# OPTION 4: zip and for-loop

geom = []
for x, y in zip(df['x'], df['y']):
    geom.append(Point(x, y))

df['geometry'] = geom
#Adding items to a GeoDataFrame
# Add a point object into the geometry-column on the first row (here, the row-label is 0)
df.at[0, 'geometry'] = point
As an alternative, you can also add new rows of data using the append method.

###Week 3
##Geocoding
# Import the geocoding tool
from geopandas.tools import geocode

# Geocode addresses using Nominatim. Remember to provide a custom "application name" in the user_agent parameter!
geo = geocode(data['addr'], provider='nominatim', user_agent='autogis_xx', timeout=4)


#Rate-limiting
#When geocoding a large dataframe, you might encounter an error when geocoding. In case you get a time out error, try first using the timeout parameter as we did above (allow the service a bit more time to respond). In case of Too Many Requests error, you have hit the rate-limit of the service, and you should slow down your requests. To our convenience, GeoPy provides additional tools for taking into account rate limits in geocoding services. This script adapts the usage of GeoPy RateLimiter to our input data:
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from shapely.geometry import Point

# Initiate geocoder
geolocator = Nominatim(user_agent='autogis_xx')

# Create a geopy rate limiter:
geocode_with_delay = RateLimiter(geolocator.geocode, min_delay_seconds=1)

# Apply the geocoder with delay using the rate limiter:
data['temp'] = data['addr'].apply(geocode_with_delay)

# Get point coordinates from the GeoPy location object on each row:
data["coords"] = data['temp'].apply(lambda loc: tuple(loc.point) if loc else None)

# Create shapely point objects to geometry column:
data["geometry"] = data["coords"].apply(Point)

#All in all, remember that Nominatim is not meant for super heavy use.

##Table Joins
#in exercise 6 we joined using merge() which works on a common KEY attribute between tables, a name for example. 
# .join() can be used on an index provided 1) the number of records is the same and 2) the order is the same.

#Point in Polygon (PIP)
# Check if p1 is within the polygon using the within function
p1.within(poly)
#Does polygon contain p1?
poly.contains(p1)

#intersect
line_a.intersects(line_b)
line_a.touches(line_b)

#import shapely.speedups
from shapely import speedups
speedups.enabled
pip_mask = data.within(southern.at[0, 'geometry'])
print(pip_mask)
#the below will return the point in polygon data
pip_data = data.loc[pip_mask]
pip_data

##Spatial Indexes
#More plotting code
%matplotlib inline
ax = postcode_areas.plot(color='red', edgecolor='black', alpha=0.5)
ax = intersections.plot(ax=ax, color='yellow', markersize=1, alpha=0.5)
# Zoom to closer (comment out the following to see the full extent of the data)
#ax.set_xlim([380000, 395000])
#ax.set_ylim([6667500, 6680000])

# Let's build spatial index for intersection points
intersection_sindex = intersections.sindex
# Let's see what it is (<geopandas.sindex.SpatialIndex at 0x7f5776feba10>) is the return
intersection_sindex

# How many groups do we have?
print("Number of groups:", len(intersection_sindex.leaves()), '\n')

# Print some basic info for few of them
n_iterations = 10
for i, group in enumerate(intersection_sindex.leaves()):
    group_idx, indices, bbox = group
    print("Group", group_idx, "contains ", len(indices), "geometries, bounding box:", bbox)
    i+=1
    if i == n_iterations:
        break
# Get the bounding box coordinates of the Polygon as a list
bounds = list(city_center_zip_area.bounds.values[0])

# Get the indices of the Points that are likely to be inside the bounding box of the given Polygon
point_candidate_idx = list(intersection_sindex.intersection(bounds))
point_candidates = intersections.loc[point_candidate_idx]

# Let's see what we have now
ax = city_center_zip_area.plot(color='red', alpha=0.5)
ax = point_candidates.plot(ax=ax, color='black', markersize=2)

# Make the precise Point in Polygon query
final_selection = point_candidates.loc[point_candidates.intersects(city_center_zip_area['geometry'].values[0])]

# Let's see what we have now
ax = city_center_zip_area.plot(color='red', alpha=0.5)
ax = final_selection.plot(ax=ax, color='black', markersize=2)

##Trick for importing Zipped folders (geopackage for example) using ZipFile
import geopandas as gpd
from zipfile import ZipFile
import io

def read_gdf_from_zip(zip_fp):
    """
    Reads a spatial dataset from ZipFile into GeoPandas. Assumes that there is only a single file (such as GeoPackage) 
    inside the ZipFile.
    """
    with ZipFile(zip_fp) as z:
        # Lists all files inside the ZipFile, here assumes that there is only a single file inside
        layer = z.namelist()[0]
        data = gpd.read_file(io.BytesIO(z.read(layer)))
    return data

# Filepaths
stops = gpd.read_file('data/pt_stops_helsinki.gpkg')
buildings = read_gdf_from_zip('data/building_points_helsinki.zip')

##Spatial Join 
# Make a spatial join
join = gpd.sjoin(addresses, pop, how="inner", op="within")

#read data from WFS wfs
import geopandas as gpd
from pyproj import CRS
import requests
import geojson

# Specify the url for web feature service
url = 'https://kartta.hsy.fi/geoserver/wfs'

# Specify parameters (read data in json format).
# Available feature types in this particular data source: http://geo.stat.fi/geoserver/vaestoruutu/wfs?service=wfs&version=2.0.0&request=describeFeatureType
params = dict(service='WFS',
              version='2.0.0',
              request='GetFeature',
              typeName='asuminen_ja_maankaytto:Vaestotietoruudukko_2018',
              outputFormat='json')

# Fetch data from WFS using requests
r = requests.get(url, params=params)

# Create GeoDataFrame from geojson
pop = gpd.GeoDataFrame.from_features(geojson.loads(r.content))

##Lesson 4

# Ensure that the CRS matches, if not raise an AssertionError
assert hel.crs == grid.crs, "CRS differs between layers!"

# Plot the layers
ax = grid.plot(facecolor='gray')
hel.plot(ax=ax, facecolor='None', edgecolor='blue')

#overlay analysis using intersection
intersection = gpd.overlay(grid, hel, how='intersection')

# Output filepath
outfp = "data/TravelTimes_to_5975375_RailwayStation_Helsinki.geojson"

# Use GeoJSON driver
intersection.to_file(outfp, driver="GeoJSON")

##Aggregating Data (using dissolve()
# Conduct the aggregation
dissolved = intersection.dissolve(by="car_r_t")
print(dissolved.columns)
print(dissolved.index)
# Select only geometries that are within 15 minutes away (iloc is an index lock)
dissolved.iloc[15]
#to visualise convert the selected row back to a GeoDataFrame:
# Create a GeoDataFrame
selection = gpd.GeoDataFrame([dissolved.iloc[15]], crs=dissolved.crs)

## simplyfying geometries (generalise)
import geopandas as gpd

# File path
fp = "data/Amazon_river.shp"
data = gpd.read_file(fp)

# Print crs
print(data.crs)

# Plot the river
data.plot();

# Generalize geometry
data['geom_gen'] = data.simplify(tolerance=20000)

# Set geometry to be our new simlified geometry
data = data.set_geometry('geom_gen')

# Plot 
data.plot()

##Reclassifying data
import geopandas as gpd

fp = "data/TravelTimes_to_5975375_RailwayStation_Helsinki.geojson"

# Read the GeoJSON file similarly as Shapefile
acc = gpd.read_file(fp)

# Let's see what we have
print(acc.head(2))

#No data values are -1, so remove these
# Include only data that is above or equal to 0
acc = acc.loc[acc['pt_r_tt'] >=0]

%matplotlib inline
import matplotlib.pyplot as plt

#Mapclassify module for schemes
# Plot using 9 classes and classify the values using "Natural Breaks" classification
acc.plot(column="pt_r_tt", scheme="Natural_Breaks", k=9, cmap="RdYlBu", linewidth=0, legend=True)

# Use tight layout
plt.tight_layout()

mapclassify.Quantiles(y=acc['pt_r_tt'])
classifier = mapclassify.NaturalBreaks(y=acc['pt_r_tt'], k=9)
classifier.bins

# Create a Natural Breaks classifier
classifier = mapclassify.NaturalBreaks.make(k=9)

# Classify the data
classifications = acc[['pt_r_tt']].apply(classifier)

# Let's see what we have
classifications.head()

# Rename the column so that we know that it was classified with natural breaks
acc['nb_pt_r_tt'] = acc[['pt_r_tt']].apply(classifier)

# Check the original values and classification
acc[['pt_r_tt', 'nb_pt_r_tt']].head()

#plotting a histogram 
# Histogram for public transport rush hour travel time
acc['pt_r_tt'].plot.hist(bins=50)

#creating a custom classifier
def custom_classifier(row, src_col1, src_col2, threshold1, threshold2, output_col):
   # 1. If the value in src_col1 is LOWER than the threshold1 value
   # 2. AND the value in src_col2 is HIGHER than the threshold2 value, give value 1, otherwise give 0
   if row[src_col1] < threshold1 and row[src_col2] > threshold2:
       # Update the output column with value 0
       row[output_col] = 1
   # If area of input geometry is higher than the threshold value update with value 1
   else:
       row[output_col] = 0

   # Return the updated row
   return row

# Create column for the classification results
acc["suitable_area"] = None

# Use the function
acc = acc.apply(custom_classifier, src_col1='pt_r_tt', 
                src_col2='walk_d', threshold1=20, threshold2=4000, 
                output_col="suitable_area", axis=1)

# See the first rows
acc.head(2)

# Get value counts
acc['suitable_area'].value_counts()

# Plot
acc.plot(column="suitable_area", linewidth=0);

# Use tight layour
plt.tight_layout()

#creating a custom binary classification function
def binaryClassifier(row, source_col, output_col, threshold):
    # If area of input geometry is lower that the threshold value
    if row[source_col] < threshold:
        # Update the output column with value 0
        row[output_col] = 0
    # If area of input geometry is higher than the threshold value update with value 1
    else:
        row[output_col] = 1
    # Return the updated row
    return row
lakes['small_big'] = None
lakes = lakes.apply(binaryClassifier, source_col='area_km2', output_col='small_big', threshold=l_mean_size, axis=1)

###Lesson 5
##Static Maps
#how to download a file from the web
$ cd /home/jovyan/work/autogis/notebooks/notebooks/L5
$ wget https://github.com/Automating-GIS-processes/Lesson-5-Making-Maps/raw/master/data/dataE5.zip
$ unzip dataE5.zip -d data

# Reproject geometries to ETRS89 / TM35FIN based on the grid crs:
roads = roads.to_crs(crs=grid.crs)
metro = metro.to_crs(crs=grid.crs)

# For better control of the figure and axes, use the plt.subplots function before plotting the layers
# https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/subplots_demo.html#a-figure-with-just-one-subplot

# Control figure size in here
fig, ax = plt.subplots(figsize=(12,8))

# Visualize the travel times into 9 classes using "Quantiles" classification scheme
grid.plot(ax=ax, column="car_r_t", linewidth=0.03, cmap="Spectral", scheme="quantiles", k=9, alpha=0.9)

# Add roads on top of the grid
# (use ax parameter to define the map on top of which the second items are plotted)
roads.plot(ax=ax, color="grey", linewidth=1.5)

# Add metro on top of the previous map
metro.plot(ax=ax, color="red", linewidth=2.5)

# Remove the empty white-space around the axes
plt.tight_layout()

# Save the figure as png file with resolution of 300 dpi
outfp = "static_map2.png"
plt.savefig(outfp, dpi=300)

#ADDING BASEMAPS FROM EXTERNAL SOURCE
import contextily as ctx

# Control figure size in here
fig, ax = plt.subplots(figsize=(12,8))

# Plot the data
data.plot(ax=ax, column='pt_r_t', cmap='RdYlBu', linewidth=0, scheme="quantiles", k=9, alpha=0.6)

# Add basemap 
ctx.add_basemap(ax)

#list of the basic url-addresses for different providers and styles
dir(ctx.tile_providers)

# Add basemap with `ST_TONER` style
ctx.add_basemap(ax, url=ctx.tile_providers.OSM_A)

# Control figure size in here
fig, ax = plt.subplots(figsize=(12,8))

# Plot the data
data.plot(ax=ax, column='pt_r_t', cmap='RdYlBu', linewidth=0, scheme="quantiles", k=9, alpha=0.6)

# Add basemap with `OSM_A` style using zoom level of 11 
# Modify the attribution 
ctx.add_basemap(ax, zoom=11, attribution="Travel time data by Digital Geography Lab, Map Data © OpenStreetMap contributors", url=ctx.tile_providers.OSM_A)

# Crop the figure
ax.set_xlim(2760000, 2800000)
ax.set_ylim(8430000, 8470000)

##or from other providers, requires parsing a url with a particular format the provider requires
# Control figure size in here
fig, ax = plt.subplots(figsize=(12,8))

# The formatting should follow: 'https://{s}.basemaps.cartocdn.com/{style}/{z}/{x}/{y}{scale}.png'
# Specify the style to use
style = "rastertiles/voyager"
cartodb_url = 'https://a.basemaps.cartocdn.com/%s/{z}/{x}/{y}.png' % style

# Plot the data from subset
subset.plot(ax=ax, column='pt_r_t', cmap='RdYlBu', linewidth=0, scheme="quantiles", k=5, alpha=0.6)
    
# Add basemap with `OSM_A` style using zoom level of 14 
ctx.add_basemap(ax, zoom=14, attribution="", url=cartodb_url)

# Crop the figure
ax.set_xlim(2770000, 2785000)
ax.set_ylim(8435000, 8442500)

##INTERACTIVE MAPS
#Folium, https://python-visualization.github.io/folium/modules.html#folium.folium.Map
#Leaflet providers http://leaflet-extras.github.io/leaflet-providers/preview/

import folium

# Create a Map instance
m = folium.Map(location=[60.25, 24.8], zoom_start=10, control_scale=True)
outfp = "base_map.html"
m.save(outfp)

##Employment Rates in Finland
# Read in data
data = pd.read_csv("data/seutukunta_tyollisyys_2013.csv", sep=",")
data.head()
# A layer saved to GeoJson in QGIS..
#geodata = gpd.read_file('Seutukunnat_2018.geojson')

# Get features directly from the wfs
url = "http://geo.stat.fi/geoserver/tilastointialueet/wfs?request=GetFeature&typename=tilastointialueet:seutukunta1000k_2018&outputformat=JSON"
geodata = gpd.read_file(url)
#region codes in the csv contain additional letters "SK" which we need to remove before the join
data["seutukunta"] = data["seutukunta"].apply(lambda x: x[2:])
data["seutukunta"].head()
#join employment stats to geodata (adminstrative areas)
#print info
print("Count of original attributes:", len(data))
print("Count of original geometries:", len(geodata))

# Merge data
geodata = geodata.merge(data, on = "seutukunta")

#Print info
print("Count after the join:", len(geodata))

geodata.head()
#Static map of employment in Finland
# Adjust figure size
fig, ax = plt.subplots(1, figsize=(10, 8))

# Adjust colors and add a legend
geodata.plot(ax = ax, column="tyollisyys", scheme="quantiles", cmap="Reds", legend=True)
#Interactive Map employment in Finland
# Create a Geo-id which is needed by the Folium (it needs to have a unique identifier for each row)
geodata['geoid'] = geodata.index.astype(str)

# Create a Map instance
m = folium.Map(location=[60.25, 24.8], tiles = 'cartodbpositron', zoom_start=8, control_scale=True)

folium.Choropleth(geo_data = geodata, 
                  data = geodata, 
                  columns=['geoid','tyollisyys'], 
                  key_on='feature.id', 
                  fill_color='RdYlBu', 
                  line_color='white',
                  line_weight=0,
                  legend_name= 'Employment rate in Finland').add_to(m)
m

#add tooltips
folium.features.GeoJson(geodata, name='Labels',
               style_function=lambda x: {'color':'transparent','fillColor':'transparent','weight':0},
                tooltip=folium.features.GeoJsonTooltip(fields=['tyollisyys'],
                                              aliases = ['Employment rate'],
                                              labels=True,
                                              sticky=False
                                             )
                       ).add_to(m)

m

#styling function with classification scheme
import branca

# Create a series (or a dictionary?) out of the variable that you want to map
employed_series = data.set_index('seutukunta')['tyollisyys']

# Setl colorscale
colorscale = branca.colormap.linear.RdYlBu_05.to_step(data = geodata['tyollisyys'], n = 6, method = 'quantiles')

#Define style function
def my_color_function(feature):
    
   employed = employed_series.get(int(feature['id']), None)

   return {
       'fillOpacity': 0.5,
       'weight': 0,
       'fillColor': '#black' if employed is None else colorscale(employed)
       }

#splice a string in a column 
data["seutukunta"] = data["seutukunta"].apply(lambda x: x[2:])
data["seutukunta"].head()

###OSMnx
import osmnx as ox
import matplotlib.pyplot as plt
%matplotlib inline

# Specify the name that is used to seach for the data
place_name = "Kamppi, Helsinki, Finland"

# Fetch OSM street network from the location
graph = ox.graph_from_place(place_name)
type(graph)
# Plot the streets
fig, ax = ox.plot_graph(graph)
#create a polygon from the placename (smart!)
area = ox.gdf_from_place(place_name)
#create polygons of the footprints of buildings from the place name
buildings = ox.footprints_from_place(place_name)
# Retrieve restaurants using the pois_from_place function and defining the tags from here:https://wiki.openstreetmap.org/wiki/Key:amenity
#other OSM tags include (e.g., `amenity`, `landuse`, `highway`, etc). Pass as dictionary.
restaurants = ox.pois_from_place(place_name, tags={'amenity':'restaurant'})
# Select some useful cols and print
cols = ['name', 'opening_hours', 'addr:city', 'addr:country', 
        'addr:housenumber', 'addr:postcode', 'addr:street']
# Print only selected cols
restaurants[cols].head(10)

#Now we can plot all of thee layers. Note the graph was not a gdf type as the other but a DiGraph type so this has to be converted
# Retrieve nodes and edges from 
nodes, edges = ox.graph_to_gdfs(graph)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12,8))

# Plot the footprint
area.plot(ax=ax, facecolor='black')

# Plot street edges
edges.plot(ax=ax, linewidth=1, edgecolor='#BC8F8F')

# Plot buildings
buildings.plot(ax=ax, facecolor='khaki', alpha=0.7)

# Plot restaurants
restaurants.plot(ax=ax, color='green', alpha=0.7, markersize=10)
plt.tight_layout()

#now reproject to a local projectionAcronym
from pyproj import CRS

# Set projection
projection = CRS.from_epsg(3067)

# Re-project layers
area = area.to_crs(projection)
edges = edges.to_crs(projection)
buildings = buildings.to_crs(projection)
restaurants = restaurants.to_crs(projection)

#not plotting with the local projection looks much better!
fig, ax = plt.subplots(figsize=(12,8))

# Plot the footprint
area.plot(ax=ax, facecolor='black')

# Plot street edges
edges.plot(ax=ax, linewidth=1, edgecolor='dimgray')

# Plot buildings
buildings.plot(ax=ax, facecolor='silver', alpha=0.7)

# Plot restaurants
restaurants.plot(ax=ax, color='yellow', alpha=0.7, markersize=10)
plt.tight_layout()

```

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing  # For preprocessing tasks
import matplotlib.pyplot as pltstrea
import seaborn as sns
import plotly.express as px
import streamlit as st
import io
from sklearn.preprocessing import OrdinalEncoder
import category_encoders as ce
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import plot_tree
from sklearn.neighbors import KNeighborsClassifier

def load_data(uploaded_file):
    """Reads the uploaded dataset and returns a Pandas DataFrame."""
    try:
        df = pd.read_csv(uploaded_file)  # Adjust for other file types if needed
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None
    
def handle_missing_values(df, method="fill_mean"):
    if(method=="drop_tuples"):
        df = df.dropna()
        pass

    if(method=="fill_with_mean"):
        try:
            df.fillna(df.mean(), inplace=True)
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Cant perform operation without numeric value: {e}")
            return None
        pass

    if(method=="fill_with_zero"):
        df = df. fillna (0)
        st.dataframe(df.head())
        pass
        
    if(method=="interpolate"):
        df = df. interpolate ()
        st.dataframe(df.head())
        pass
    # Display data info
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    pass


# Define additional functions for other preprocessing tasks

def create_distplot(df, column):
    # Create a Graph Plot visualization for the specified column
    plot=sns.distplot(df[column], bins=10, kde=True, rug=False)
    st.pyplot(plot.get_figure())
    pass


# Set up the main layout
st.title("Datavize")

#-------------------------------------
#-------------------------------------
#---------------UI--------------------
#-------------------------------------
#-------------------------------------
#-------------------------------------

# Uploading data
uploaded_file = st.file_uploader("Choose a dataset to upload")
if uploaded_file is not None:
    df = load_data(uploaded_file)
    # Display data preview
    st.dataframe(df.head())
    # Display data info
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)


#---------------------------selecting operations------------------
#--------------------------Preprocessing options------------------
    preprocess_options = st.multiselect("Select preprocessing operations:", ["Handle missing values", "Outlier detection", "Data Transformation"])
#-------------------------------------------
#--------Handiling missing values-----------
#-------------------------------------------
    if "Handle missing values" in preprocess_options:
        # Handle missing values based on user input
        preprocess_options = st.multiselect("Select method to remove missing values:", ["Drop Tuples", "Fill with mean", "Fill with Zero", "Interpolate Null Values"])
        if "Drop Tuples" in preprocess_options:
             handle_missing_values(df, method="drop_tuples")
             pass
        if "Fill with mean" in preprocess_options:
             handle_missing_values(df, method="fill_with_mean")
             pass
        if "Fill with Zero" in preprocess_options:
             handle_missing_values(df, method="fill_with_zero")
             pass
        if "Interpolate Null Values" in preprocess_options:
             handle_missing_values(df, method="interpolate")
             pass
        
#-------------------------------------------
#------------OutLier Detection--------------
#-------------------------------------------
    if "Outlier detection" in preprocess_options:
        try:
            Values = st.selectbox("Choose a column", df.columns)
            # Calculate IQR (Interquartile Range)
            Q1 = df[Values].quantile(0.25)
            Q3 = df[Values].quantile(0.75)
            IQR = Q3 - Q1
            # Define the lower and upper bounds for outliers
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            # Detect outliers
            outliers = df[(df[Values] < lower_bound) | (df[Values] > upper_bound)]
            st.write(outliers)
        except Exception as e:
            st.error(f"Enter Column Name: {e}")
        pass

#-------------------------------------------
#------------Data transformation------------
#-------------------------------------------
    if "Data Transformation" in preprocess_options:
        # Handle noisy data based on user input
        preprocess_options = st.multiselect("Select method to remove missing values:", ["Normalization with mix maxscaling", "One Hot Encoding", "Ordinal Encoding", "Nominal Encoding"])
        if "Normalization with mix maxscaling" in preprocess_options:
             for column in df.columns:
                  df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
                  st.dataframe(df.head())
             pass
        if "One Hot Encoding" in preprocess_options:
             st.dataframe(df.head())
             try:
                 # define one hot encoding
                 column = st.selectbox("Choose a column for the Graph Plot:", df.columns)
                 one_hot_encoded = pd.get_dummies(df[column], prefix=column)
                 # Concatenate the one-hot encoded columns with the original DataFrame
                 df = pd.concat([df, one_hot_encoded], axis=1)
                 st.dataframe(df.head())
             except Exception as e:
                 st.error(f"Error loading data: {e}")
             pass
        if "Ordinal Encoding" in preprocess_options:
             st.dataframe(df.head())
             try:
                 # define one hot encoding
                 column = st.selectbox("Choose a column for the Graph Plot:", df.columns)
                 # Create an ordinal encoder instance
                 ordinal_encoder = ce.OrdinalEncoder(cols=[column])
                 # Apply ordinal encoding on the 'Size' column
                 df[column+'_encoded'] = ordinal_encoder.fit_transform(df[column])

                 st.dataframe(df.head())
             except Exception as e:
                 st.error(f"Error loading data: {e}")
             pass
        if "Nominal Encoding" in preprocess_options:
             st.dataframe(df.head())
             try:
                 column = st.selectbox("Choose a column for the Graph Plot:", df.columns)
                 # Perform one-hot encoding (Nominal encoding) on the 'Color' column
                 nominal_encoded = pd.get_dummies(df[column], prefix=column)
                 # Concatenate the nominal encoded columns with the original DataFrame
                 df = pd.concat([df, nominal_encoded], axis=1)
                 st.dataframe(df.head())
             except Exception as e:
                 st.error(f"Error loading data: {e}")
             pass
        pass
# ... implement other preprocessing options


#-------------------------------------------------------------
#----------------------Visualization options------------------
#-------------------------------------------------------------
    visualization_options = st.multiselect("Select visualizations:", ["Graph Plot on full DataSet","Graph Plot", 
                                                                      "Box Plot", "Violin Plot", "Line Plot", "Bar Plot", "Scatter Plot",
                                                                      "Scatter Plot between Two Attributes","Multivariate Analysis", "Pair Plot"])
    if "Graph Plot" in visualization_options:
        column = st.selectbox("Choose a column for the Graph Plot:", df.columns)
        create_distplot(df, column)
        pass

    if "Graph Plot on full DataSet" in visualization_options:
        # Create a Graph Plot visualization for the specified column
        plot=sns.distplot(df, bins=10, kde=True, rug=False)
        st.pyplot(plot.get_figure())
        pass

    if "Box Plot" in visualization_options:
        column = st.selectbox("Choose a column for the Box Plot:", df.columns)
        # Create a Graph Plot visualization for the specified column
        plt.boxplot(df[column])
        plt.xlabel(column)
        plt.ylabel('Value')
        plt.title('Boxplot of '+column)
        st.pyplot(plt.gcf())
        pass

    if "Violin Plot" in visualization_options:
        column = st.selectbox("Choose a column for the Violin Plot:", df.columns)
        try:
            selected_attribute_values = df[column]
            # Create a violin plot
            sns.violinplot(x=selected_attribute_values)  # Create the violin plot
            plt.title('Violin Plot of '+column)  # Set the title
            plt.xlabel(column)  # Set the x-axis label
            plt.ylabel('Values')  # Set the y-axis label
            st.pyplot(plt.gcf())
        except Exception as e:
                 st.error(f"Error loading data: {e}")
        pass

    if "Line Plot" in visualization_options:
        column = st.selectbox("Choose a column for the Line Plot:", df.columns)
        try:
            attribute_values = df[column]
            # Plotting a line plot
            plt.plot(attribute_values, marker='o', linestyle='-', color='b')  # Plotting the line
            plt.title('Line Plot of '+column)  # Set the title
            plt.xlabel('Index')  # Set the x-axis label
            plt.ylabel(column)  # Set the y-axis label
            plt.grid(True)  # Show grid
            st.pyplot(plt.gcf())
        except Exception as e:
                 st.error(f"Error loading data: {e}")
        pass

    if "Bar Plot" in visualization_options:
        column = st.selectbox("Choose a column for the Bar Plot:", df.columns)
        try:
            selected_attribute = column
            # Group by the selected attribute and count the occurrences
            attribute_counts = df[selected_attribute].value_counts()
            # Plotting a bar plot
            plt.figure(figsize=(10, 6))  # Set the figure size
            attribute_counts.plot(kind='bar', color='skyblue', edgecolor='black')  # Plotting the bar plot
            plt.title(f'Bar Plot of {selected_attribute}')  # Set the title
            plt.xlabel(selected_attribute)  # Set the x-axis label
            plt.ylabel('Count')  # Set the y-axis label
            plt.grid(axis='y')  # Show grid on the y-axis
            st.pyplot(plt.gcf())
        except Exception as e:
                 st.error(f"Error loading data: {e}")
        pass

    if "Scatter Plot" in visualization_options:
        try:
            plot=sns.jointplot(df)
            plt.xlabel('Values')  # Set the x-axis label
            plt.ylabel('Values')  # Set the y-axis label
            plt.grid(axis='y')
            st.pyplot(plot.fig)
        except Exception as e:
                 st.error(f"Error loading data: {e}")
        pass

    if "Scatter Plot between Two Attributes" in visualization_options:
        column_1 = st.selectbox("Choose column_1:", df.columns)
        column_2 = st.selectbox("Choose column_2:", df.columns)
        try:
            x_values = df[column_1]
            y_values = df[column_2]
            # Plotting a scatter plot
            plt.scatter(x_values, y_values, color='blue', marker='o', alpha=0.7)  # Plotting the scatter plot
            plt.title('Scatter Plot of '+column_1 +' vs '+column_2 )  # Set the title
            plt.xlabel('X')  # Set the x-axis label
            plt.ylabel('Y')  # Set the y-axis label
            plt.grid(True)  # Show grid
            st.pyplot(plt.gcf())
        except Exception as e:
                 st.error(f"Error loading data: {e}")
        pass

    if "Pair Plot" in visualization_options:
        try:
            plot=sns.pairplot(df)
            plt.xlabel('Attributes')  # Set the x-axis label
            plt.ylabel('Attributes')  # Set the y-axis label
            plt.grid(axis='y')
            st.pyplot(plot.fig)
        except Exception as e:
                 st.error(f"Error loading data: {e}")
        pass

    if "Multivariate Analysis" in visualization_options:
        column = st.selectbox("Choose column:", df.columns)
        column_3 = st.selectbox("Choose Categorical column:", df.columns)
        try:
            # Create a pair plot for the selected attribute and other numeric columns
            plot=sns.catplot(x=column, hue=column_3, kind="bar", data = df)
            plt.suptitle(f'Multivariate Analysis for {column}', y=1.02)
            st.pyplot(plot.fig)
        except Exception as e:
                 st.error(f"Error loading data: {e}")
        pass

# ... implement other visualization options



#-------------------------------------------
#---------------Classification--------------
#-------------------------------------------
classification_options = st.multiselect("Select classification operations:", ["Decision tree", "KNN Classifier"])

if "Decision tree" in classification_options:
     column = st.selectbox("Choose target column:", df.columns)
     X = df.drop(column, axis=1)
     y = df[column]
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
     unprunned_tree = DecisionTreeClassifier(criterion='gini')
     unprunned_tree.fit(X_train, y_train)
     y_pred = unprunned_tree.predict(X_test)
     accuracy = accuracy_score(y_test, y_pred)
     conf_matrix = confusion_matrix(y_test, y_pred)
     class_report = classification_report(y_test, y_pred)
     st.write("Accuracy: ")
     st.write(accuracy)
     st.write("Confusion Matrix: ")
     st.write(conf_matrix)
     st.write("Calssification Report: ")
     st.write(class_report)
     plt.figure(figsize=(12, 8))
     plot_tree(unprunned_tree, filled=True, feature_names=X.columns, class_names=list(map(str, unprunned_tree.classes_)))
     st.pyplot(plt.gcf())
     pass

if "KNN Classifier" in classification_options:
     column = st.selectbox("Choose target column:", df.columns)
     kneighbprs=int(st.text_input("Enter k values to determine neihbors: "))
     try:
          X = df.drop(column, axis=1)
          y = df[column]
          X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
          KNN = KNeighborsClassifier(n_neighbors=kneighbprs)
          KNN.fit(X_train, y_train)
          y_pred = KNN.predict(X_test)
          accuracy = accuracy_score(y_test, y_pred)
          conf_matrix = confusion_matrix(y_test, y_pred)
          class_report = classification_report(y_test, y_pred)
          st.write("Accuracy: ")
          st.write(accuracy)
          st.write("Confusion Matrix: ")
          st.write(conf_matrix)
          st.write("Calssification Report: ")
          st.write(class_report)
          error = []
          # Calculating error for K values between 1 and 40
          for i in range(1, kneighbprs):
               knn = KNeighborsClassifier(n_neighbors=i)
               knn.fit(X_train, y_train)
               pred_i = knn.predict(X_test)
               error.append(np.mean(pred_i != y_test))

          plt.plot(range(1, kneighbprs), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
          plt.title('Error Rate K Value')
          plt.xlabel('K Value')
          plt.ylabel('Mean Error')
          st.pyplot(plt.gcf())
     except Exception as e:
                 st.error(f"Error loading data: {e}")
     pass
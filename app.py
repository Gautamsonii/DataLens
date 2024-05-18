import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_option('deprecation.showPyplotGlobalUse', False)


# Function to load dataset
def load_dataset():
    st.header("Data Upload")
    uploaded_file = st.file_uploader("Upload a dataset (CSV)", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Dataset loaded successfully.")
            return df
        except Exception as e:
            st.error(f"Error: {e}")
    return None


# Function for data visualization
def perform_data_visualization(df):
    st.subheader("Data Visualization")

    # Visualization type selection
    visualization_type = st.selectbox("Select Visualization Type",
                                      ["Scatter Plot", "Pair Plot", "Pie Plot", "Violin Plot"])

    if visualization_type == "Scatter Plot":
        st.write("Scatter Plot:")
        scatter_columns = st.multiselect("Select columns for scatter plot", df.columns)
        if len(scatter_columns) >= 2:
            fig, ax = plt.subplots()
            ax.scatter(df[scatter_columns[0]], df[scatter_columns[1]])
            st.pyplot(fig)
        else:
            st.warning("Select at least two columns for the scatter plot.")

    elif visualization_type == "Pair Plot":
        st.write("Pair Plot:")
        pair_columns = st.multiselect("Select columns for pair plot", df.columns)
        if len(pair_columns) >= 2:
            pair_plot = sns.pairplot(df[pair_columns])
            st.pyplot(pair_plot.fig)
        else:
            st.warning("Select at least two columns for the pair plot.")

    elif visualization_type == "Pie Plot":
        st.write("Pie Plot:")
        categorical_column = st.selectbox("Select Categorical Column for Pie Plot",
                                          df.select_dtypes(include=['object']).columns)
        if categorical_column:
            pie_data = df[categorical_column].value_counts()
            fig = px.pie(names=pie_data.index, values=pie_data.values, title=f"Pie Plot for {categorical_column}")
            st.plotly_chart(fig)
        else:
            st.warning("Select a categorical column for the pie plot.")

    elif visualization_type == "Violin Plot":
        st.write("Violin Plot:")
        numerical_column = st.selectbox("Select Numerical Column for Violin Plot",
                                        df.select_dtypes(include=['float64', 'int64']).columns)
        if numerical_column:
            fig, ax = plt.subplots()
            sns.violinplot(x=numerical_column, data=df, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Select a numerical column for the violin plot.")


# Function for data preprocessing
def perform_preprocessing(df):
    st.subheader("Data Preprocessing")

    # Display missing values
    missing_values = df.isnull().sum()
    missing_values_df = pd.DataFrame({
        'Column': missing_values.index,
        'Null Count': missing_values.values
    })

    st.write(missing_values_df)

    if missing_values_df['Null Count'].sum() == 0:
        st.success("No null values found in the dataset.")
    else:
        # Fill missing values in numeric columns
        numeric_columns = df.select_dtypes(include='number').columns

        if fill_method := st.radio("Select a method to fill null values in numeric columns:",
                                   ['Mean', 'Median', 'Mode']):
            if not numeric_columns.empty:
                columns_to_fill = [column for column in numeric_columns if df[column].isnull().sum() > 0]
                for column in columns_to_fill:
                    if fill_method == 'Mean':
                        fill_value = df[column].mean()
                    elif fill_method == 'Median':
                        fill_value = df[column].median()
                    elif fill_method == 'Mode':
                        fill_value = df[column].mode().iloc[0]

                    df[column].fillna(fill_value, inplace=True)

                st.success(f"Null values in {', '.join(columns_to_fill)} filled using {fill_method}.")

                # Display the updated DataFrame with filled values
                st.write("DataFrame after filling null values:")
                st.write(df)

                # Add threshold option for dropping columns with null values
                drop_threshold = st.slider("Set Null Drop Threshold (%)", 0, 100, 20)
                columns_to_drop = missing_values_df[missing_values_df['Null Count'] > (drop_threshold / 100 * len(df))]['Column'].tolist()

                if columns_to_drop:
                    df.drop(columns=columns_to_drop, inplace=True)
                    st.success(f"Dropped columns with null count exceeding {drop_threshold}%: {', '.join(columns_to_drop)}")
                    st.write("Updated DataFrame after dropping columns:")
                    st.write(df)
                else:
                    st.warning(f"No columns found with null count exceeding {drop_threshold}%.")
            else:
                st.warning("No numeric columns found in the dataset.")
        else:
            st.warning("No fill method selected.")


# Function for data mining operations
def perform_data_mining(df):
    st.subheader("Data Mining Operations")
    # Show the correlation matrix
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df_numerical = df[numerical_columns]

    # Display correlation matrix
    st.subheader("Correlation Matrix:")
    correlation_matrix = df_numerical.corr()
    st.write(correlation_matrix)

    # Show the skewness of numerical features
    st.subheader("Skewness of Numerical Features:")
    skewness = df_numerical.skew()
    st.write(skewness)

    # Show the kurtosis of numerical features
    st.subheader("Kurtosis of Numerical Features:")
    kurtosis = df_numerical.kurtosis()
    st.write(kurtosis)

    # Display value counts for categorical columns
    st.subheader("Value Counts for Categorical Columns:")
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        st.write(f"Value counts for {col}:")
        st.table(df[col].value_counts())


# Function for additional visualizations
def perform_additional_visualizations(df):
    st.subheader("Additional Visualizations")
    # Implement additional visualizations here
    # Correlation Heatmap for Numerical Columns
    st.subheader("Correlation Heatmap:")
    numerical_columns = df.select_dtypes(include=['float64']).columns
    correlation_matrix = df[numerical_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    st.pyplot()

    st.subheader("Distribution Plots:")
    palette = 'Set2'
    for i, column in enumerate(df.columns):
        if df[column].dtype != 'object':
            st.write(f"{column} Distribution:")
            sns.histplot(df[column], kde=True, color=sns.color_palette(palette)[i % len(sns.color_palette(palette))])
            st.pyplot()

    st.subheader("Box Plots:")
    for i, column in enumerate(df.columns):
        if df[column].dtype != 'object':
            st.write(f"{column} Box Plot:")
            sns.boxplot(x=df[column], color=sns.color_palette('pastel')[i % len(sns.color_palette('pastel'))])
            st.pyplot()

    st.write("Count Plots:")
    for column in df.columns:
        if df[column].dtype == 'object':
            st.write(f"{column} Count Plot:")
            sns.countplot(x=df[column])
            st.pyplot()


# Function for running Apriori algorithm
def run_apriori(df, min_support=0.1, min_confidence=0.5):
    df = df.applymap(str)

    # One-hot encode categorical columns
    encoder = OneHotEncoder(sparse=False, drop='first')
    df_encoded = pd.DataFrame(encoder.fit_transform(df), columns=encoder.get_feature_names_out(df.columns))

    # Binarize the data
    df_binarized = df_encoded.applymap(lambda x: 1 if x > 0 else 0)

    # Apply Apriori algorithm
    frequent_itemsets = apriori(df_binarized, min_support=min_support, use_colnames=True)

    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    # Filter out only strong association rules
    strong_rules = rules[rules['confidence'] >= min_confidence]

    return frequent_itemsets, strong_rules


# Function for applying KMeans clustering and visualization
def perform_clustering(df, feature1, feature2, num_clusters):
    st.subheader("KMeans Clustering")

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters)
    clustered_data = df[[feature1, feature2]].dropna()
    kmeans.fit(clustered_data)

    # Add cluster labels to the DataFrame
    df['Cluster'] = kmeans.labels_

    st.write("Clustered Data:")
    st.table(df)

    # Visualization
    st.subheader("Clustering Visualization")
    fig, ax = plt.subplots()

    # Use 'clustered_data' for scatter plot
    scatter = ax.scatter(clustered_data[feature1], clustered_data[feature2], c=kmeans.labels_, cmap='viridis',
                         marker='o', edgecolors='k')
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X',
               label='Centroids')
    ax.set_xlabel(feature1)
    ax.set_ylabel(feature2)
    ax.legend()
    st.pyplot(fig)

# ... (previous code)

# Function for performing simple linear regression
def perform_linear_regression(df):
    st.subheader("Linear Regression")

    # Select target and feature variables
    target_variable = st.selectbox("Select Target Variable (Dependent)", df.columns)
    feature_variable = st.selectbox("Select Feature Variable (Independent)", df.columns)

    # Split the data into training and testing sets
    X = df[[feature_variable]]
    y = df[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and fit the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Display regression results
    st.write(f"Regression Coefficients: {model.coef_}")
    st.write(f"Intercept: {model.intercept_}")

    # Plot regression line
    fig, ax = plt.subplots()
    ax.scatter(X_test, y_test, label="Actual")
    ax.plot(X_test, y_pred, color='red', label="Predicted")
    ax.set_xlabel(feature_variable)
    ax.set_ylabel(target_variable)
    ax.legend()
    st.pyplot(fig)

    # Display regression metrics
    st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
    st.write(f"R-squared Score: {r2_score(y_test, y_pred)}")

# Main Streamlit app
def main():
    st.title("DWM Mini Project")

    # Load dataset
    df = load_dataset()

    if df is not None:
        # Display basic information about the dataset
        st.header("Dataset Information:")
        st.table(df.describe())

        st.header("Quick Look:")
        st.table(df.head())

        # Dropdown in the main page
        menu_choice = st.selectbox("Select Operation",
                                   ["Home", "Data Visualization", "Data Preprocessing", "Data Mining",
                                    "Interactive Data Exploration", "Clustering", "Apriori Algorithm", "Regression"])

        if menu_choice == "Home":
            pass
        elif menu_choice == "Data Visualization":
            perform_data_visualization(df)
        elif menu_choice == "Data Preprocessing":
            perform_preprocessing(df)
        elif menu_choice == "Apriori Algorithm":
            st.subheader("Apriori Algorithm")

            # Display current DataFrame
            st.header("Current DataFrame:")
            st.table(df)

            # Set minimum support and confidence
            min_support = st.slider("Minimum Support", 0.01, 1.0, 0.1, 0.01)
            min_confidence = st.slider("Minimum Confidence", 0.01, 1.0, 0.5, 0.01)

            if st.button("Run Apriori Algorithm"):
                frequent_itemsets, strong_rules = run_apriori(df, min_support, min_confidence)

                # Display results
                st.header("Frequent Itemsets:")
                st.table(frequent_itemsets)

                st.header("Strong Association Rules:")
                st.table(strong_rules)
        elif menu_choice == "Data Mining":
            perform_data_mining(df)
        elif menu_choice == "Clustering":
            st.subheader("Select Features for Clustering")
            feature1 = st.selectbox("Select first feature", df.columns)
            feature2 = st.selectbox("Select second feature", df.columns)
            num_clusters = st.number_input("Number of Clusters", min_value=2, max_value=10, value=3)

            if st.button("Apply Clustering"):
                perform_clustering(df, feature1, feature2, num_clusters)
        elif menu_choice == "Interactive Data Exploration":
            perform_additional_visualizations(df)
        elif menu_choice == "Regression":
            perform_linear_regression(df)

if __name__ == "__main__":
    main()

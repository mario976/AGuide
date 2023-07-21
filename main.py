import csv
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# # Adding new student to the data
# with open('Students Grades.csv', mode='a', newline='') as file:
#     writer = csv.writer(file)
#
#     while True:
#         name = input("Enter your ID (or 'quit' to exit): ")
#         if name == 'quit':
#             break
#         print("Prerequisites of CS")
#         grade1 = int(input("Enter grade of CS213: "))
#         grade2 = int(input("Enter grade of CS214: "))
#         print("Prerequisites of IS")
#         grade3 = int(input("Enter grade of IS211: "))
#         grade4 = int(input("Enter grade of CS251: "))
#         print("Prerequisites of AI")
#         grade5 = int(input("Enter grade of MA112: "))
#         grade6 = int(input("Enter grade of IT212: "))
#         print("Prerequisites of IT")
#         print("Enter grade of IT212: ", grade6)
#         grade7 = int(input("Enter grade of IT221: "))
#         print("Prerequisites of DS")
#         grade8 = int(input("Enter grade of DS211: "))
#         grade9 = int(input("Enter grade of ST222: "))
#         GPA = float(input("Enter your GPA: "))
#
#         writer.writerow([name, grade1, grade2, grade3, grade4, grade5, grade6, grade7, grade8, grade9, GPA])
#         print("Grades added successfully.")
#
# file.close()
#
#


# def add_random_students(num_students):
#     courses = ["CS213", "CS214", "IS211", "CS251", "MA112", "IT212", "IT221", "DS211", "ST222"]
#     with open('Students Grades.csv', mode='a', newline='') as file1, \
#             open('Students Grades2.csv', mode='a', newline='') as file2:
#         writer1 = csv.writer(file1)
#         writer2 = csv.writer(file2)
#         for i in range(num_students):
#             name = "Student" + str(i + 1)
#             grades = [random.randint(0, 100) for _ in range(len(courses))]
#             GPA = round(random.uniform(1.0, 4.0), 2)
#             writer1.writerow([name] + grades + [GPA])
#             writer2.writerow([name] + grades + [GPA])
#             print("Grades added for", name)
#     print("Done adding", num_students, "students.")
#
#
# add_random_students(20)

Load the data
df = pd.read_csv("Students Grades.csv")
# Extract the features and labels
X = df.drop(["Students IDs", "GPAs", "Dep"], axis=1)
y = df["Dep"]

# Scale the data
scale = StandardScaler()
X_scaled = scale.fit_transform(X)

# Perform PCA to reduce the dimensionality
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Perform K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=25)
clusters = kmeans.fit_predict(X_pca)

# Visualize the clusters
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, s=50, cmap='viridis')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()

# Save the cluster labels to a CSV file
df_clusters = pd.DataFrame({"Student ID": df["Students IDs"], "Cluster Label": clusters})

# Combine the original data with the cluster labels and save to a CSV file
df_combined = pd.concat([df, df_clusters["Cluster Label"]], axis=1)
df_combined.to_csv("Students Clusters.csv", index=False)

# Select data where "Dep" column is not null
df = df[df['Dep'].notnull()]
df = df.drop(["Students IDs", "GPAs"], axis=1)
print(df)
# Calculate correlation matrix
corr_matrix = df.corr(numeric_only=True)

# Extract correlations between "Dep" and Grades
dep_corr = corr_matrix.iloc[:-1, -1]

# Calculate sum of absolute correlations
sum_abs_corr = np.sum(np.abs(corr_matrix.values.flatten()))

print("Correlation between Dep and Grades:", sum_abs_corr)

# Plot heatmap of correlation matrix
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, vmin=-1, vmax=1)
plt.title("Correlation Matrix")
plt.show()

# Illustrate each cluster
for i in range(5):
    print(f"Cluster {i}:")
    cluster_data = df_combined[df_combined["Cluster Label"] == i]
    print(cluster_data["Dep"].value_counts())

# Courses recommendation system
data = pd.read_csv("Students Grades.csv")

# define the majors and their prerequisite courses
majors = {"CS": ["CS213", "CS214"], "IS": ["CS251", "IS211"], "AI": ["IT212", "MA112"],
          "IT": ["IT212", "IT221"], "DS": ["DS211", "ST222"]}

# calculate the sum of scores for each major
for major, courses in majors.items():
    data[major] = data[courses].sum(axis=1)

# recommend majors for each student
for i in range(1, 6):
    data[f"{i}st recommendation"] = data[majors.keys()].idxmax(axis=1)
    for major in majors:
        data = data[data[major] != data[major].max()]

# sort the recommendations in descending order
for i in range(1, 6):
    data[f"{i}st recommendation"] = data.apply(lambda x: sorted(majors.keys(), key=lambda y: x[y], reverse=True)[i - 1],
                                               axis=1)

data.to_csv("Recommended Departments.csv", index=False)

# Eligible courses due the GPA constrain
df = pd.read_csv('Recommended Departments.csv')

# Create a copy of the dataframe without the "Dep" column
df_without_dep = df.drop(columns=['Dep'])

# Define the GPA ranges and their corresponding major eligibility
gpa_ranges = {'range1': {'min': 1, 'max': 2.40, 'majors': ['IT', 'DS']},
              'range2': {'min': 1, 'max': 2.71, 'majors': ['AI', 'IT', 'DS']},
              'range3': {'min': 1, 'max': 2.79, 'majors': ['CS', 'AI', 'IT', 'DS']},
              'range4': {'min': 1, 'max': 4.00, 'majors': ['IS', 'CS', 'AI', 'IT', 'DS']}}

# Create a new dataframe to store the output
output_df = pd.DataFrame(columns=['Student ID', 'GPA', 'Eligible Major 1', 'Eligible Major 2', 'Eligible Major 3',
                                  'Eligible Major 4', 'Eligible Major 5'])

# Initialize an empty list to store output dataframes
output_dfs = []

# Iterate over each row in the dataframe without the "Dep" column
for index, row in df_without_dep.iterrows():

    # Get the student's GPA
    gpa = row['GPAs']

    # Check which GPA range the student falls into
    for key, value in gpa_ranges.items():
        if value['min'] <= gpa <= value['max']:

            # Check if the student's department is eligible for all recommended majors
            eligible_majors = []

            for major in row[1:]:
                if major in value['majors']:
                    eligible_majors.append(major)

            # Create a new output dataframe for the current student
            output_row = {'Students IDs': row['Students IDs'], 'GPA': row['GPAs']}
            i = 0
            for i, major in enumerate(eligible_majors):
                output_row[f'Eligible Major {i + 1}'] = major

            while i < 4:
                i += 1
                output_row[f'Eligible Major {i + 1}'] = ''

            output_df = pd.DataFrame(output_row, index=[0])
            output_dfs.append(output_df)
            break

# Concatenate all output dataframes into a single dataframe
output_df = pd.concat(output_dfs, ignore_index=True)

output_df.to_csv('Final result.csv', index=False)

# # Filter the data to keep only the rows where the 'dep' column has a value
# filtered_df = df[df['Dep'].notnull()]
#
# # Define the features (9 courses grades) and the target (department)
# features = ["CS213", "CS214", "IS211", "CS251", "MA112", "IT212", "IT221", "DS211", "ST222"]
# target = '1st recommendation'
#
# # Split the filtered data into training and test sets (67% training, 33% test)
# X_train, X_test, y_train, y_test = train_test_split(filtered_df[features], filtered_df[target],
#                                                     test_size=0.33, random_state=42)
#
# # Scale the data using StandardScaler
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# # Create and train the logistic regression model
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train_scaled, y_train)
#
# # Use the trained model to predict the department for the rows where the 'dep' column is empty
# unlabeled_df = df[df['Dep'].isnull()]  # Filter to keep only the rows where the 'dep' column is empty
# unlabeled_X = unlabeled_df[features]  # Extract the features for the unlabeled rows
# unlabeled_X_scaled = scaler.transform(unlabeled_X)  # Scale the features using the same scaler used for the training set
# unlabeled_y_pred = model.predict(unlabeled_X_scaled)  # Predict the department for the unlabeled rows
#
# unlabeled_ids = unlabeled_df['Students IDs'].values  # Extract the IDs for the unlabeled rows
# unlabeled_grades = unlabeled_df[features].values  # Extract the grades for the unlabeled rows
# results_df = pd.DataFrame({'Students IDs': unlabeled_ids, 'Dep': unlabeled_y_pred,
#                            **{f: unlabeled_grades[:, i] for i, f in enumerate(features)}})
# results_df.to_csv('predicted_departments.csv', index=False)
#
# # Print the accuracy of the model on the test set
# y_pred = model.predict(X_test_scaled)
#
# accuracy = sum(y_pred == y_test) / len(y_test)
# print("Accuracy:", accuracy)

df1 = pd.read_csv("System test lvl1,2.csv")
df1_no_ids = df1.drop("Students IDs", axis=1)
df2 = pd.read_csv("System test lvl3,4.csv")
df2_no_ids = df2.drop("Students IDs", axis=1)

# Calculate the average for each row in both files
avg1 = df1_no_ids.mean(axis=1)
avg2 = df2_no_ids.mean(axis=1)

# Combine the results into a single DataFrame
results = pd.DataFrame({"Students IDs": df1["Students IDs"], "Avg lvl1,2": avg1, "Avg lvl3,4": avg2})

print(results)

# Calculate the percentage of students with better performance in the second 2 years
num_students = len(results)
num_greater = len(results[results["Avg lvl3,4"] > results["Avg lvl1,2"]])
percentage_greater = (num_greater / num_students) * 100

# Print the percentage of students
print("Percentage of students whose their performance in  lvl 3 and 4 has been improved "
      "than their performance in  lvl 1 and 2: {:.2f}%".format(percentage_greater))

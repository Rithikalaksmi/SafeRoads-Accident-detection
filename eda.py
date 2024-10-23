import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create images directory if it doesn't exist
images_dir = 'C:/Users/rithi/OneDrive/Documents/sem 5/machine learning lab/project/Front end final/Front end/static/images'  # Path to save images
os.makedirs(images_dir, exist_ok=True)

# Load the dataset
file_path = 'combined_accidents_data.csv'  # Update this path
df = pd.read_csv(file_path)

# Set a consistent figure size for all plots
FIGURE_SIZE = (10, 6)
FIGURE_SIZE1 = (10,3)
# Step 1: Analyze the distribution of Urban and Rural Areas
plt.figure(figsize=FIGURE_SIZE)
df['Urban_or_Rural_Area'].value_counts().plot(kind='pie', autopct='', labels=None)  # No labels directly on the pie
plt.legend(loc="upper right", labels=["Urban", "Rural"])  # Set the legend labels here
plt.ylabel('')  # Remove the y-axis label
plt.title("Accidents in Urban Or Rural Areas")
plt.tight_layout()
plt.savefig(os.path.join(images_dir, "urban_rural_distribution.png"))
plt.close()

# Step 2: Analyze the distribution of Accident Severity
plt.figure(figsize=FIGURE_SIZE1)
sns.countplot(x='Accident_Severity', data=df)
plt.title('Distribution of Accident Severity')
plt.xlabel('Accident Severity')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(images_dir, "accident_severity_distribution.png"))
plt.close()

# Step 3: Analyze Weather Conditions
plt.figure(figsize=FIGURE_SIZE)
sns.countplot(y='Weather_Conditions', data=df, order=df['Weather_Conditions'].value_counts().index)
plt.title('Distribution of Weather Conditions')
plt.xlabel('Count')
plt.ylabel('Weather Conditions')
plt.tight_layout()
plt.savefig(os.path.join(images_dir, "weather_conditions_distribution.png"))
plt.close()

# Step 4: Analyze Day of the Week
plt.figure(figsize=FIGURE_SIZE)
sns.countplot(x='Day_of_Week', data=df)
plt.title('Distribution of Accidents by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(images_dir, "accidents_by_day_of_week.png"))
plt.close()

# Step 5: Analyze the relationship between Weather Conditions and Accident Severity
plt.figure(figsize=FIGURE_SIZE)
sns.countplot(data=df, x='Weather_Conditions', hue='Accident_Severity')
plt.title('Weather Conditions vs. Accident Severity')
plt.xlabel('Weather Conditions')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Accident Severity')
plt.tight_layout()
plt.savefig(os.path.join(images_dir, "weather_vs_severity.png"))
plt.close()

# Step 6: Analyze Road Surface Conditions
plt.figure(figsize=FIGURE_SIZE)
sns.countplot(data=df, x="Road_Surface_Conditions", hue="Year")
plt.title('Accidents by Road Surface Conditions Over Years')
plt.tight_layout()
plt.savefig(os.path.join(images_dir, "road_surface_conditions.png"))
plt.close()

# Step 7: Total Number of Casualties by Speed Limit
grouped_data = df.groupby('Speed_limit')['Number_of_Casualties'].sum().reset_index()
plt.figure(figsize=FIGURE_SIZE)
sns.barplot(data=grouped_data, x='Speed_limit', y='Number_of_Casualties')
plt.xlabel('Speed Limit')
plt.ylabel('Total Number of Casualties')
plt.title('Total Number of Casualties by Speed Limit')
plt.tight_layout()
plt.savefig(os.path.join(images_dir, "casualties_by_speed_limit.png"))
plt.close()

# Step 8: Total Number of Casualties by Accident Severity
grouped_data = df.groupby('Accident_Severity')['Number_of_Casualties'].sum().reset_index()
plt.figure(figsize=FIGURE_SIZE)
sns.barplot(data=grouped_data, x='Accident_Severity', y='Number_of_Casualties')
plt.xlabel('Accident Severity')
plt.ylabel('Total Number of Casualties')
plt.title('Total Number of Casualties by Accident Severity')
plt.tight_layout()
plt.savefig(os.path.join(images_dir, "casualties_by_accident_severity.png"))
plt.close()

# Step 9: Monthly Casualties Over Years
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # Convert to datetime if not already done
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
monthly_casualties = df.groupby(['Year', 'Month'])['Number_of_Casualties'].sum().reset_index()

plt.figure(figsize=FIGURE_SIZE)
sns.barplot(x='Year', y='Number_of_Casualties', hue='Month', data=monthly_casualties, width=0.8)
plt.title('Monthly Number of Casualties (2005-2014)')
plt.xlabel('Year')
plt.ylabel('Number of Casualties')
plt.legend(title='Month', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(images_dir, "monthly_casualties.png"))
plt.close()

# Step 10: Number of Accidents by Hour
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M').dt.hour  # Ensure the Time is parsed as hour values
plt.figure(figsize=FIGURE_SIZE)
sns.countplot(x="Time", data=df)
plt.title("Number Of Accidents by Hours")
plt.xlabel("Hours")
plt.ylabel("Number Of Accidents")
plt.tight_layout()
plt.savefig(os.path.join(images_dir, "accidents_by_hour.png"))
plt.close()

print("EDA completed successfully and images are saved.")

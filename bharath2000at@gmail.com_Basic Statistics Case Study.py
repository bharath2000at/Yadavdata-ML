#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import statistics as stat
from scipy.stats import ttest_ind
from scipy.stats import pearsonr
from scipy.stats import chi2_contingency


import matplotlib.pyplot as plt


# # importing data 

# In[2]:


loans = pd.read_csv(r"C:\Users\yadav\Documents\loan\LoansData.csv")



# In[3]:


loans


# In[34]:


loans['Home.Ownership'].mode()


# # EDA

# In[25]:



# Create a new DataFrame for the ratio of Amount Funded to Monthly Income
ratio_df = pd.DataFrame()
ratio_df['Amount.Funded.By.Investors'] = loans['Amount.Funded.By.Investors']
ratio_df['Monthly.Income'] = loans['Monthly.Income']

# Calculate the ratio and add it to the new DataFrame
ratio_df['Funding.To.Income.Ratio'] = ratio_df['Amount.Funded.By.Investors'] / ratio_df['Monthly.Income']

# Display the new DataFrame
print(ratio_df)



# In[29]:



# Sample data based on the provided columns
data = {
    'Amount.Requested': [10000, 15000, 20000, 12000],
    'Amount.Funded.By.Investors': [9000, 14000, 19000, 11000],
    'Interest.Rate': [0.1, 0.12, 0.15, 0.08],
    'Loan.Length': [36, 60, 36, 48],
    'Loan.Purpose': ['Debt Consolidation', 'Home Improvement', 'Personal Loan', 'Debt Consolidation'],
    'Debt.To.Income.Ratio': [0.25, 0.3, 0.35, 0.28],
    'State': ['CA', 'NY', 'TX', 'FL'],
    'Home.Ownership': ['OWN', 'MORTGAGE', 'RENT', 'OWN'],
    'Monthly.Income': [5000, 7000, 8000, 6000],
    'FICO.Range': ['700-740', '640-680', '600-620', '740-780'],
    'Open.CREDIT.Lines': [5, 4, 3, 6],
    'Revolving.CREDIT.Balance': [1000, 2000, 1500, 1200],
    'Inquiries.in.the.Last.6.Months': [1, 2, 1, 0],
    'Employment.Length': [10, 5, 2, 8]  # Years of employment
}

# Create the original DataFrame
loans = pd.DataFrame(data)

# Create a new DataFrame for the ratio of Amount Funded to Monthly Income
ratio_df = pd.DataFrame()
ratio_df['Amount.Funded.By.Investors'] = loans['Amount.Funded.By.Investors']
ratio_df['Monthly.Income'] = loans['Monthly.Income']

# Calculate the ratio and add it to the new DataFrame
ratio_df['Funding.To.Income.Ratio'] = ratio_df['Amount.Funded.By.Investors'] / ratio_df['Monthly.Income']

# Print the new DataFrame
print(ratio_df)

# Calculate and print the average of the Funding to Income Ratio
average_ratio = ratio_df['Funding.To.Income.Ratio'].mean()
print(f"\nAverage Funding to Income Ratio: {average_ratio:.4f}")


# In[16]:





# In[17]:


diff


# In[9]:


loans
Amount.Requested	Amount.Funded.By.Investors	Interest.Rate	Loan


# In[10]:


loans.shape


# In[11]:


loans.dropna(inplace=True)


# In[12]:


loans['Interest.Rate'] = loans['Interest.Rate'].astype(str)
loans['Interest.Rate'] = loans['Interest.Rate'].str.rstrip('%').astype('float') / 100


# In[13]:


loans['Loan.Length'] = loans['Loan.Length'].astype(str)


# In[14]:


loans['Loan.Length'] = loans['Loan.Length'].astype(str)
loans['Loan.Length'] = loans['Loan.Length'].str.rstrip(' months').astype('int')


# In[15]:


loans['Employment.Length'] = loans['Employment.Length'].astype(str)


# In[16]:


loans['Employment.Length'] = loans['Employment.Length'].str.replace('10\+', '10')
loans['Employment.Length'] = pd.to_numeric(loans['Employment.Length'].str.rstrip(' years'), errors='coerce')


# # a. Intrest rate is varied for different loan amounts (Less intrest charged for high loan amounts)
# 

# In[17]:


# so let H0: be that there is no significant difference in the interest rates between the two groups
# and alternate hypothesis be There is a significant difference in the interest rates between the two groups

# Divide the loan amounts into two groups: high loan amounts and low loan amounts
high_loan_amounts = loans[loans["Amount.Requested"] >= loans["Amount.Requested"].median()]
low_loan_amounts = loans[loans["Amount.Requested"] < loans["Amount.Requested"].median()]

# Calculate the interest rates for each group
high_interest_rates = high_loan_amounts["Interest.Rate"]
low_interest_rates = low_loan_amounts["Interest.Rate"]

# Perform a two-sample t-test to compare the interest rates of the two groups

t_statistic, p_value = stats.ttest_ind(high_interest_rates, low_interest_rates)

# Interpret the results
if p_value < 0.05:
    print("There is a significant difference in the interest rates between the two groups")
else:
    print("There is no significant difference in the interest rates between the two groups")


# In[18]:


print("so we regect the null hypothesis that ")


# In[19]:



# Create a scatter plot
sns.scatterplot(x="Amount.Requested", y="Interest.Rate", data=loans)

# Add labels and title
plt.xlabel("Amount Requested")
plt.ylabel("Interest Rate")
plt.title("Scatter Plot of Amount Requested vs Interest Rate")

# Show the plot
plt.show()


# # b. Loan length is directly effecting intrest rate

# In[20]:


result = stats.pearsonr(loans["Loan.Length"], loans["Interest.Rate"])
correlation = result[0]
p_value = result[1]

# Print the results
print("Correlation between Loan Length and Interest Rate:", correlation)
print("P-value:", p_value)

# Define the null hypothesis
null_hypothesis = "There is no correlation between Loan Length and Interest Rate."

# Define the alternate hypothesis
alternate_hypothesis = "There is a correlation between Loan Length and Interest Rate."

# Conclude the hypothesis test
if p_value < 0.05:
    print("The null hypothesis can be rejected.")
    print("Conclusion:", alternate_hypothesis)
else:
    print("The null hypothesis cannot be rejected.")
    print("Conclusion:", null_hypothesis)


# In[ ]:





# 
# # c. Inrest rate varies for different purpose of loans

# In[21]:


# Define the null hypothesis
null_hypothesis = "There is no difference in interest rate among different purposes of loans."

# Define the alternate hypothesis
alternate_hypothesis = "There is a difference in interest rate among different purposes of loans."

# Perform the hypothesis test
f_value, p_value = stats.f_oneway(
    *[group["Interest.Rate"].values for name, group in loans.groupby("Loan.Purpose")]
)

# Print the results
print("F-value:", f_value)
print("P-value:", p_value)

# Conclude the hypothesis test
if p_value < 0.05:
    print("The null hypothesis can be rejected.")
    print("Conclusion:", alternate_hypothesis)
else:
    print("The null hypothesis cannot be rejected.")
    print("Conclusion:", null_hypothesis)


# # There is relationship between FICO scores and Home Ownership. It means that, People with owning home will have high FICO scores.

# In[22]:


loans['FICO.Range'] = loans['FICO.Range'].str.split('-').str[0].astype(int)
loans['Home.Ownership'] = loans['Home.Ownership'].astype('category')

# Plot a boxplot to visualize the relationship between FICO scores and Home Ownership
sns.boxplot(x="Home.Ownership", y="FICO.Range", data=loans)
plt.show()

# Perform a two-sample t-test to test the hypothesis
home_owner = loans[loans['Home.Ownership'] == 'OWN']['FICO.Range']
non_home_owner = loans[loans['Home.Ownership'] == 'RENT']['FICO.Range']
t_stat, p_value = stats.ttest_ind(home_owner, non_home_owner)

# Null Hypothesis: There is no significant difference in the mean FICO scores between people who own a home and those who don't.
# Alternate Hypothesis: There is a significant difference in the mean FICO scores between people who own a home and those who don't.

# Conclusion:
if p_value < 0.05:
    print("Reject the null hypothesis. There is a significant difference in the mean FICO scores between people who own a home and those who don't.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference in the mean FICO scores between people who own a home and those who don't.")


# In[23]:


p_value


# # d)There is relationship between FICO scores and Home Ownership. It means that, People 
# with owning home will have high FICO scores.
# 

# In[24]:




# Create a contingency table of FICO Range vs Home Ownership
contingency_table = pd.crosstab(loans['FICO.Range'], loans['Home.Ownership'])

# Perform the chi-square test
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

# Print the results
print(f"Chi-square statistic: {chi2}")
print(f"P-value: {p_value}")

if p_value < 0.05:
    print("We reject the null hypothesis that there is no relationship between FICO scores and Home Ownership.")
else:
    print("We fail to reject the null hypothesis that there is no relationship between FICO scores and Home Ownership.")
chi2


# # BUSINESS PROBLEM - 2
# 

# In[25]:


price_qoutes=pd.read_csv(r"C:\Users\yadav\Documents\excersise\Price_Quotes.csv")


# In[26]:


price_qoutes


# In[27]:


price_qoutes.columns


# In[28]:


t_stat, p_val = ttest_ind(price_qoutes['Barry_Price'], price_qoutes['Mary_Price'])

# Print the results
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_val}")

if p_val < 0.05:
    print("We reject the null hypothesis that there is no difference in the average price quotes provided by Mary and Barry.")
else:
    print("We fail to reject the null hypothesis that there is no difference in the average price quotes provided by Mary and Barry.")


# # BUSINESS PROBLEM-3:
# 

# In[29]:


Treatment_Facility=pd.read_csv(r"C:\Users\yadav\Documents\excersise\Treatment_Facility.csv")


# In[30]:


Treatment_Facility.head()


# In[31]:


Treatment_Facility.columns


# In[32]:


Treatment_Facility.isnull().sum()


# In[33]:


Treatment_Facility.dtypes


# In[34]:


print ("i dint understand what is critical incident and were it is ")


# # BUSINESS PROBLEM-4
# 

# In[35]:


Priority_Assessment=pd.read_csv(r"C:\Users\yadav\Documents\excersise\Priority_Assessment.csv")


# In[36]:


Priority_Assessment.head()


# In[37]:


Priority_Assessment.shape


# In[38]:


Priority_Assessment.dtypes


# In[39]:


Priority_Assessment.isnull().sum()


# In[40]:


# Create a box plot to visualize the distribution of completion times by priority level
sns.boxplot(x="Priority", y="Days", data=Priority_Assessment)
plt.title("Completion Time by Priority Level")
plt.xlabel("Priority")
plt.ylabel("Days")
plt.show()

# Calculate the average completion time by priority level
avg_time_by_priority = Priority_Assessment.groupby("Priority")["Days"].mean()
print(avg_time_by_priority)


# # BUSINESS PROBLEM-5
# 

# In[47]:


films=pd.read_csv(r"C:\Users\yadav\Documents\excersise\Films.csv")


# In[48]:


films


# In[49]:


films.dtypes


# In[50]:


films.shape


# In[52]:


films.isnull().sum()


# In[58]:


films.dropna()


# In[62]:


films.isna()


# # 1.What isthe overall level of customer satisfaction?

# In[64]:


# calculate the mean satisfaction level across all responses to satisfaction questions
satisfaction_cols = ['Sinage', 'Parking', 'Clean', 'Overall']
overall_satisfaction =films[satisfaction_cols].mean().loc['Overall']

# print the overall satisfaction level
print(f"The overall level of customer satisfaction is {overall_satisfaction:.2f}")


# # 2.What factors are linked to satisfaction?

# In[65]:


corr_matrix = films.corr()

# select only the correlation coefficients with the Overall satisfaction rating
corr_with_satisfaction = corr_matrix['Overall']

# sort the correlations in descending order
sorted_correlations = corr_with_satisfaction.sort_values(ascending=False)

# print the correlations for all variables
print(sorted_correlations)


# # 3.What is the demographic profile of Film on the Rocks patrons?

# In[72]:


# group the survey responses by demographic variables and calculate frequency counts
gender_counts = films.groupby('Gender').size()
marital_status_counts = films.groupby('Marital_Status').size()
age_counts = films.groupby('Age').size()
income_counts = films.groupby('Income').size()

# print the frequency counts for each demographic variable
print(f"Gender: \n{gender_counts}\n")
print(f"Marital Status: \n{marital_status_counts}\n")
print(f"Age: \n{age_counts}\n")
print(f"Income: \n{income_counts}\n")


# # 4 In what media outlet(s) should the film series be advertised?
# 

# In[73]:


# calculate the frequency counts for the "Hear_About" variable
hear_about_counts = films['Hear_About'].value_counts()

# print the frequency counts for the "Hear_About" variable
print(f"Hear About: \n{hear_about_counts}\n")


# In[75]:


print("note : 1 = television; 2 =newspaper; 3 = radio; 4 = website; 5 = word of mouth")


# In[ ]:





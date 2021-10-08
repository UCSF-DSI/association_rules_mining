# Import Required Libraries

We'll use [pandas](https://pandas.pydata.org/docs/index.html) for data processing and [mlxtend](http://rasbt.github.io/mlxtend/) to implement association rules mining.


```python
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
```

# Data Processing

We'll first load the data from the NHANES website ([data documentation](https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/RXQ_RX_J.htm))to a pandas dataframe. We'll only look at prescriptions taken within the last month, and we'll only need the patient identifier and drug name columns. 


```python
# Load data from NHANES into prescriptions_df
prescriptions_df = pd.read_sas('https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/RXQ_RX_J.XPT', encoding='UTF-8')

# Filter to prescriptions that were taken in the last month
prescriptions_df = prescriptions_df[prescriptions_df['RXDUSE'] == 1]

# Select patients (SEQN) and drug name (RXDDRUG) 
prescriptions_df = prescriptions_df[['SEQN', 'RXDDRUG']]

# View first few rows in dataframe
prescriptions_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SEQN</th>
      <th>RXDDRUG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>93705.0</td>
      <td>ENALAPRIL; HYDROCHLOROTHIAZIDE</td>
    </tr>
    <tr>
      <th>3</th>
      <td>93705.0</td>
      <td>MELOXICAM</td>
    </tr>
    <tr>
      <th>4</th>
      <td>93705.0</td>
      <td>OMEPRAZOLE</td>
    </tr>
    <tr>
      <th>7</th>
      <td>93708.0</td>
      <td>AMLODIPINE</td>
    </tr>
    <tr>
      <th>8</th>
      <td>93708.0</td>
      <td>LOSARTAN</td>
    </tr>
  </tbody>
</table>
</div>



For mlxtend to work, we'll need to label encode the drug names. This means creating a separate binary indicator column for each unique drug name. Then we'll need to group the data by the patient identifier, so that there is only one row for each unique patient identifier. As a final step, we removed all patients with no drugs.


```python
# Convert data to label encoded format
prescriptions_df = pd.concat(
    [prescriptions_df['SEQN'], pd.get_dummies(prescriptions_df['RXDDRUG'])],
    axis=1
)

# Group data by patients
prescriptions_df = prescriptions_df \
    .drop(columns=['55555', '77777', '99999']) \
    .groupby('SEQN') \
    .any()

# Remove patients with no prescription drugs
prescriptions_df = prescriptions_df[prescriptions_df.sum(axis=1) > 0]

# View first few rows in dataframe
prescriptions_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ABACAVIR; DOLUTEGRAVIR; LAMIVUDINE</th>
      <th>ABACAVIR; LAMIVUDINE</th>
      <th>ABATACEPT</th>
      <th>ACARBOSE</th>
      <th>ACEBUTOLOL</th>
      <th>ACETAMINOPHEN; BUTALBITAL</th>
      <th>ACETAMINOPHEN; BUTALBITAL; CAFFEINE</th>
      <th>ACETAMINOPHEN; CODEINE</th>
      <th>ACETAMINOPHEN; HYDROCODONE</th>
      <th>ACETAMINOPHEN; OXYCODONE</th>
      <th>...</th>
      <th>VERAPAMIL</th>
      <th>VILAZODONE</th>
      <th>VORTIOXETINE</th>
      <th>WARFARIN</th>
      <th>ZAFIRLUKAST</th>
      <th>ZALEPLON</th>
      <th>ZIDOVUDINE</th>
      <th>ZIPRASIDONE</th>
      <th>ZOLPIDEM</th>
      <th>ZONISAMIDE</th>
    </tr>
    <tr>
      <th>SEQN</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>93705.0</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>93708.0</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>93709.0</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>93713.0</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>93714.0</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 666 columns</p>
</div>



# Mine Frequent Itemsets

Now we can implement association rules mining. First we'll mine the frequent itemsets using the [apriori algorithm](https://en.wikipedia.org/wiki/Apriori_algorithm). We'll generate a list of itemsets with a support of at least 0.005.


```python
frequent_itemsets = apriori(prescriptions_df, min_support=0.005, use_colnames=True)
frequent_itemsets.sort_values('support', ascending=False).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>support</th>
      <th>itemsets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17</th>
      <td>0.150089</td>
      <td>(ATORVASTATIN)</td>
    </tr>
    <tr>
      <th>89</th>
      <td>0.144267</td>
      <td>(METFORMIN)</td>
    </tr>
    <tr>
      <th>82</th>
      <td>0.126044</td>
      <td>(LISINOPRIL)</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.114148</td>
      <td>(AMLODIPINE)</td>
    </tr>
    <tr>
      <th>93</th>
      <td>0.109593</td>
      <td>(METOPROLOL)</td>
    </tr>
  </tbody>
</table>
</div>



Next let's look at association rules. From looking at the rules with the highest lift, we see that the insulin drugs are highly associated. What are other patterns that may be interesting?


```python
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)
rules.sort_values('lift', ascending=False, inplace=True)
rules.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>antecedents</th>
      <th>consequents</th>
      <th>antecedent support</th>
      <th>consequent support</th>
      <th>support</th>
      <th>confidence</th>
      <th>lift</th>
      <th>leverage</th>
      <th>conviction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>250</th>
      <td>(INSULIN ASPART)</td>
      <td>(INSULIN GLARGINE)</td>
      <td>0.012149</td>
      <td>0.029866</td>
      <td>0.007593</td>
      <td>0.625000</td>
      <td>20.926907</td>
      <td>0.007230</td>
      <td>2.587024</td>
    </tr>
    <tr>
      <th>251</th>
      <td>(INSULIN GLARGINE)</td>
      <td>(INSULIN ASPART)</td>
      <td>0.029866</td>
      <td>0.012149</td>
      <td>0.007593</td>
      <td>0.254237</td>
      <td>20.926907</td>
      <td>0.007230</td>
      <td>1.324619</td>
    </tr>
    <tr>
      <th>255</th>
      <td>(INSULIN GLARGINE)</td>
      <td>(INSULIN LISPRO)</td>
      <td>0.029866</td>
      <td>0.011136</td>
      <td>0.006328</td>
      <td>0.211864</td>
      <td>19.024461</td>
      <td>0.005995</td>
      <td>1.254687</td>
    </tr>
    <tr>
      <th>254</th>
      <td>(INSULIN LISPRO)</td>
      <td>(INSULIN GLARGINE)</td>
      <td>0.011136</td>
      <td>0.029866</td>
      <td>0.006328</td>
      <td>0.568182</td>
      <td>19.024461</td>
      <td>0.005995</td>
      <td>2.246626</td>
    </tr>
    <tr>
      <th>522</th>
      <td>(METOPROLOL, FUROSEMIDE)</td>
      <td>(POTASSIUM CHLORIDE)</td>
      <td>0.017717</td>
      <td>0.025816</td>
      <td>0.005821</td>
      <td>0.328571</td>
      <td>12.727311</td>
      <td>0.005364</td>
      <td>1.450912</td>
    </tr>
    <tr>
      <th>527</th>
      <td>(POTASSIUM CHLORIDE)</td>
      <td>(METOPROLOL, FUROSEMIDE)</td>
      <td>0.025816</td>
      <td>0.017717</td>
      <td>0.005821</td>
      <td>0.225490</td>
      <td>12.727311</td>
      <td>0.005364</td>
      <td>1.268264</td>
    </tr>
    <tr>
      <th>526</th>
      <td>(FUROSEMIDE)</td>
      <td>(METOPROLOL, POTASSIUM CHLORIDE)</td>
      <td>0.053657</td>
      <td>0.008605</td>
      <td>0.005821</td>
      <td>0.108491</td>
      <td>12.607242</td>
      <td>0.005360</td>
      <td>1.112040</td>
    </tr>
    <tr>
      <th>523</th>
      <td>(METOPROLOL, POTASSIUM CHLORIDE)</td>
      <td>(FUROSEMIDE)</td>
      <td>0.008605</td>
      <td>0.053657</td>
      <td>0.005821</td>
      <td>0.676471</td>
      <td>12.607242</td>
      <td>0.005360</td>
      <td>2.925059</td>
    </tr>
    <tr>
      <th>187</th>
      <td>(FUROSEMIDE)</td>
      <td>(POTASSIUM CHLORIDE)</td>
      <td>0.053657</td>
      <td>0.025816</td>
      <td>0.014680</td>
      <td>0.273585</td>
      <td>10.597392</td>
      <td>0.013295</td>
      <td>1.341084</td>
    </tr>
    <tr>
      <th>186</th>
      <td>(POTASSIUM CHLORIDE)</td>
      <td>(FUROSEMIDE)</td>
      <td>0.025816</td>
      <td>0.053657</td>
      <td>0.014680</td>
      <td>0.568627</td>
      <td>10.597392</td>
      <td>0.013295</td>
      <td>2.193794</td>
    </tr>
  </tbody>
</table>
</div>



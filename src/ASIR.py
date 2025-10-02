import pandas as pd

bc = pd.read_csv('./data/raw/thyroid_cancer.csv')
pop = pd.read_csv('./data/raw/pop.csv')

w = {'5-14 years':17.29 , '15-49 years':52.01,  '50-69 years':16.6, '70+ years':5.275}

total = sum(w.values())
w = {k: v/total for k, v in w.items()}

years = (bc['year']>=1990) & (bc['year']>=2000)


y = 1990
sex = 'Both'
age = '15-49 years'
m = 'Number'
location = 'China'

bc1 = bc[(bc['year'] == y) & (bc['sex_name'] == sex) & (bc['age_name'] == age) & (bc['metric_name'] == m) & (bc['location_name'] == location)]
pop1 = pop[(pop['year'] == y) & (pop['sex'] == sex) & (pop['age'] == age) & (pop['location'] == location)]
ASIR = sum(bc1['val']) / pop1['val'].item() * w[age]
ASIR = bc1['val'] / pop1['val'].item() * w[age]
ASIR = round(ASIR* 100000, 2)
if __name__ == '__main__':
    print("number", bc1['val'])
    print("pop", pop1['val'].item())
    print("weight", w[age])
    print(ASIR)
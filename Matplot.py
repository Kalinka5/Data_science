import matplotlib.pyplot as plt
import pandas as pd

data = {
    'cases': [130, 140, 148, 160, 165, 172, 177],
    'deaths': [10, 20, 30, 30, 35, 60, 55],
    'months': ['January', 'February', 'March', 'April', 'May', 'June', 'July'],

}
df = pd.DataFrame(data, index=data['months'])

# Bar Chart
df['deaths'].plot(kind='bar')
plt.savefig('Dio(Histo)grams/Bar_Chart.png')
# Bar Chart With 2 Values
df = df[['cases', 'deaths']]
df.plot(kind='bar', stacked=True)  # kind = 'barh' - make horizontal bar chart
plt.savefig('Dio(Histo)grams/Double_Bar_Chart.png')

# Range Diagram
df['deaths'].describe().plot(kind="box")
plt.savefig('Dio(Histo)grams/Range_Diagram.png')

# Histogram
df['deaths'].plot(kind="hist")
plt.savefig('Dio(Histo)grams/Histogram.png')

# Scatter-plot
df.plot(kind="scatter", x="cases", y="deaths")
plt.savefig('Dio(Histo)grams/Scatter-plot.png')

# Pie Chart
df["deaths"].plot(kind="pie")
plt.savefig('Dio(Histo)grams/Pie_Chart.png')

# Line Chart
df.plot()
plt.savefig('Line_Charts/Line_Chart.png')
# Setting The Charts
df[['cases', 'deaths']].plot(kind="line", legend=True, color=['#000000', '#FF0000'])
plt.xlabel('Year')  # X axis
plt.ylabel('Number')  # Y axis
plt.suptitle("Death Statistic")  # create title
plt.savefig('Line_Charts/Double_Line_Chart.png')

# Double Area Chart
df = df[['cases', 'deaths']]
df.plot(kind="area", stacked=False)
plt.savefig('Line_Charts/Double_Area_Chart.png')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

"""
This program was used to determine the diffusion constant of carbon diffusing
in 1018 steel via pack carburization. Data plotted is Knoop hardness for each student
in the second semester materials science laboratory class at UW-Madison. While the 
data is quite messy, an accurate approximation of the diffusion constant could be 
obtained. 

The diffusion constant measurement was based on the relationship between knoop 
hardness and the carbon concentration in steel, which can then be written in terms of 
the diffusion distance, time, and diffusion constant which is given by the following:

(H(x,t)-H_0)/(H_s-H_0)=erf((x/(2*sqrt(D*t))))

where the hardness at surface is H_s, 
the unaltered steel hardness is H_0, 
the hardness at a given diffusion depth in time is H(x,t)
the diffusion depth is x
the diffusion constant is D
the diffusion time is t



Returns:
     
    This gives a diffusion constant of 8.426*10^-10 (m^2/s)
    The theoretically predicted value is 3.028*10^-11 (m^2/s), not bad!
    
"""

########################################################################################################################
#Plot
########################################################################################################################
data_file='path.xlsx'

data=pd.read_excel(data_file)

#time interals for pack carburization (seconds)
tkeys=['500','750','1000','1250','1500','1750','2000','2250','2500','2750','3000']

data_length=len(tkeys)

xdat=[]
ydat=[]

for dset in range(data_length):

    #data was ordered in excel such that column titles were blank
    #gave default name 'Unnamed: X'

    ydat+=[data[f'Unnamed: {7+2*dset}'].dropna(axis=0)]

    xvar=data[f'Unnamed: {6+2*dset}'].dropna(axis=0)

    if xvar[0] < 1:
        xvar*=1000

    xdat+=[xvar]


# #position of spines:

# gxmax="graph x max"
# gymax="graph y max"
# gxmin="graph x min"
# gymin="graph y min"

# #xlabel, ylabel, title

# x_label='label'
# y_label='label'
# # t_label='label'
    
##############
#space ticks
tick_num=7


xmin=min([min(x) for x in xdat])
xmax=max([max(x) for x in xdat])

ymin=min([min(x) for x in ydat])
ymax=max([max(x) for x in ydat])



#get logarithmic scaled tick marks for log plot

xrange= np.logspace(np.log(xmin),np.log(xmax),tick_num, base=np.e)
xlabels=["{:.0f}".format(round(x,2)) for x in xrange]


yrange= np.logspace(np.log(ymin),np.log(ymax),tick_num, base=np.e)
ylabels=["{:.0f}".format(round(x,2)) for x in yrange]



plt.ylabel("Hardness (Knoop)", fontsize=14)
plt.xlabel("Depth From Surface"+r" $(\mu m)$", fontsize=14)


ax=plt.gca()

#colors chosen to make each dataset seem as noticable as possible 
custom_colors = ['#e6194B', '#3cb44b', '#ffe119', '#000000', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080']


#plot data
for dset in range(data_length):
    ax.plot(np.array(xdat[dset]), np.array(ydat[dset]), color=custom_colors[dset], label=f'{tkeys[dset]}'+' s', linewidth=3)

ax.legend(loc='upper right', ncol=2, title='Pack Carburization Duration')


#log scale each axis (can also use plt.loglog())

plt.yscale('symlog')
plt.xscale('symlog')



plt.xticks(ticks=xrange, labels=xlabels, fontsize=8)
plt.yticks(ticks=yrange, labels=ylabels, fontsize=8)



plt.grid('on')
#save/show fig

plt.savefig("out_path", dpi=1200)


plt.show()

########################################################################################################################
#Diffusion constant
########################################################################################################################


from scipy.special import erfinv



#estimate hardness of 1018 steel as lowest hardness measurement across all data
h_0=228.6



def get_hardness(xdat, ydat, tkeys, point,h_0=h_0):

    
    h_0=h_0

    D_hardness=[]
    for i in range(len(tkeys)):
        
        ydata=np.array(ydat[i])
        xdata=np.array(xdat[i])


        try:

            if len(ydata)<3:

                #if length is less than 3, take H(x,t) to be first point

                average=np.average(ydata[:])

                inner=(ydata[point]-h_0)/(average-h_0)

                D=erfinv(inner)

                other=(2*np.sqrt(float(tkeys[i]))*(1/xdata[point]))

                d_const=(D*other)**(-2)

                D_hardness+=[[d_const,tkeys[i]]]

            else:

                #if length is longer, take a point further from surface since a lot of data went down then up

                average=np.average(ydata[:3])

                inner=(ydata[point]-h_0)/(average-h_0)

                D=erfinv(inner)


                other=(2*np.sqrt(float(tkeys[i]))*(1/xdata[point]))

                d_const=((D*other)**(-2))*(10**(-6))**2

                D_hardness+=[[d_const,tkeys[i], point]]
                
        except IndexError:
            continue

    return D_hardness

end=max(len(x) for x in xdat)

all_hardness_data=[]
for i in range(end-3):
    j=i+3
    
    all_hardness_data+=[get_hardness(xdat, ydat, tkeys, j)]
    

data1=[item for sublist in all_hardness_data for item in sublist]

data1=pd.DataFrame(data1,columns=['Value', 'Time', 'Index'])

pivot_df = data1.pivot(index='Index', columns='Time', values='Value')    


#number of measurement points at each carburization time duration
weights = {
     '500': 14,
    '750': 13,
    '1000': 13,
    '1250': 11,
    '1500': 12,
    '1750': 15,
    '2000': 15,
    '2250': 2,
    '2500': 13,
    '2750': 12,
    '3000': 18
    # Add weights for all other time columns
}

#remove lowest value estimated as h_0 to not skew the result
pivot_df.at[13,'500']=0

df_nonzero = pivot_df.replace(np.nan, 0)


pivot_df_nonzero = df_nonzero

# Calculate weighted values for each column based on the weights.
# This ensures we only work with DataFrame operations.
weighted_values = pivot_df_nonzero * pivot_df_nonzero.columns.map(weights)

# Sum the weighted values across all columns (numerator for the weighted average)
numerator = weighted_values.sum().sum()

# Calculate the denominator: the sum of weights for each non-NaN entry in each column
denominator_weights = pivot_df_nonzero.notna() * pivot_df_nonzero.columns.map(weights)
denominator = denominator_weights.sum().sum()

# Calculate the weighted average
weighted_average = numerator / denominator    
    
print(weighted_average)

#This gives a diffusion constant of 8.426*10^-10 (m^2/s)
#The theoretically predicted value is 3.028*10^-11 (m^2/s), not bad!

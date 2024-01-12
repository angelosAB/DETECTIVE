#uncomment the following to install 
#pip install datapackage
import chardet

#-----------------------Pre-processing of company dataset--------------------
from datapackage import Package
import numpy as np
package = Package('https://datahub.io/core/s-and-p-500-companies/datapackage.json')
# print list of all resources:
# print processed tabular data (if exists any)
Company = []
for resource in package.resources:
    if resource.descriptor['datahub']['type'] == 'derived/csv':
        Company = resource.read()
Sectors = np.array([i[2] for i in Company])
Com_name = np.array([i[1] for i in Company])
Stock_name = np.array([i[0] for i in Company])
Com_name[Com_name=='Estée Lauder Companies'] = 'Estee Lauder'
Com_name[Com_name=='Brown–Forman'] = 'Brown Forman'
Com_name[Com_name=='Verizon Communications'] = 'Verizon'
Com_name[Com_name=='Progressive Corporation'] = 'Progressive Insurance'
Com_name[Com_name=='NortonLifeLock'] = 'LifeLock'
Com_name[Com_name=='News Corp (Class A)'] = 'News Corporation'
Com_name[Com_name=='News Corp (Class B)'] = 'News Corporation'
Com_name[Com_name=='Mondelez International'] = 'Mondelez'
Com_name[Com_name=='Molson Coors Beverage Company'] = 'Molson Coors'
Com_name[Com_name=='S&P Global'] = 'McGraw-Hill'
Com_name[Com_name=='Ralph Lauren Corporation'] = 'Polo Ralph Lauren'
Com_name[Com_name=='Public Service Enterprise Group'] = 'Public Service Electric and Gas'
Com_name[Com_name=='Pinnacle West Capital'] = 'Pinnacle West'
Com_name[Com_name=='Pioneer Natural Resources'] = 'Pioneer Natural'
Com_name[Com_name=='Philip Morris International'] = 'Philip Morris'
Com_name[Com_name=='Truist Financial'] = 'Truist'
Com_name[Com_name=='Tractor Supply Company'] = 'Tractor Supply'
Com_name[Com_name=='The Mosaic Company'] = 'Mosaic Company'
Com_name[Com_name=='The Cooper Companies'] = 'The Cooper Companies, Inc.'
Com_name[Com_name=='Seagate Technology'] = 'Seagate'
Com_name[Com_name=='Regency Centers'] = 'Regency Consulting Inc. (RCI)'
Com_name[Com_name=='Alphabet (Class C)'] = 'Google'
Com_name[Com_name=='Alphabet (Class A)'] = 'Google'
Com_name[Com_name=='Fox Corporation (Class A)'] = 'Fox'
Com_name[Com_name=='Fox Corporation (Class B)'] = 'Fox'
Com_name[Com_name=='International Flavors & Fragrances'] = 'International Flavors & Fragrances Inc'
Com_name[Com_name=='PPL'] = 'PPL Electric Utilities'
Com_name[Com_name=='PNC Financial Services'] = 'PNC Bank'
Com_name[Com_name=='Norwegian Cruise Line Holdings'] = 'Norwegian Cruise Line'
Com_name[Com_name=='NiSource'] = 'NiSource Inc'
Com_name[Com_name=='AES Corp'] = 'The AES Corporation'
Com_name[Com_name=='Agilent Technologies'] = 'Agilent'
Com_name[Com_name=='Air Products & Chemicals'] = 'Air Products'
Com_name[Com_name=='Akamai Technologies'] = 'Akamai'
Com_name[Com_name=='Alaska Air Group'] = 'Alaska Airlines'
Com_name[Com_name=='Albemarle Corporation'] = 'Albemarle Corp'
Com_name[Com_name=='Allstate Corp'] = 'Allstate'
Com_name[Com_name=='Ameren Corp'] = 'Ameren'
Com_name[Com_name=='American Airlines Group'] = 'American Airlines'
Com_name[Com_name=='American Water Works'] = 'American Water'
Com_name[Com_name=='APA Corporation'] = 'Apache'
Com_name[Com_name=='Aptiv'] = 'Delphi Automotive'
Com_name[Com_name=='Arthur J. Gallagher & Co.'] = 'Arthur J. Gallagher'
Com_name[Com_name=='Automatic Data Processing'] = 'ADP'
Com_name[Com_name=='Ball Corp'] = 'Ball Corporation'
Com_name[Com_name=='Bath & Body Works Inc.'] = 'Bath & Body Works'
Com_name[Com_name=='Bio-Rad Laboratories'] = 'Bio-Rad'
Com_name[Com_name=='Biogen'] = 'Biogen Idec'
Com_name[Com_name=='Booking Holdings'] = 'Booking.com'
Com_name[Com_name=='Broadridge Financial Solutions'] = 'Broadridge Financial'
Com_name[Com_name=='C. H. Robinson'] = 'CH Robinson Worldwide'
Com_name[Com_name=='Capital One Financial'] = 'Capital One'
Com_name[Com_name=='Carrier Global'] = 'Carrier'
Com_name[Com_name=='Centene Corporation'] = 'Centene'
Com_name[Com_name=='Carnival Corporation'] = 'Carnival Cruise Lines'
Com_name[Com_name=='Charles River Laboratories'] = 'Charles River Laboratories, Inc.'
Com_name[Com_name=='Charles Schwab Corporation'] = 'Charles Schwab'
Com_name[Com_name=='Chevron Corporation'] = 'Chevron'
Com_name[Com_name=='Chipotle Mexican Grill'] = 'Chipotle'
Com_name[Com_name=='Chubb'] = 'The Chubb Corporation'
Com_name[Com_name=='Cintas Corporation'] = 'CINTAS'
Com_name[Com_name=='Citizens Financial Group'] = 'Citizens Bank'
Com_name[Com_name=='Coca-Cola Company'] = 'Coca-Cola Enterprises'
Com_name[Com_name=='Conagra Brands'] = 'ConAgra Foods'
Com_name[Com_name=='Coterra'] = 'Cabot Oil & Gas'
Com_name[Com_name=='Crown Castle'] = 'Crown Castle International'
Com_name[Com_name=='D. R. Horton'] = 'D.R. Horton, Inc.'
Com_name[Com_name=='Danaher Corporation'] = 'Danaher'
Com_name[Com_name=='Deere & Co.'] = 'John Deere'


Com_name[Com_name=='Dentsply Sirona'] = 'DENTSPLY'
Com_name[Com_name=='Diamondback Energy'] = 'DiamondBack Automotive Accessories'
Com_name[Com_name=='Digital Realty Trust'] = 'Digital Realty'
Com_name[Com_name=='Discovery (Series A)'] = 'Discovery'
Com_name[Com_name=='Discovery (Series C)'] = 'Discovery'
Com_name[Com_name=='Dominion Energy'] = 'Dominion'
Com_name[Com_name=='Dover Corporation'] = 'Dover'
Com_name[Com_name=='Duke Realty Corp'] = 'Duke Realty'
Com_name[Com_name=='Eaton Corporation'] = 'Eaton'
Com_name[Com_name=='Emerson Electric Company'] = 'Emerson'
Com_name[Com_name=='Expedia Group'] = 'Expedia'
Com_name[Com_name=='Expeditors'] = 'Expeditors International Of Washington'
Com_name[Com_name=='Fifth Third Bancorp'] = 'Fifth Third Bank'
Com_name[Com_name=='Fleetcor'] = 'FleetCor'
Com_name[Com_name=='Gap'] = 'GAP'

Com_name[Com_name=='West Pharmaceutical Services'] = 'West Pharmaceutical'
Com_name[Com_name=='W. R. Berkley Corporation'] = 'w.r. berkely corporation'
Com_name[Com_name=='Fifth Third Bancorp'] = 'Fifth Third Bank'
Com_name[Com_name=='Zimmer Biomet'] = 'Zimmer Holdings'
Com_name[Com_name=='Zions Bancorp'] = 'Zions Bank'
Com_name[Com_name=='Williams Companies'] = 'Williams'

Com_name[Com_name=='Vornado Realty Trust'] = 'Vornado Realty'
Com_name[Com_name=='ViacomCBS'] = 'Paramount Pictures'
Com_name[Com_name=='Universal Health Services'] = 'Universal Hospital Services'
Com_name[Com_name=='United Parcel Service'] = 'UPS'
Com_name[Com_name=='Under Armour (Class A)'] = 'Under Armour Record'
Com_name[Com_name=='Under Armour (Class C)'] = 'Under Armour Record'

Com_name[Com_name=='Ulta Beauty'] = 'Ulta Salon Cosmetics & Fragrance'
Com_name[Com_name=='U.S. Bancorp'] = 'U.S. Bank'
Com_name[Com_name=='TJX Companies'] = 'TJX'
Com_name[Com_name=='The Travelers Companies'] = 'Travelers J'
Com_name[Com_name=='T-Mobile US'] = 'T Mobile'
Com_name[Com_name=='Royal Caribbean Group'] = 'Royal Caribbean International'

Com_name[Com_name=='Roper Technologies'] = 'Northrop Grumman'
Com_name[Com_name=='Regeneron Pharmaceuticals'] = 'Regeneron'
Com_name[Com_name=='Raytheon Technologies'] = 'Raytheon'
Com_name[Com_name=='Prologis'] = 'ProLogis'
Com_name[Com_name=='Parker-Hannifin'] = 'Parker Hannifin'
Com_name[Com_name=='Otis Worldwide'] = 'Otis Elevator'

Com_name[Com_name=='American International Group'] = 'AIG.com'
Com_name[Com_name=='Catalent'] = 'Catalent Pharma Solutions'
Com_name[Com_name=='Ceridian'] = 'Ceridian Corporation'
Com_name[Com_name=='Eli Lilly & Co'] = 'Eli Lilly & Company'
Com_name[Com_name=='Freeport-McMoRan'] = 'Freeport-McMoRan Copper & Gold'
Com_name[Com_name=='Generac Holdings'] = 'Generac Power'
Com_name[Com_name=='Hess Corporation'] = 'Hess'
Com_name[Com_name=='Host Hotels & Resorts'] = 'Host Hotels'
Com_name[Com_name=='Huntington Bancshares'] = 'Huntington'
Com_name[Com_name=='IDEX Corporation'] = 'IDEX'

Com_name[Com_name=='Idexx Laboratories'] = 'IDEXX'
Com_name[Com_name=='Ingersoll Rand'] = 'Ingersoll-Rand'
Com_name[Com_name=='Intercontinental Exchange'] = 'ICE'
Com_name[Com_name=='Kraft Heinz'] = 'Kraft Foods'
Com_name[Com_name=='LabCorp'] = 'Labcorp'
Com_name[Com_name=='LKQ Corporation'] = 'LKQ'
Com_name[Com_name=='Mastercard'] = 'MasterCard'
Com_name[Com_name=='McKesson Corporation'] = 'McKesson'
Com_name[Com_name=='Mettler Toledo'] = 'METTLER TOLEDO'
Com_name[Com_name=='MGM Resorts International'] = 'MGM Resorts'
Com_name[Com_name=='Nielsen Holdings'] ='Nielsen'

#----------------------------Network -----------------------------------------------------------------------
Link_path = '/Users/kyriakoslalousis/Downloads/19967886/links.csv'
import pandas as pd
DF_link = pd.read_csv(Link_path)
Layer_name = ['partnership', 'customer','competitor','investment','supplier']
DF_link_remove = DF_link[DF_link.iloc[:,2].isin(Layer_name)]
DF_link_1 = DF_link[(DF_link['Node_1'].isin(Com_name)) & (DF_link['Node_2'].isin(Com_name))]

#--------
Com_name_filter = []
Stock_name_filter = []
Sectors_filter = []
for i in range(len(Com_name)):
    Temp_1 = DF_link[DF_link['Node_1']==Com_name[i]].shape
    Temp_2 = DF_link[DF_link['Node_2']==Com_name[i]].shape
    if Temp_1[0]==0 and Temp_2[0]==0:
        print(i)
    else:
        if Stock_name[i]!='KSU':
            Com_name_filter.append(Com_name[i])
            Stock_name_filter.append(Stock_name[i])
            Sectors_filter.append(Sectors[i])
Stock_name_filter_rep = [i.replace('.','-') for i in Stock_name_filter]
Business_DF = pd.DataFrame({'Company':Com_name_filter,'Stock_symbol':Stock_name_filter_rep,'Sector':Sectors_filter})
Business_DF.to_csv('DF_Stock')

DF_link_2 = DF_link_1[(DF_link_1['Node_1'].isin(Com_name_filter)) & (DF_link_1['Node_2'].isin(Com_name_filter))]
DF_link_2.to_csv('Stock_connection_DF')

#----------------------------------Covariates Construction------------------------------------
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()
import pandas as pd
DFSTOCK = pd.read_csv('C:\D_Disk\Covariate-assisted Multi-Layer Community Detection\Stock_processing_features\DF_Stock')
Stock_name_filter_rep = DFSTOCK['Stock_symbol'].tolist()
stock = pdr.get_data_yahoo(Stock_name_filter_rep[0], '2021-01-01', '2021-6-01')
Covs = stock.iloc[:,3]
for name in Stock_name_filter_rep[1::]:
    stock = pdr.get_data_yahoo(name,'2021-01-01','2021-06-01')
    Covs = pd.concat([Covs,stock.iloc[:,3]],axis=1)
Covs = Covs.dropna(axis=0)
import numpy as np
Features = np.array(Covs).T
#Features_per = (Features[:,1::] - Features[:,0::-1])/(Features[:,0:-1])
Features_std = (Features - Features.mean(1, keepdims=True)) / Features.std(1, keepdims=True)
np.save('Stock_features',Features_std)


"""
Project title: 888 Financial Advisor

Project objective:
Our vision was to create an application which allows users to have a high-level overview of financial performance
of a listed company using financial ratios, while also providing trading suggestions based on a well-known
technical indicator – Bollinger bands, which is widely used in algorithmic trading realm.
The application aims to combine both fundamental and technical analysis under the hood of a user-friendly GUI.

Group 8 members:
Han YiChen
Kee E Peng
Neogy Debarati
Wendy Yang YingTong
"""

#import the modules we need for creating a GUI

import tkinter as tk
import tkinter.messagebox

import lxml
from lxml import html
import requests

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as dt

#---------------start of functions for GUI------------------

# user guide & clear button & details
# for "user guide" GUI button
def UserGuide():
    """show user guide in a messagebox"""
    
    tkinter.messagebox.showinfo('User Guide',
                                '1. Enter the stock ticker, the start date, end date, and rolling period into the box following the format stated.\n\n'
                                '2. Click ‘check now’ button.\n\n'
                                '3. The results will be displayed below.\n\n'
                                '4. For more information on Bollinger Band Strategy, visit https://www.investopedia.com/trading/using-bollinger-bands-to-gauge-trends.\n\n'
                                '5. For more information on fundamental analysis, visit https://www.investopedia.com/terms/r/ratioanalysis.asp \n'
                                '6. To re-enter, click ‘clear’ button.\n\n'
                                '7. For detailed explanation, click ’explain’ button.'
)

# output following strings for "explain" GUI button
def ResultExplanation():
    """show result explanation in a messagebox"""
    
    tkinter.messagebox.showinfo('Result Explanation',"""
About Bollinger Band:

Bollinger Bands are a type of chart indicator for technical analysis and have
become widely used by traders in many markets, including stocks, futures, and
currencies. The bands are often used to determine overbought and oversold
conditions. Using only the bands to trade is a risky strategy since the indicator
focuses on price and volatility, while ignoring a lot of other relevant information.

The Bollinger Band formula consists of the following:
Upper Band = MA(TP,n)+m∗σ[TP,n]
Lower Band = MA(TP,n)−m∗σ[TP,n]
where:
MA = Moving average
TP (typical price) = Adjusted close price is used here
n = Number of days in lookback period
m = Number of standard deviations
σ[TP,n] = Standard deviation over last n periods of TP

Using the bands as overbought/oversold indicators relies on the concept of mean
reversion of the price. When the price of the asset breaks below the lower band
of the Bollinger Bands, prices have perhaps fallen too much and are due to bounce.
On the other hand, when price breaks above the upper band, the market is perhaps
overbought and due for a pullback.

About ratio analysis:

The Current Ratio measures a company's ability to pay short-term obligations or
those due within one year.A good current ratio is between 1.2 to 2, which means that
the business has 2 times more current assets than liabilities to covers its debts.
A current ratio below 1 means that the company doesn't have enough liquid assets
to cover its short-term liabilities.

A good net profit ratio will vary considerably by industry,
but a 10% net profit margin is considered average,
a 20% margin is considered high and a 5% margin is low.

The higher the asset turnover ratio, the more efficient a company
is at generating revenue from its assets.

Return on assets gives an indication of the capital intensity of the company.
Depending on industry, ROAs over 5% are generally considered good.

ROE is a measure of management's ability to generate income from the equity
available to it. ROEs of 15-20% are generally considered good. ROE is also a factor
in stock valuation, in association with other financial ratios.

""")

#---------------end of functions for GUI------------------

#---------------start of functions for Bollinger Band Strategy Output------------------

def ConvertToDatetime(strDate):
    """
    Convert string to idealized calendar, the Gregorian calendar.

    The function takes in a date in string format and convert it into an
    Gregorian calendar format using class datetime.date from datetime library.
    Class datetime.date takes in arguments in following form: (year, month, day).

    Args:
        strDate: A date in string format.

    Return:
        The converted date in Gregorian calendar format.

    Raises:
        ValueError: if the followings are not satisfied:
            MINYEAR <= year <= MAXYEAR
            1 <= month <= 12
            1 <= day <= number of days in the given month and year
    """

    listDate = strDate.split("-") # split the string based on this separator

    # convert all strings in the list to int
    for i in range(len(listDate)):
        listDate[i] = int(listDate[i])

    dateTime = dt.date(listDate[0], listDate[1], listDate[2]) # convert to Gregorian datetime format

    return dateTime

def ExecuteBollingerBandStrategy(ticker, startDate, endDate, rolling_period):
    """
    Execute Bollinger Band strategy and compare with buy and hold returns of the stock.

    The function will run Bollinger Band strategy on given ticker based on given
    start date and end date. It will use a given parameter as rolling period for
    the Bollinger Band strategy. Finally, the returns from speculating the stock
    using Bollinger Band is compared with returns result from buying and holding
    of the stock. It aims to give general sense of returns from different investment
    approaches.

    Args:
        ticker: stock ticker.
        startDate: intended start date for Bollinger Band strategy backtest
        endDate: intended end date for Bollinger Band strategy backtest
        rolling_period: intended rolling parameter for Bollinger Band strategy backtest

    Return:
        None. The final graph will be saved as an image to output to GUI.
    """
    
    df = yf.download(ticker, startDate, endDate) # download Open, High, Low, CLose, Adj Close and Volume data, store into dataframe,

    # create the indicators
    # sma_rolling_period = Simple Moving Average calculated based on "rolling_period" as its lookback parameter
    # rolling_period_STD = standard deviations calculated based on prices througout the "rolling_period"
    # BB_upper_band = 2 standard deviations above sma_rolling_period
    # BB_lower_band = 2 standard deviations below sma_rolling_period
    df['sma_rolling_period'] = df['Adj Close'].rolling(window = rolling_period).mean()
    df['rolling_period_STD'] = df['Adj Close'].rolling(window = rolling_period).std()
    df['BB_upper_band'] = df['sma_rolling_period'] + 2 * df['rolling_period_STD']
    df['BB_lower_band'] = df['sma_rolling_period'] - 2 * df['rolling_period_STD']

    # create signals

    # buy signals
    df['signal'] = np.where((df['Adj Close'] < df['BB_lower_band'])
                              & (df['Adj Close'].shift(1) > df['BB_lower_band'].shift(1)),
                              1, 0)
    #sell signals
    df['signal'] = np.where((df['Adj Close'] > df['BB_upper_band'])
                              & (df['Adj Close'].shift(1) < df['BB_upper_band'].shift(1)),
                              -1, df['signal'])

    # identify positions hold throughout the use of this strategy
    df['position'] = df['signal'].shift(1) # assume that we only able to get into trade after signal generated
    df['position'] = df['position'].replace(to_replace = 0, method = 'ffill') # position is closed only when signal for
    # opposite trading direction is generated

    # calculate and compare returns resulting from:
    # (1) buy and hold return
    # (2) return realized through this strategy
    df['Buy & Hold Returns'] = df['Adj Close'].pct_change() # buy and hold return, pct_change() is percentage change between
    # current and previous day's price
    df['Bollinger Band Strategy Returns'] = df['Buy & Hold Returns'] * df['position'] # the return realized through this strategy

    # get return values from respective approach and plot equity curves
    # df['Buy & Hold Returns'] and df['Bollinger Band Strategy Returns'] refer to return on that day itself, in order to get equity curve we
    # need to plus one on each of the returns before calculate cumulative product.
    # Use vectorized operations to carry out calculation of cumulative product
    buyAndHoldReturn = np.round((1 + df['Buy & Hold Returns']).cumprod()[-1],2)
    bollingerReturn = np.round((1 + df['Bollinger Band Strategy Returns']).cumprod()[-1],2)
    (1 + df[['Buy & Hold Returns','Bollinger Band Strategy Returns']]).cumprod().plot(grid = True, figsize = (6,4))
    plt.title('Bollinger Band Strategy Return: {}  Buy and Hold Return: {}'.format(bollingerReturn,buyAndHoldReturn))
    
    plt.savefig('rolling.png') # save the graph as an image

#---------------end of functions for Bollinger Band Strategy Output------------------

#---------------start of functions for financial ratios output------------------

# Scraping data from website and building a data frame
def GetTable(url):
    """
    Fetch the page, parse it - splits the given sequence of characters or values (text) into smaller parts,
    scrapes data from yahoo finance and build a data frame

    Args:
        url: a string of website where the data are extracted from

    Return:
        dfOrg: The Original Dataframe
        dfRot: Transpose Dataframe
    """
    
    # Define request headers to fetch the page by the browser. 
    headers = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'en-US,en;q=0.9',
    'Cache-Control': 'max-age=0',
    'Pragma': 'no-cache',
    'Referrer': 'https://google.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.120 Safari/537.36'
}
    
    # Fetch the page that we're going to parse, using the request headers defined above
    page = requests.get(url, headers)
    
    # Parse the page with LXML, so that we can start doing some XPATH queries
    # to extract the data that we want
    tree = html.fromstring(page.content)

    # Smoke test that we fetched the page by fetching and displaying the H1 element
    tree.xpath("//h1/text()")
    tableRows = tree.xpath("//div[contains(@class, 'D(tbr)')]")

    # Ensure that some table rows are found; if none are found, then the layout of the website has been changed.
    parsedRows = []
    #Start of the loop - this will create the dataframe of the rows parsed  
    for tableRow in tableRows:
        parsedRow = []
        el = tableRow.xpath("./div")
        noneCount = 0
        #Start of the loop where each value of a row is checked, if value found, 
        #the cell value will be replaced with the value found
        for rs in el:
            try:
                (text,) = rs.xpath('.//span/text()[1]')
                parsedRow.append(text)
            #If there is no value or there is error, for that particular element parsed, 
            #it will be replaced with Nan 
            except ValueError:
                parsedRow.append(np.NaN)
                noneCount += 1
        #The loop would be continued for each instance end ensure it is less than 4 and form the rows
        if (noneCount < 4):
            parsedRows.append(parsedRow)
            
    # Create a DataFrame of the Parsed rows, set the index to the first column as Period ending
    # Transpose the DataFrame, so that our header contains the account names
    df = pd.DataFrame(parsedRows)
    dfOrg = df
    df = pd.DataFrame(parsedRows)
    df = df.set_index(0) 
    df = df.transpose() 

    # Rename the "Breakdown" column to "Date"
    cols = list(df.columns)
    cols[0] = 'Date'
    df = df.set_axis(cols, axis='columns', inplace=False)
    dfRot = df
    
    return dfOrg, dfRot

# Calculate the ratios needed, and build 2 tables for Income Statement and Balance Sheet ratios separately
def CalculateRatio(ticker):
    """
    Calculate the ratios needed from Balance Sheet and Income Statement

    Args:
        ticker: represents company user is searching for

    Return:
        Two table containing Income Statement and Balance Sheet Ratios and dates separately

    """
    
    # Yahoo Finance links
    urlBs = 'https://sg.finance.yahoo.com/quote/' + ticker + '/balance-sheet?p=' + ticker
    urlIs = 'https://sg.finance.yahoo.com/quote/' + ticker + '/financials?p=' + ticker
    urlCf = 'https://sg.finance.yahoo.com/quote/' + ticker + '/cash-flow?p='+ ticker

    # get Balance Sheet
    bsOrginal, bsTranspose = GetTable(urlBs)

    # get Income Statement
    isOrginal, isTranspose = GetTable(urlIs)

    # copy columns of dataframe to do BS analysis
    bsAnalysis = pd.DataFrame(bsTranspose['Date'])
    year = bsTranspose['Date'].str.replace('/','').astype(int)%10000
    bsAnalysis['Year'] = year
    bsAnalysis

    #calculation of Current Ratio
    currentAssets = bsTranspose['Total current assets'].str.replace(',', '').astype(int)
    currentLiabilities = bsTranspose['Total current liabilities'].str.replace(',', '').astype(int)
    currentRatio = currentAssets / currentLiabilities
    bsAnalysis['Current Ratio'] = currentRatio

    # copy columns of dataframe to do income statement analysis
    isAnalysis = pd.DataFrame(isTranspose['Date'])
    year = (isTranspose['Date'].str.replace('/','')[1:]).astype(int)%10000
    isAnalysis['Year'] = year
    
    #calculation of Net profit Ratio
    revenue = isTranspose['Total revenue'].str.replace(',', '').astype(int)
    netIncome = isTranspose['Net income'].str.replace(',', '').astype(int)
    netProfitMargin = netIncome / revenue
    isAnalysis['NPR'] = netProfitMargin

    #calculation of Return on Equity
    netProfit = isTranspose['Net income available to common shareholders'].str.replace(',', '').astype(int)
    averageShareholderEquityForThePeriod = bsTranspose['Total stockholders\' equity'].str.replace(',', '').astype(int)
    returnOnEquity = netProfit / averageShareholderEquityForThePeriod
    isAnalysis['ROE'] = returnOnEquity

    #calculation of Return on Assets
    averageAssetsForThePeriod = bsTranspose['Total assets'].str.replace(',', '').astype(int)
    assetsTurnover = revenue / averageAssetsForThePeriod
    isAnalysis['Asset Turnover'] = assetsTurnover
    returnOnAssets = netProfitMargin / assetsTurnover
    isAnalysis['ROA'] = returnOnAssets
    isAnalysis.drop([1])

    return isAnalysis,bsAnalysis

# Function to print out the ratios as asked by the user or print error message if it is unavailable.
def GetRatioOutput(symbol,bsAnalysis,isAnalysis,startYear,endYear):
    """
    Print out the ratios as asked by the user or print error message if it is unavailable.

    Args:
        symbol: represents company the user is searching for
        bsAnalysis: table containing balance sheet ratios
        isAnalysis: table containing income statement ratios
        startYear: the year of input start date
        endYear: the year of input end date

    Return:
        msg: the balance sheet ratios to be displayed to the users
        printableIsDf: the income statement ratios to be displayed to the users
    """ 

    msg = "Company with symbol: {} are to be analysed.".format(symbol) +'\n'
    userInput = [n for n in range(startYear,endYear+1)]
    
    unavailableYear = [] # build list to record year with unavailable data

    # for year in between user input start and end date
    for year in userInput:
        # if ratois for the year is avaliable
        if year in list(bsAnalysis['Year']):
            # prepare output message for BS_analysis--current ratio
            dateNumber = list(bsAnalysis['Year']).index(year) + 1
            msg = msg + "The Current Ratio for year {} is {:.2f}".format(year,bsAnalysis['Current Ratio'][dateNumber]) + '\n'
        # if ratois for the year is not avaliable
        else:
            unavailableYear.append(year)
            
    # inform users which year's ratio is unavailable
    if unavailableYear == []:
        print()
    else:
        msg = msg + "Data needed to calculate current ratio for year{} is unavailable".format(unavailableYear) + '\n'

    # Build a table for Income Statement Analysis
    IS = list() # build list to record output
    i = 0 # initiate i for building dataframe
    printableIsDf = pd.DataFrame(IS) # store initial income statement analysis into printable form

    # check each year with available data in analysis
    for year in list(bsAnalysis['Year']):
        i=i+1
        # check whether that year is required by user, create output data frame
        if year in userInput:
            pd.DataFrame(isAnalysis.iloc[int(i),:]) 
            IS.append(isAnalysis.iloc[int(i),:]) 

    isDf = pd.DataFrame(IS) # record data to be output
    isDf = isDf.drop(isDf.columns[0],axis=1) # drop the date column to avoid redundancy
    isDf['Year'] = isDf['Year'].astype(int) # convert existing 'float' data type in 'Year' column to 'int' type
    printableIsDf = printableIsDf.append(isDf) # transform into printable table
    printableIsDf.index = np.arange(1, len(printableIsDf) + 1) # reset the index in such a way that when output to screen, the index shown starts from 1

    return msg,printableIsDf

#---------------end of functions for financial ratios output------------------

#---------------start of main program function------------------

def ClearResultLabels():
    """clear all outputs on the GUI"""

    # clear all output labels on GUI
    imageOutput.pack_forget() # it is declared global in CheckStock() to access it from here
    ratioAnalysisLabel_1.pack_forget()
    ratioAnalysisLabel_2.pack_forget()
    
def CheckStock():

    # clear exsiting labels
    
    # at the start of this function, clear all exisitng output labels
    # exception will be raised if any label is not existent
    try:
        ClearResultLabels()

    # simply do nothing and continue the function, if any of the label is not existent and exception is raised
    except:
        pass

    # check for ticker validity
    
    ticker = stock.get() # get stock ticker from GUI entry box

    errorMessage = '' # create an empty string for error message

    # check if ticker is valid through yahoo finance's balance sheet URL for that particular ticker
    try:
        test_url = 'https://sg.finance.yahoo.com/quote/' + ticker + '/balance-sheet?p=' + ticker
        response = GetTable(test_url) # if the ticker is not valid, exception will be raised here

    # if ticker is not valid, output error message through GUI
    except:
        errorMessage= 'please enter a valid ticker'
        tkinter.messagebox.showinfo('ERROR',errorMessage)
        return # function stops here

    # check for start and end date validity
    
    # get start date and end date from GUI entry box
    startDate = startDateEntry.get()
    endDate = endDateEntry.get()

    # check if start date and end date are in proper format and convertible to Gregorian datetime format
    try:
        startDate = ConvertToDatetime(startDate)
        endDate = ConvertToDatetime(endDate)

        # if start date is later than end date, output error message through GUI
        if startDate>endDate:
            errorMessage = 'start date must be ealier that end date'
            tkinter.messagebox.showinfo('ERROR',errorMessage)
            return # function stops here

    # if start date or end date is invalid, output error message through GUI
    except:
        errorMessage = 'please enter a valid date'
        tkinter.messagebox.showinfo('ERROR',errorMessage)
        return # function stops here

    # check for rolling period validity

    # get rolling period from GUI entry box
    rolling_period = rollingPeriodEntry.get()

    # check if rolling period is valid and convertible to int data type
    try:
        rolling_period = int(rolling_period)

        # if rolling period is larger than available date range, output error message through GUI
        if rolling_period > (endDate-startDate).days:
            errorMessage = 'please enter a valid rolling period'
            tkinter.messagebox.showinfo('ERROR',errorMessage)
            return # function stops here

    # if rolling period is invalid, output error message through GUI
    except:
        errorMessage = 'please enter a valid rolling period'
        tkinter.messagebox.showinfo('ERROR',errorMessage)
        return # function stops here

    # Bollinger Band strategy calculation starts
    ExecuteBollingerBandStrategy(ticker, startDate, endDate, rolling_period)

    # Financial ratios calculation starts
    (isAnalysis,bsAnalysis)=CalculateRatio(ticker) 

    #----------------start to show result-----------------------------------

    global imageOutput # declare global such that ClearResultLabels() able to access

    #add a image
    imageStored = tk.PhotoImage(file='rolling.png')
    imageOutput = tk.Label(root, image=imageStored)
    imageOutput.pack(side=tk.LEFT)

    # Get required output from ratio analysis
    (msg,printableIsDf) = GetRatioOutput(ticker,bsAnalysis,isAnalysis,startDate.year,endDate.year)

    # Display message of Balance Sheet Ratio
    ratioAnalysisLabel_1.configure(text = msg)
    ratioAnalysisLabel_1.pack()

    # Display table of Income Statement Ratio
    ratioAnalysisLabel_2.configure(text = printableIsDf)
    ratioAnalysisLabel_2.pack()    

    root.mainloop()
    
#---------------end of main program function------------------

#---------------main body of code------------------

#create a GUI window.
root = tk.Tk()

#set the title
root.title("888 Financial Advisor v1.0")

#set the size
root.geometry("1500x800")

#add a label for the start text.
welcomeLabel_1 = tk.Label(root, text = '888 Financial Advisor',  fg = 'green', relief = tk.RAISED, borderwidth = 3,font = ('Courier',15,'bold'))
welcomeLabel_1.pack()

#user guide button
welcomeLabel_2 = tk.Label(root, text="If this is the first time you use our app, make sure to check user guide", font=('Helvetica', 12))
welcomeLabel_2.pack()
guideBtn = tk.Button(root,text = "check user guide",command=UserGuide)
guideBtn.pack()
startLabel = tk.Label(root, text="\n Enter the stock you want to check", font=('Helvetica', 12))
startLabel.pack()

#add entry box
stock = tk.Entry()
stock.pack()

#start date, end date, rolling period entry box
startDateLabel = tk.Label(root, text="\n Please input start date (e.g. 2018-01-02)", font=('Helvetica', 12))
startDateLabel.pack()
startDateEntry = tk.Entry()
startDateEntry.pack()
endDateLabel = tk.Label(root, text="\n Please input end date (e.g. 2019-05-01)", font=('Helvetica', 12))
endDateLabel.pack()
endDateEntry = tk.Entry()
endDateEntry.pack()
rollingLabel = tk.Label(root, text="\n Enter rolling period(e.g. 20)", font=('Helvetica', 12))
rollingLabel.pack()
rollingPeriodEntry = tk.Entry()
rollingPeriodEntry.pack()

#add check button
ckeckBtn = tk.Button(root,text = "check now",command=CheckStock)
ckeckBtn.pack()

#add clear button
clearBtn = tk.Button(root,text = "clear",command=ClearResultLabels)
clearBtn.pack()

#add Result Explanation button
explainBtn = tk.Button(root,text = "explain",command=ResultExplanation)
explainBtn.pack()

#result labels
ratioAnalysisLabel_1 = tk.Label(root)
ratioAnalysisLabel_2 = tk.Label(root)

root.mainloop()

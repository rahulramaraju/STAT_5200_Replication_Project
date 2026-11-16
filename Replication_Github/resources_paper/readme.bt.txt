Huixin Bi and Nora Traum, "Estimating Fiscal Limits: The Case of 
Greece," Journal of Applied Econometrics, Vol. 29, No. 7, 2014, pp. 
1053-1072.

The file BiTraum_Appendix.pdf contains an online appendix.

The data used for estimation are contained in BiTraum_data.txt, an
ASCII file in DOS format that is zipped in bt-data.zip. Unix/Linux
users should use "unzip -a".

The file contains 40 observations (in rows) for 5 data series (in
columns).  The series are organized as follows: real output (Y), government
spending to GDP ratio (G), tax revenue to GDP ratio (TR), government
debt to GDP ratio (B), and the 10-year real interest rate (R). 

Five observables for Greece over the period 2001Q1-2010Q4 are used for
the estimation: real output, government spending to GDP ratio, tax
revenue to GDP ratio, government debt to GDP ratio, and the 10-year
real interest rate.  Thus, we have 40 observations for each 5 data
series.  Details of how each data series were constructed are as
follows: 

Real GDP. Constructed by dividing the nominal quarterly gross domestic
product from the OECD quarterly National Accounts (using the
expenditure approach, series B1 GE) by the gross domestic product
deflator (constructed using the expenditure approach, series B13).

Real Gov. Spending. Constructed using general government final
consumption expenditure from the OECD quarterly National Accounts
(series P3S13) divided by the gross domestic product deflator
(constructed using the expenditure approach, series B13).

Real Tax Revenue. A quarterly tax revenue series is not provided by
the OECD. Thus, we construct a quarterly measure in the following way.
First, we construct a measure of total tax revenue by combining
Eurostat quarterly series for tax receipts on income/wealth,
production and imports, capital taxes, and social contributions. We
seasonally adjust this series using Demetra+ and the tramo-seat RSA4
specification. Next, using an annual nominal tax revenue series from
the OECD volume 90 (consisting of indirect and direct taxes and social
security contributions, TIND + TY + SSRG), we interpolate a quarterly
frequency series using the method of Chow and Lin (1971) and the
seasonally adjusted quarterly Eurostat tax revenue series for the
interpolation. Finally, we construct a quarterly real tax revenue
series by dividing the interpolated series by the OECD’s gross
domestic product deflator (constructed using the expenditure approach,
series B13).

Real Gov. Debt. A quarterly government debt series is not provided by
the OECD. Thus, we construct a quarterly measure in the following way.
First, we seasonally adjust using Demetra+ and the tramo-seat RSA4
specification the Eurostat quarterly series for nominal gross
government consolidated debt. Next, using the annual nominal gross
public debt series (under the Maastricht criterion) from the OECD
volume 90, we interpolate a quarterly frequency series using the
method of Chow and Lin (1971) and the seasonally adjusted quarterly
Eurostat debt series for the interpolation. Finally, we construct a
quarterly real debt series by dividing the interpolated series by the
OECD’s gross domestic product deflator (constructed using the
expenditure approach, series B13).

Real Interest Rate. To construct a 10-year real interest rate measure,
we use data for the nominal interest rate, (taken from the BIS) and
the expected inflation rate, πe. Our measure of expected inflation for
Greece is the expected inflation series from the Survey of
Professional Forecasts EU-area five year ahead expected inflation
series. The gross real interest rate is constructed using the
relation, R = (1 + i)/(1+πe), where R stands for the gross real
interest rate, i stands for the nominal interest rate, and πe is our
measure of inflation expectations.

We calculate the government spending to GDP ratio, tax revenue to GDP
ratio, and government debt to GDP ratio by taking each real fiscal
series described above and dividing by our real GDP series. The
estimation uses these three series, along with the real GDP and real
interest rate series described above. For each series, we transform
the series into percentage deviations from its mean value over the
period 2001Q1-2010Q4. In addition, the real GDP series is linearly
detrended.


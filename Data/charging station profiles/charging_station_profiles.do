import excel "C:\Users\atpha\Documents\Postdocs\Projects\HDV\Data\charging station profiles\charging_station_profiles.xlsx", cellrange(A1:C171) sheet("case 1") firstrow clear
drop A
save "C:\Users\atpha\Documents\Postdocs\Projects\HDV\Data\charging station profiles\case1.dta", replace

import excel "C:\Users\atpha\Documents\Postdocs\Projects\HDV\Data\charging station profiles\charging_station_profiles.xlsx", cellrange(A1:C162) sheet("case 2") firstrow clear
drop A
save "C:\Users\atpha\Documents\Postdocs\Projects\HDV\Data\charging station profiles\case2.dta", replace

import excel "C:\Users\atpha\Documents\Postdocs\Projects\HDV\Data\charging station profiles\charging_station_profiles.xlsx", cellrange(A1:C153) sheet("case 3") firstrow clear
drop A
save "C:\Users\atpha\Documents\Postdocs\Projects\HDV\Data\charging station profiles\case3.dta", replace


use "C:\Users\atpha\Documents\Postdocs\Projects\HDV\Data\charging station profiles\case1.dta", clear

merge 1:1 lon lat using "C:\Users\atpha\Documents\Postdocs\Projects\HDV\Data\charging station profiles\case2.dta"
rename _merge _merge1

merge 1:1 lon lat using "C:\Users\atpha\Documents\Postdocs\Projects\HDV\Data\charging station profiles\case3.dta"
rename _merge _merge2

gen case1 = 1 if _merge1==1 | _merge1==3
gen case2 = 1 if _merge1==2 | _merge1==3
replace case1=0 if case1==.
replace case2=0 if case2==.

gen case3=1 if _merge2==3
replace case1=1 if _merge2==3
replace case2=1 if _merge2==3

replace case1=1 if _merge2==1 & _merge1==1
replace case3=0 if _merge2==1 & _merge1==1
replace case2=1 if _merge2==1 & _merge1==2
replace case3=0 if _merge2==1 & _merge1==2
replace case1=1 if _merge2==1 & _merge1==3
replace case2=1 if _merge2==1 & _merge1==3
replace case3=0 if _merge2==1 & _merge1==3

replace case3=1 if _merge2==2
replace case1=0 if _merge2==2
replace case2=0 if _merge2==2

drop _merge1 _merge2

sort lon lat

gen index = _n

export excel "C:\Users\atpha\Documents\Postdocs\Projects\HDV\Data\charging station profiles\three_cases.xlsx", replace

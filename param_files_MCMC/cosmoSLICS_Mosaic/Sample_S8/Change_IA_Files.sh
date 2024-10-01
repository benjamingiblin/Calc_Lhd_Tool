#!/bin/bash
# 11/09/2024: change the IA paramfiles:

for IA in `seq -6.0 6.0`; do
    echo $IA;
    p1=params_Data_xipm-SNR_tomoKiDS1000_Omm-S8-h-w0_Data50LOS_UC-C-Auto-Cross_All_IA0.0-added.dat;
    p2=params_Data_xipm-SNR_tomoKiDS1000_Omm-S8-h-w0_Data50LOS_UC-C-Auto-Cross_All_IA${IA}-added.dat;
    if [ "$IA" != "0.0" ]; then # dont change the 0.0 file.
	cp $p1 $p2
	find $p2 -exec sed -i "s#IAbias0.0-added.dat *#IAbias${IA}-added.dat #" {} + # change PDFs
	find $p2 -exec sed -i "s#IAbias0.0-added.asc *#IAbias${IA}-added.asc #" {} + # change xipm
	find $p2 -exec sed -i "s#IA0.0-added *#IA${IA}-added #" {} + # change CombName
    fi
done

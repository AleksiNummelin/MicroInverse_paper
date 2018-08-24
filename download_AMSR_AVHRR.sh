##################
# DOWNLOAD FILES #
##################
filepath='~/sst_data/amsr_avhrr/'
mkdir ${filepath}
cd ${filepath}
#
for i in {4..10} 
    do 
    for k in {1..12} 
        do 
        printf -v j "%02d" $i
        printf -v m "%02d" $k
        wget -r -l1 --no-parent --no-check-certificate -A.nc https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/access/amsr-avhrr/20$j$m/; 
    done
done


###########################
# Combine to annual files #
###########################
#
for i in {4..10} 
    do 
    printf -v j "%02d" $i
    mkdir ${filepath}sst_20${j}/
    cd ${filepath}/sst_20${j}/
    mv ${filepath}www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/access/amsr-avhrr/20${j}*/*.nc .
    FILES=amsr-avhrr-v2.20${j}*.nc
    #
    echo "create dummy files \n"
    for f in $FILES
        do
          filein=$f
          fileout='tmp.'$f
          ncks -O --mk_rec_dmn time $filein $fileout
        done
    #
    files=`ls tmp.amsr-avhrr-v2.20${j}*.nc`
    #
    echo "create combined file \n"
    ncrcat -O $files sst_amsr_avhrr_20${j}.nc
    rm tmp.amsr-avhrr-v2.20${j}*.nc
    mv sst_amsr_avhrr_20${j}.nc ${filepath}
done
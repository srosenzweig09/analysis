rect='false'
sphere='false'

ARGS=$(getopt -a --options rs --long "rectangular,spherical" -- "$@")
eval set -- "$ARGS"
while true; do
  case "$1" in
    -r|--rectangular)
      rect="true"
      shift;;
    -s|--spherical)
      sphere="true"
      shift;;
    --)
      break;;
     *)
      printf "Unknown option %s\n" "$0";;
    #   exit 1;;
  esac
done

# if ${rect}; then
#     echo "[BASH] Running rectangular skimRegions"
#     python scripts/skimRegions.py --cfg config/rectConfig.cfg --rectangular --all-signal || exit 
#     echo "[BASH] Running HiggsCombine"
#     ssh srosenzw@cmslpc-sl7.fnal.gov 'cd /uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/; source env_standalone.sh > /dev/null; make -j 8 > /dev/null; make > /dev/null; python3 runCombine.py --rect'
# fi

if ${sphere}; then
    echo "[BASH] Generating new spherical config file"
    python scripts/sphere_scanner.py --rIn $2 --rOut $3
    echo "[BASH] Running spherical skimRegions"
    python scripts/skimRegions.py --cfg config/sphereConfig_new.cfg --spherical --all-signal || exit 
    ssh srosenzw@cmslpc-sl7.fnal.gov 'cd /uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/; source env_standalone.sh > /dev/null; make -j 8 > /dev/null; make > /dev/null; python3 runCombine.py --sphere --tag '${2}_${3}
fi
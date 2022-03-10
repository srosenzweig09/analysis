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
      printf "Unknown option %s\n" "$1";;
    #   exit 1;;
  esac
done

if ${rect}; then
    echo "[BASH] Running rectangular skimRegions"
    python scripts/skimRegions.py --cfg config/rectConfig.cfg --rectangular || exit 
fi

if ${sphere}; then
    echo "[BASH] Running spherical skimRegions"
    python scripts/skimRegions.py --cfg config/rectConfig.cfg --spherical || exit 
fi

echo "[BASH] Running HiggsCombine"
ssh srosenzw@cmslpc-sl7.fnal.gov 'cd /uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/; source env_standalone.sh > /dev/null; make -j 8 > /dev/null; make > /dev/null; python3 generateDataCards.py'
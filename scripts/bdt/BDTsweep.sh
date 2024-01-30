# Nestimators=(10 20 30 40 50 60 75 85 100)
Nestimators=(75 80 85 90 95 100)
# learningRate=(.01 .05 .1 .15 .2 .25 .3)
learningRate=(.13 .14 .15 .16 .17)
minLeaves=(200 250 300 350 400)
# minLeaves=(220 230 240 250 260 270 280)
# minLeaves=(195 196 197 198 199 200 201 202 203)
maxDepth=(1 2 3 4 5 6 7)
GBsubsample=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

for n in ${Nestimators[@]}; do
   python scripts/skimRegions.py --bias --testing --Nestimators ${n} || exit
done


# hyperparameters=(${Nestimators} ${learningRate} ${maxDepth} ${minLeaves} ${GBsubsample})

# hypernames=("nestimators" "learningRate" "maxDepth" "minLeaves" "GBsubsample")

# i=0
# for hyper in ${hyperparameters[@]}; do
#     for n in ${hyper[@]}; do
#         echo ${hypernames[i]}
#         python scripts/skimRegions.py --cfg config/regionConfig.cfg --dHHH --spherical --${hypernames[i]} ${n}
#     done
#     let i=i+1
# done
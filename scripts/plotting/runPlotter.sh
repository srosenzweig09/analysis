set -e

python scripts/regionPlotter.py --cfg config/rectConfig.cfg --CRedge 100 --VRedge 50
python scripts/regionPlotter.py --cfg config/rectConfig.cfg --CRedge 100 --VRedge 60
python scripts/regionPlotter.py --cfg config/rectConfig.cfg --CRedge 100 --VRedge 70

python scripts/regionPlotter.py --cfg config/rectConfig.cfg --CRedge 125 --VRedge 50
python scripts/regionPlotter.py --cfg config/rectConfig.cfg --CRedge 125 --VRedge 60
python scripts/regionPlotter.py --cfg config/rectConfig.cfg --CRedge 125 --VRedge 75

python scripts/regionPlotter.py --cfg config/rectConfig.cfg --CRedge 150 --VRedge 50
python scripts/regionPlotter.py --cfg config/rectConfig.cfg --CRedge 150 --VRedge 75
python scripts/regionPlotter.py --cfg config/rectConfig.cfg --CRedge 150 --VRedge 100

python scripts/regionPlotter.py --cfg config/rectConfig.cfg --CRedge 175 --VRedge 50
python scripts/regionPlotter.py --cfg config/rectConfig.cfg --CRedge 175 --VRedge 75
python scripts/regionPlotter.py --cfg config/rectConfig.cfg --CRedge 175 --VRedge 100
python scripts/regionPlotter.py --cfg config/rectConfig.cfg --CRedge 175 --VRedge 125

python scripts/regionPlotter.py --cfg config/rectConfig.cfg --CRedge 200 --VRedge 50
python scripts/regionPlotter.py --cfg config/rectConfig.cfg --CRedge 200 --VRedge 60
python scripts/regionPlotter.py --cfg config/rectConfig.cfg --CRedge 200 --VRedge 75
python scripts/regionPlotter.py --cfg config/rectConfig.cfg --CRedge 200 --VRedge 100
python scripts/regionPlotter.py --cfg config/rectConfig.cfg --CRedge 200 --VRedge 125

python scripts/regionPlotter.py --cfg config/rectConfig.cfg --CRedge -1 --VRedge 50
rm mAP/input/detection-results/*.txt
rm mAP/input/ground-truth/*.txt
python3 itsd_map_calc.py
echo "map file run completed"
cd mAP
cd scripts/extra
python3 intersect-gt-and-dr.py
cd ..
cd ..
python3 main.py
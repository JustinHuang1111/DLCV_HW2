python inference.py -model1 $1 -model2 $2 -seed $3
python ../utility/face_recog.py --image_dir ./output
python -m pytorch_fid ./output ../hw2_data/face/val/

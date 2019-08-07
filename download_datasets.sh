wget http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/data/label_colors.txt

wget http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/data/LabeledApproved_full.zip
unzip -d train_labels LabeledApproved_full.zip

wget http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/files/701_StillsRaw_full.zip
unzip 701_StillsRaw_full.zip
mv 701_StillsRaw_full train_inputs
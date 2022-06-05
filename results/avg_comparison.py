import csv
import numpy as np
import os
import pdb

feas_ns_oa=0.
feas_ns=0.
feas_ol_oa=0.
feas_ol=0.

haus_ns_oa=0.
haus_ns=0.
haus_ol_oa=0.
haus_ol=0.

dmin_ns_oa=0.
dmin_ns=0.
dmin_ol_oa=0.
dmin_ol=0.
# pdb.set_trace()
csv_path = os.path.join(os.path.abspath(__file__).split('avg_')[0], 'df_norm.csv')

with open(csv_path, mode='r') as csv_file:
	csv_reader = csv.DictReader(csv_file)
	ctr=0
	# pdb.set_trace()
	for row in csv_reader:
		if ctr > 0:
			# pdb.set_trace()
			if row["policy"].split("smpc_")[-1]=="no_switch":
				feas_ns+=float(row["feasibility_percent"])/1.
				haus_ns+=float(row["hausdorff_dist_notv"])/1.
				dmin_ns+=float(row["dmin_TV"])/10.
			elif row["policy"].split("smpc_")[-1]=="no_switch_OAinner":
				feas_ns_oa+=float(row["feasibility_percent"])/1.
				haus_ns_oa+=float(row["hausdorff_dist_notv"])/1.
				dmin_ns_oa+=float(row["dmin_TV"])/1.
			elif row["policy"].split("smpc_")[-1]=="no_switch_obca":
				feas_ol+=float(row["feasibility_percent"])/1.
				haus_ol+=float(row["hausdorff_dist_notv"])/1.
				dmin_ol+=float(row["dmin_TV"])/1.
			# elif row["policy"].split("smpc_")[-1]=="open_loop_OAinner":
			# 	feas_ol_oa+=float(row["feasibility_percent"])/10.
			# 	haus_ol_oa+=float(row["hausdorff_dist_notv"])/10.
			# 	dmin_ol_oa+=float(row["dmin_TV"])/10.
		ctr+=1


print(["NS:", feas_ns, haus_ns, dmin_ns])
print(["NS_OA:", feas_ns_oa, haus_ns_oa, dmin_ns_oa])
print(["OL:", feas_ol, haus_ol, dmin_ol])
print(["OL_OA:", feas_ol_oa, haus_ol_oa, dmin_ol_oa])

import pandas as pd
from torch.utils.tensorboard import SummaryWriter

csv_path = "eval_best.csv"
log_dir = "./tb_logs/eval_csv"
df = pd.read_csv(csv_path)  #df = dataframe


df["ias_knots"] = df["ias_ms"] * 1.944

# use steps as x axis, if steps column doens't exist, use the numbers
steps = df["step"] if "step" in df.columns else range(len(df))

writer = SummaryWriter(log_dir=log_dir) #open the writer

for col in df.columns:
    if col == "step":
        continue        #we don't want step to plot against itself
    values = pd.to_numeric(df[col], errors="coerce") #numericalize everything in col, if non number exist, turn into NaN
    if values.isna().all():
        continue            #Non number columns
    for step, val in zip(steps, values):
        if pd.isna(val):    #if non number or missing
            continue
        writer.add_scalar(f"flight/{col}", float(val), int(step))
writer.close()

'''
python csv_plot.py
tensorboard --logdir=./tb_logs/
tensorboard --logdir=./tb_logs/ --samples_per_plugin=scalars=0 (if want to see every data)
'''
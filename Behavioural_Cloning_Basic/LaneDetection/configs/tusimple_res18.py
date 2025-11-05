# ==========================
# Ultra-Fast-Lane-Detection-v2 – Config für TuSimple ResNet-18 (Inference-fähig)
# ==========================

dataset = 'Tusimple'
data_root = ''           # bleibt leer (wird nicht benötigt für Demo)
test_model = ''          # wird per CLI (--test_model) übergeben
test_work_dir = ''

# --- Modellparameter ---
backbone = '18'
griding_num = 100
num_lanes = 4
use_aux = False
num_row = 56
num_col = 41
num_cell_row = 100
num_cell_col = 100


# --- Trainingsparameter (irrelevant für Demo, aber müssen definiert bleiben) ---
epoch = 100
batch_size = 32
optimizer = 'SGD'
learning_rate = 0.05
weight_decay = 0.0001
momentum = 0.9
scheduler = 'multi'
steps = [50, 75]
gamma = 0.1
warmup = 'linear'
warmup_iters = 100
note = ''
log_path = ''
finetune = None
resume = None
auto_backup = True

# --- Bildgröße (wie Training) ---
train_width = 800
train_height = 320
crop_ratio = 1.0  # ergänzen! wichtig für Resize im Demo-Script

# --- Loss-Parameter (irrelevant im Demo) ---
sim_loss_w = 0.0
shp_loss_w = 0.0
var_loss_power = 2.0
mean_loss_w = 0.05
fc_norm = False
soft_loss = True
cls_loss_col_w = 1.0
cls_ext_col_w = 1.0
mean_loss_col_w = 0.05
eval_mode = 'normal'

# ==========================
# Anchors (aus Original-UFLD TuSimple-Konfig)
# ==========================
anchor_base_height = 288 

row_anchor = [64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120,
              124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172,
              176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216, 220, 224,
              228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268, 272, 276, 280, 284]

row_anchor = [r / 288.0 for r in row_anchor]

col_anchor = [i / 40 for i in range(41)]
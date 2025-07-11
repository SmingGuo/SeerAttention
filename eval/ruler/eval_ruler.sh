# model="/home/v-shumingguo/gsm_blob/models/SeerAttention-Llama-3.1-8B-AttnGates"
# model="SeerAttention/SeerAttention-Qwen2.5-7B-AttnGates"
model="/home/v-shumingguo/gsm_blob/distilled_model_gates/Llama-3.1-8B-Instruct-seer/redpajama_Qproj_Kmaxminavg_bs16_steps500_gdim128_block64_wd0.0_lr1e-3_slice0.0_qknormfalse"
# change model to the path of your model if needed
basedir=./results/llama
threshold=5e-4
export CUDA_VISIBLE_DEVICES=0
export PROFILE_FILE=${basedir}/${threshold}.txt # Comment this line to disable profiling
# bash run_seer.sh \
#     $model \
#     SeerAttn \
#     $basedir \
#     $threshold

## Get profiled sparsity
python average_sparsity.py --file $PROFILE_FILE
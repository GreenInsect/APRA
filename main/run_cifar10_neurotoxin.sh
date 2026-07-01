#!/bin/bash

# ==========================================
# 实验配置中心
# ==========================================
GPU_ID=0
PYTHON_EXEC="python"
BASE_CONFIG_DIR="yamls"
TEMP_DIR="temp"

# 在这里统一控制实验的总轮数（不再硬编码）
EPOCHS=500
MIA_ANALYSIS="false"

# AggRecorder switch. These values are written into each copied config by sed.
# Set AGG_RECORD_ENABLED=true when aggregation defense records are needed.
AGG_RECORD_ENABLED="${AGG_RECORD_ENABLED:-true}"
AGG_RECORD_TENSOR_MODE="${AGG_RECORD_TENSOR_MODE:-stats}" # stats | full
AGG_RECORD_UPDATE_SUMMARY="${AGG_RECORD_UPDATE_SUMMARY:-true}"
AGG_RECORD_DIR="${AGG_RECORD_DIR:-}"

RECORD_LOCAL_POST_TRAIN="${RECORD_LOCAL_POST_TRAIN:-false}"

MIA_ENHANCED="${MIA_ENHANCED:-false}"
MIA_PULL_WEIGHT="${MIA_PULL_WEIGHT:-0.1}"
MIA_CLEAN_RETRAIN="${MIA_CLEAN_RETRAIN:-5}"

MIA_CLASS_METHOD="${MIA_CLASS_METHOD:-nobackdoor_random}"

MALICIOUS_ENHANCED="${MALICIOUS_ENHANCED:-false}"
MALICIOUS_PULL_WEIGHT="${MALICIOUS_PULL_WEIGHT:-0.1}"
MALICIOUS_CLEAN_RETRAIN="${MALICIOUS_CLEAN_RETRAIN:-5}"
MIALICIOUS_ENHANCED_VERSION="${MALICIOUS_ENHANCED_VERSION:-6}"

NUM_SAMPLED_PARTICIPANTS="${NUM_SAMPLED_PARTICIPANTS:-10}"

export CUDA_MODULE_LOADING=LAZY

# 1. 简单注释：将作为实验记录目录的后缀（建议使用英文、数字、下划线，避免空格和特殊字符）
EXP_NOTE_SHORT="6-29_APRA"

# 2. 详细注释：Markdown 格式，会自动写入该实验目录下的 README.md
EXP_NOTE_DETAIL="# $EXP_NOTE_SHORT
实验时间 $(date '+%Y-%m-%d %H:%M:%S')
"
# --------------------

# 创建实验记录根目录（根据简单注释动态调整）
if [ -n "$EXP_NOTE_SHORT" ]; then
    RESULT_BASE="experiment_records_${EXP_NOTE_SHORT}"
else
    RESULT_BASE="experiment_records"
fi

mkdir -p "$TEMP_DIR"
mkdir -p "$RESULT_BASE"

# 自动生成或追加 Markdown 详细注释到 README.md
README_FILE="${RESULT_BASE}/README.md"
if [ -n "$EXP_NOTE_DETAIL" ]; then
    # 如果文件不存在，写入标题和时间
    if [ ! -f "$README_FILE" ]; then
        echo -e "# 实验总体说明 $EXP_NOTE_SHORT \n\n> 创建时间: $(date '+%Y-%m-%d %H:%M:%S')\n" > "$README_FILE"
    fi
    # 追加当前的详细 Markdown 注释
    cat << EOF >> "$README_FILE"

---
## 实验批次详细日志 ($(date '+%Y-%m-%d %H:%M:%S'))
$EXP_NOTE_DETAIL

EOF
    echo ">>>> 详细注释已成功写入: $README_FILE"
fi

DATASETS=( "cifar10")
AGG_METHODS=( "apra" "avg" "clip" "foolsgold" "rflbat" "deepsight")
# AGG_METHODS=( "deepsight")
MIA_OPTIONS=("false")
ATTACKER_METHODS=( "neurotoxin")

START_TOTAL=$SECONDS

# ==========================================
# 辅助工具函数
# ==========================================

format_time() {
    local T=$1
    local H=$((T/3600))
    local M=$((T%3600/60))
    local S=$((T%60))
    printf "%02dh %02dm %02ds" $H $M $S
}

send_notification() {
    local status=$1
    local msg_body=$2
    local duration_str=$3
    local subject="[${status} - ${msg_body}]-$(date '+%H:%M')"
    
    local full_body="--- 实验通知 ---       \n     "
    full_body="${full_body}当前状态: ${status}\n"
    full_body="${full_body}执行时间: $(date '+%Y-%m-%d %H:%M:%S')     \n      "
    full_body="${full_body}阶段耗时: ${duration_str}         \n       "
    full_body="${full_body}详情: ${msg_body}        \n        "
    
    python3 send_mail.py "$subject" "$full_body"
}

# ==========================================
# 实验核心执行函数
# ==========================================
run_experiment() {
    local dataset=$1
    local agg=$2
    local mia=$3
    local attacker=$4
    
    local start_single=$SECONDS
    # 生成带时间的唯一 ID
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local exp_id="${dataset}_${agg}_${attacker}_MIA_${mia}_${timestamp}"
    
    # 为本次实验创建独立痕迹目录
    local exp_trace_dir="${RESULT_BASE}/${exp_id}"
    mkdir -p "$exp_trace_dir"

    local base_config="${BASE_CONFIG_DIR}/common.yaml"
    if [ ! -f "$base_config" ]; then
        send_notification "配置错误" "基础文件 $base_config 不存在" "00h 00m 00s"
        return 1
    fi

    # 1. 复制带时间戳的配置文件痕迹
    local saved_config="${exp_trace_dir}/config.yaml"
    cp "$base_config" "$saved_config"

    # 修改备份配置中的参数（通过正则精准覆盖，防止误伤内循环参数）
    sed -i "s@agg_method:.*@agg_method: $agg@" "$saved_config"
    sed -i "s@mia:.*@mia: $mia@" "$saved_config"
    sed -i "s@attacker_method:.*@attacker_method: $attacker@" "$saved_config"
    sed -i "s@dataset:.*@dataset: $dataset@" "$saved_config"
    sed -i "s@^\s*epochs:.*@epochs: $EPOCHS@" "$saved_config"
    
    sed -i "s@^\s*mia_analysis:.*@mia_analysis: $MIA_ANALYSIS@" "$saved_config"

    sed -i "s@^\s*agg_record_enabled:.*@agg_record_enabled: $AGG_RECORD_ENABLED@" "$saved_config"
    sed -i "s@^\s*agg_record_tensor_mode:.*@agg_record_tensor_mode: $AGG_RECORD_TENSOR_MODE@" "$saved_config"
    sed -i "s@^\s*agg_record_update_summary:.*@agg_record_update_summary: $AGG_RECORD_UPDATE_SUMMARY@" "$saved_config"
    sed -i "s@^\s*agg_record_dir:.*@agg_record_dir: \"$AGG_RECORD_DIR\"@" "$saved_config"

    sed -i "s@^\s*record_local_post_train:.*@record_local_post_train: $RECORD_LOCAL_POST_TRAIN@" "$saved_config"

    sed -i "s@^\s*mia_enhanced:.*@mia_enhanced: $MIA_ENHANCED@" "$saved_config"
    sed -i "s@^\s*mia_pull_weight:.*@mia_pull_weight: $MIA_PULL_WEIGHT@" "$saved_config"
    sed -i "s@^\s*mia_clean_retrain:.*@mia_clean_retrain: $MIA_CLEAN_RETRAIN@" "$saved_config"

    sed -i "s@^\s*mia_class_method:.*@mia_class_method: \"$MIA_CLASS_METHOD\"@" "$saved_config"

    sed -i "s@^\s*malicious_enhanced:.*@malicious_enhanced: $MALICIOUS_ENHANCED@" "$saved_config"
    sed -i "s@^\s*malicious_pull_weight:.*@malicious_pull_weight: $MALICIOUS_PULL_WEIGHT@" "$saved_config"
    sed -i "s@^\s*malicious_clean_retrain:.*@malicious_clean_retrain: $MALICIOUS_CLEAN_RETRAIN@" "$saved_config"
    sed -i "s@^\s*malicious_enhanced_version:.*@malicious_enhanced_version: $MALICIOUS_ENHANCED_VERSION@" "$saved_config"

    sed -i "s@^\s*num_sampled_participants:.*@num_sampled_participants: $NUM_SAMPLED_PARTICIPANTS@" "$saved_config"


    if [ "$dataset" = "cifar10" ]; then
        local_scale="2"
        sed -i "s@fl_weight_scale:.*@fl_weight_scale: $local_scale@" "$saved_config"    
    elif [ "$dataset" = "cifar100" ]; then
        local_scale="5"
        sed -i "s@fl_weight_scale:.*@fl_weight_scale: $local_scale@" "$saved_config"
    elif [ "$dataset" = "mnist" ]; then
        local_scale="1.5"
        sed -i "s@fl_weight_scale:.*@fl_weight_scale: $local_scale@" "$saved_config"    
    elif [ "$dataset" = "fashion-mnist" ]; then
        local_scale="5"
        sed -i "s@fl_weight_scale:.*@fl_weight_scale: $local_scale@" "$saved_config"   
    else
        echo "[WARNING] 未知数据集: $dataset, 保持原有 fl_weight_scale 不变或请手动指定。"
    fi    

    # 2. 定义 Log 文件路径
    local run_log="${exp_trace_dir}/run.log"

    echo ">>>> [Running] ID: $exp_id"
    echo ">>>> Log stored in: $run_log"
    
    # 3. 核心优雅投递：干净地传递物理卡隔离环境，并显式传入内部对齐的局部逻辑卡号（gpu=0）
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    $PYTHON_EXEC clean_common.py \
        --params "$saved_config" \
        --comment "$EXP_NOTE_SHORT" \
        --gpu 0 2>&1 | tee "$run_log"
    
    local exit_code=${PIPESTATUS[0]} # 获取 tee 前面那个命令的退出码

    # 计算耗时
    local duration=$((SECONDS - start_single))
    local duration_fmt=$(format_time $duration)

    if [ $exit_code -eq 0 ]; then
        echo ">>>> [Success] Trace: $exp_trace_dir"
        send_notification "单次成功" "$EXP_NOTE_SHORT 实验 $exp_id 已完成。轨迹已存入 $exp_trace_dir" "$duration_fmt"
    else
        echo ">>>> [Failed] Check Log: $run_log"
        # 提取报错给邮件
        local error_detail=$(tail -n 100 "$run_log")
        send_notification "单次失败" "$EXP_NOTE_SHORT 实验 $exp_id 失败。\n轨迹目录: $exp_trace_dir" "\n\n[最近日志预览]:\n$error_detail \n $duration_fmt"
        exit 1 
    fi
}

# ==========================================
# 主循环逻辑
# ==========================================

for ds in "${DATASETS[@]}"; do
    START_DS=$SECONDS
    for atk in "${ATTACKER_METHODS[@]}"; do
        START_ATK=$SECONDS
        for mia_val in "${MIA_OPTIONS[@]}"; do
            for agg_val in "${AGG_METHODS[@]}"; do
                run_experiment "$ds" "$agg_val" "$mia_val" "$atk"
            done
        done
        FMT_ATK=$(format_time $((SECONDS - START_ATK)))
        send_notification "攻击级完成" "$EXP_NOTE_SHORT Dataset: $ds, Atk: $atk 组结束。" "$FMT_ATK"
    done
    FMT_DS=$(format_time $((SECONDS - START_DS)))
    send_notification "数据集级完成" "$EXP_NOTE_SHORT Dataset: $ds 组结束。" "$FMT_DS"
done

FMT_TOTAL=$(format_time $((SECONDS - START_TOTAL)))
echo "全部任务执行完毕！总耗时: $FMT_TOTAL"
send_notification "总任务全部完成" "$EXP_NOTE_SHORT 所有实验序列已结束。" "$FMT_TOTAL"
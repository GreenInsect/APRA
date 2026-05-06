#!/bin/bash

# 1. 定义实验参数矩阵
# "apra" "avg" "clip" "deepsight" "foolsgold" "rflbat"
DEFENSES=("apra" "avg" "clip" "deepsight" "foolsgold" "rflbat")
# ATTACKS=("a3fl" "modelreplace" "sin-adv"  )
# "a3fl" "modelreplace" "sin-adv" "neurotoxin" "reba"
ATTACKS=("reba" )

CONFIG_PATH="yamls/cifar10_apra.yaml"

# 记录开始总时间
START_TIME=$(date +%s)

# 2. 嵌套循环进行交叉验证
for DEFENSE in "${DEFENSES[@]}"
do
    for ATTACK in "${ATTACKS[@]}"
    do
        echo "================================================"
        echo "正在开始交叉实验:"
        echo "防御方法 (Defense):  $DEFENSE"
        echo "攻击方法 (Attacker): $ATTACK"
        echo "------------------------------------------------"

        # 3. 使用 sed 动态修改 YAML 文件字段
        # 匹配以 agg_method: 开头的行并替换
        sed -i "s/^agg_method:.*/agg_method: $DEFENSE/" "$CONFIG_PATH"
        # 匹配以 attacker_method: 开头的行并替换
        sed -i "s/^attacker_method:.*/attacker_method: $ATTACK/" "$CONFIG_PATH"

        # 4. 运行实验脚本
        # 假设你的入口文件是 main.py，如果是之前的 clean_cifar10.py 请自行更名
        python main.py --params "$CONFIG_PATH"

        # 5. 检查运行状态
        if [ $? -eq 0 ]; then
            echo "实验成功: [Defense: $DEFENSE, Attack: $ATTACK]"
            # 发送单次实验完成通知（可选，若嫌邮件太多可以注释掉）
            python3 send_mail.py "实验成功: $DEFENSE vs $ATTACK" "配置已生效并运行完成。"
        else
            echo "错误: [Defense: $DEFENSE, Attack: $ATTACK] 运行失败！"
            python3 send_mail.py "实验失败警告" "警告: 攻击 $ATTACK 与防御 $DEFENSE 的组合运行出错，脚本已停止。"
            exit 1
        fi
    done
done

# 6. 全部完成通知
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "所有交叉验证实验全部完成! 总耗时: $((DURATION / 60)) 分钟。"
python3 send_mail.py "CIFAR10 交叉实验全部完成!" "所有配置 ($(( ${#DEFENSES[@]} * ${#ATTACKS[@]} )) 组) 已跑完。完成时间: $(date '+%Y-%m-%d %H:%M:%S')"

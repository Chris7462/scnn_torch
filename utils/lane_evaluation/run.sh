#!/bin/bash

# Auto-detect paths
bin_dir=$(cd "$(dirname "$0")" && pwd)
root=$(cd "${bin_dir}/../.." && pwd)
data_dir=${root}/data/CULane/
detect_dir=${root}/outputs/predictions/
eval_dir=${root}/outputs/evaluate/

# Evaluation parameters
w_lane=30
iou=0.5  # Set iou to 0.3 or 0.5
im_w=1640
im_h=590
frame=1

# Create evaluation output directory
mkdir -p ${eval_dir}

# Check if evaluator binary exists
if [ ! -f "${bin_dir}/evaluate" ]; then
    echo "Error: evaluate binary not found at ${bin_dir}/evaluate"
    echo "Please build it first:"
    echo "  cd ${bin_dir}"
    echo "  mkdir -p build && cd build"
    echo "  cmake .."
    echo "  make"
    exit 1
fi

# Check if predictions directory exists
if [ ! -d "${detect_dir}" ]; then
    echo "Error: predictions directory not found at ${detect_dir}"
    echo "Please run test.py first to generate predictions."
    exit 1
fi

echo "============================================"
echo "CULane Evaluation"
echo "============================================"
echo "Project root: ${root}"
echo "Data dir:     ${data_dir}"
echo "Predictions:  ${detect_dir}"
echo "Output:       ${eval_dir}"
echo "IoU:          ${iou}"
echo "Lane width:   ${w_lane}"
echo "============================================"

# Test split lists
list0=${data_dir}list/test_split/test0_normal.txt
list1=${data_dir}list/test_split/test1_crowd.txt
list2=${data_dir}list/test_split/test2_hlight.txt
list3=${data_dir}list/test_split/test3_shadow.txt
list4=${data_dir}list/test_split/test4_noline.txt
list5=${data_dir}list/test_split/test5_arrow.txt
list6=${data_dir}list/test_split/test6_curve.txt
list7=${data_dir}list/test_split/test7_cross.txt
list8=${data_dir}list/test_split/test8_night.txt

# Output files
out0=${eval_dir}out0_normal.txt
out1=${eval_dir}out1_crowd.txt
out2=${eval_dir}out2_hlight.txt
out3=${eval_dir}out3_shadow.txt
out4=${eval_dir}out4_noline.txt
out5=${eval_dir}out5_arrow.txt
out6=${eval_dir}out6_curve.txt
out7=${eval_dir}out7_cross.txt
out8=${eval_dir}out8_night.txt

# Run evaluation for each category
echo "Evaluating normal..."
${bin_dir}/evaluate -a $data_dir -d $detect_dir -i $data_dir -l $list0 -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out0

echo "Evaluating crowd..."
${bin_dir}/evaluate -a $data_dir -d $detect_dir -i $data_dir -l $list1 -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out1

echo "Evaluating hlight..."
${bin_dir}/evaluate -a $data_dir -d $detect_dir -i $data_dir -l $list2 -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out2

echo "Evaluating shadow..."
${bin_dir}/evaluate -a $data_dir -d $detect_dir -i $data_dir -l $list3 -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out3

echo "Evaluating noline..."
${bin_dir}/evaluate -a $data_dir -d $detect_dir -i $data_dir -l $list4 -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out4

echo "Evaluating arrow..."
${bin_dir}/evaluate -a $data_dir -d $detect_dir -i $data_dir -l $list5 -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out5

echo "Evaluating curve..."
${bin_dir}/evaluate -a $data_dir -d $detect_dir -i $data_dir -l $list6 -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out6

echo "Evaluating cross..."
${bin_dir}/evaluate -a $data_dir -d $detect_dir -i $data_dir -l $list7 -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out7

echo "Evaluating night..."
${bin_dir}/evaluate -a $data_dir -d $detect_dir -i $data_dir -l $list8 -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out8

# Combine all results
cat ${eval_dir}out*.txt > ${eval_dir}summary_iou${iou}.txt

echo "============================================"
echo "Evaluation complete!"
echo "Results saved to: ${eval_dir}"
echo "Summary: ${eval_dir}summary_iou${iou}.txt"
echo "============================================"

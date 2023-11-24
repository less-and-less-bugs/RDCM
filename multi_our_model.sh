#lr=0.001
phase='train'
#temperature=3
tag='kernel'
#lm1_list=("0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1." "1.5" "2." "4." '5.')
#lm1_list=("0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8")
#lm1_list=("0.1")
#lm2_list=("0.1"  "0.3"  "0.5"  "0.7"  "0.9" "1." "1.5" "2." "5." "10." '4.')
#lm2_list=("0.1"  "0.3"  "0.5"  "0.7")
#lm1_list=('0.1' '0.2')
#lm2_list=('0.1' '0.2')
#for lam1 in ${lm1_list[*]}
#do
#  for lam2 in ${lm2_list[*]}
#  do
##    CUDA_VISIBLE_DEVICES=0 python drive_ourmodel.py --data='Pheme' --tag=$tag  --da='True'  --lr=0.001 --log='dgintra'   --phase=$phase --epochs=20  --max_iter=10 --lambda1=$lam1 --tsigma="2#4#8#16" --vsigma="2#4#8#16" --lambda2=$lam2 --temperature=0.5 --threshold=0.5 --ctsize=64 &
##    CUDA_VISIBLE_DEVICES=1 python drive_ourmodel.py --data='Pheme' --tag=$tag  --da='False'  --lr=0.001 --log='dgintra'   --phase=$phase --epochs=20  --max_iter=10 --lambda1=$lam1 --tsigma="2#4#8#16" --vsigma="2#4#8#16" --lambda2=$lam2 --temperature=0.5 --threshold=0.5 --ctsize=64 &
##    CUDA_VISIBLE_DEVICES=1 python drive_ourmodel.py --data='Pheme' --tag=$tag   --da='True' --lr=0.0008 --log='dgintra'  --phase=$phase --epochs=20  --max_iter=10 --lambda1=$lam1 --tsigma="2#4#8#16" --vsigma="2#4#8#16"  --lambda2=$lam2 --temperature=0.5  --threshold=0.5 --ctsize=64 &
##    CUDA_VISIBLE_DEVICES=1 python drive_ourmodel.py --data='Pheme'  --tag=$tag  --da='True' --lr=0.0008 --log='dgintra'  --phase=$phase --epochs=20  --max_iter=10 --lambda1=$lam1 --tsigma="2#4#8#16" --vsigma="2#4#8#16"  --lambda2=$lam2 --temperature=0.5 --threshold=0.5 --ctsize=64 &
##    CUDA_VISIBLE_DEVICES=1 python drive_ourmodel.py --data='Pheme' --tag=$tag   --da='False' --lr=0.001 --log='dgintra'  --phase=$phase --epochs=20  --max_iter=10 --lambda1=$lam1 --tsigma="2#10#30#40" --vsigma="2#10#30#40"  --lambda2=$lam2 --temperature=0.5  --threshold=0.5 --ctsize=64
##    CUDA_VISIBLE_DEVICES=2 python drive_ourmodel.py --data='Pheme'  --tag=$tag  --da='True' --lr=0.0008 --log='dgintra'  --phase=$phase --epochs=20  --max_iter=10 --lambda1=$lam1 --tsigma="2#10#30#40" --vsigma="2#10#30#40" --lambda2=$lam2 --temperature=0.5 --threshold=0.5 --ctsize=64
#  done
#done


#CUDA_VISIBLE_DEVICES=0 python drive_ourmodel.py --data='Pheme' --tag=$tag  --da='False'  --lr=0.001 --log='vismmdj'   --phase=$phase --epochs=30  --max_iter=10 --lambda1=1 --tsigma="2#4#8#12" --vsigma="2#4#8#12"  --lambda2=0 --temperature=0.5 --threshold=0.5 --ctsize=64 &
#CUDA_VISIBLE_DEVICES=0 python drive_ourmodel.py --data='Twitter'  --tag=$tag  --da='True' --lr=0.001 --log='dgintra'   --phase=$phase --epochs=30  --max_iter=10 --lambda1=1 --tsigma="2#4#8#12" --vsigma="2#4#8#12"  --lambda2=0.8 --temperature=0.5 --threshold=0.5 --ctsize=64 &
#CUDA_VISIBLE_DEVICES=0 python drive_ourmodel.py --data='Twitter'  --tag=$tag   --da='True' --lr=0.001 --log='dgintra'  --phase=$phase --epochs=30  --max_iter=10 --lambda1=1 --tsigma="2#4#8#12"    --vsigma="2#4#8#12"   --lambda2=1 --temperature=0.5  --threshold=0.5 --ctsize=64 &
#CUDA_VISIBLE_DEVICES=1 python drive_ourmodel.py --data='Twitter'  --tag=$tag  --da='True' --lr=0.001 --log='dgintra'  --phase=$phase --epochs=30  --max_iter=10 --lambda1=1 --tsigma="2#4#8#12"  --vsigma="2#4#8#12"    --lambda2=5 --temperature=0.5 --threshold=0.5 --ctsize=64 &
#CUDA_VISIBLE_DEVICES=1 python drive_ourmodel.py --data='Twitter'  --tag=$tag   --da='True' --lr=0.001 --log='dgintra'  --phase=$phase --epochs=30  --max_iter=10 --lambda1=1 --tsigma="2#4#8#16"  --vsigma="2#4#8#16"  --lambda2=3 --temperature=0.5  --threshold=0.5 --ctsize=64 &
#CUDA_VISIBLE_DEVICES=1 python drive_ourmodel.py --data='Twitter'  --tag=$tag  --da='True' --lr=0.001 --log='dgintra'  --phase=$phase --epochs=30  --max_iter=10 --lambda1=1 --tsigma="2#4#8#16"   --vsigma="2#4#8#16"   --lambda2=0.95 --temperature=0.5 --threshold=0.5 --ctsize=64 &
#CUDA_VISIBLE_DEVICES=2 python drive_ourmodel.py --data='Twitter'  --tag=$tag   --da='True' --lr=0.001 --log='dgintra'  --phase=$phase --epochs=30  --max_iter=10 --lambda1=1 --tsigma="2#4#8#16"   --vsigma="2#4#8#16"  --lambda2=1.2 --temperature=0.5  --threshold=0.5 --ctsize=64 &
#CUDA_VISIBLE_DEVICES=2 python drive_ourmodel.py --data='Twitter'   --tag=$tag  --da='True' --lr=0.001 --log='dgintra'  --phase=$phase --epochs=30  --max_iter=10 --lambda1=1 --tsigma="2#4#8#16"    --vsigma="2#4#8#16"   --lambda2=1.5 --temperature=0.5 --threshold=0.5 --ctsize=64 &
#CUDA_VISIBLE_DEVICES=3 python drive_ourmodel.py --data='Twitter'  --tag=$tag   --da='True' --lr=0.001 --log='dgintra'  --phase=$phase --epochs=30  --max_iter=10 --lambda1=1 --tsigma="2#4#8#16"  --vsigma="2#4#8#16"  --lambda2=0.9 --temperature=0.5  --threshold=0.5 --ctsize=64 &
#CUDA_VISIBLE_DEVICES=3 python drive_ourmodel.py --data='Twitter'   --tag=$tag  --da='True' --lr=0.001 --log='dgintra'  --phase=$phase --epochs=30  --max_iter=10 --lambda1=1 --tsigma="2#4#8#16"    --vsigma="2#4#8#16"    --lambda2=2 --temperature=0.5 --threshold=0.5 --ctsize=64 &




#CUDA_VISIBLE_DEVICES=1 python drive_ourmodel.py --data='Pheme'  --tag=$tag   --da='False' --lr=0.001 --log='dgintra'  --phase=$phase --epochs=20  --max_iter=10 --lambda1=1  --tsigma="2#4#8#16" --vsigma="2#4#8#16"  --lambda2=0 --temperature=0.5 --threshold=0.5 --ctsize=64 &
#CUDA_VISIBLE_DEVICES=2 python drive_ourmodel.py --data='Pheme'  --tag=$tag   --da='False' --lr=0.001 --log='dgintra'  --phase=$phase --epochs=20  --max_iter=10 --lambda1=1 --tsigma="2#10#20#30#40" --vsigma="2#5#10#15#20"--lambda2=0 --temperature=0.5 --threshold=0.5 --ctsize=64 &
#CUDA_VISIBLE_DEVICES=2 python drive_ourmodel.py --data='Pheme'  --tag=$tag   --da='False' --lr=0.001 --log='dgintra'  --phase=$phase --epochs=20  --max_iter=10 --lambda1=1 --tsigma="2#10#20#30#40" --vsigma="2#5#8#11#14" --lambda2=0  --temperature=0.5  --threshold=0.5 --ctsize=64 &
#CUDA_VISIBLE_DEVICES=3 python drive_ourmodel.py --data='Pheme'  --tag=$tag   --da='False' --lr=0.001 --log='dgintra'  --phase=$phase --epochs=20  --max_iter=10 --lambda1=0.5 --tsigma="2#4#8#16" --vsigma="2#4#8#16" --lambda2=0.1  --temperature=0.5 --threshold=0.5 --ctsize=64 &
#CUDA_VISIBLE_DEVICES=3 python drive_ourmodel.py --data='Pheme'  --tag=$tag   --da='False' --lr=0.001 --log='dgintra'  --phase=$phase --epochs=20  --max_iter=10 --lambda1=0.5 --tsigma="2#4#8#16" --vsigma="2#4#8#16"  --lambda2=1 --temperature=0.5  --threshold=0.5 --ctsize=64  &
#CUDA_VISIBLE_DEVICES=3 python drive_ourmodel.py --data='Pheme'  --tag=$tag   --da='False' --lr=0.001 --log='dgintra'  --phase=$phase --epochs=20  --max_iter=10 --lambda1=0 --tsigma --vsigma  --lambda2=0 --temperature=$temperature --threshold=0.9 --ctsize=64  &

#CUDA_VISIBLE_DEVICES=1 python drive_ourmodel.py --data='Twitter'  --da='False' --tag='-0.001'  --lr=0.001 --log='dgintra'   --phase=$phase --epochs=30  --max_iter=10 --lambda1=0.001 --lambda2=0.01  &
#CUDA_VISIBLE_DEVICES=1 python drive_ourmodel.py --data='Twitter'  --da='False' --tag='-0.001'  --lr=0.001--log='dgintra'  --phase=$phase --epochs=30  --max_iter=10 --lambda1=0.001 --lambda2=0.01  &
#CUDA_VISIBLE_DEVICES=2 python drive_ourmodel.py --data='Twitter'  --da='False' --tag='-0.001'  --lr=0.001 --log='dgintra'  --phase=$phase --epochs=30  --max_iter=10 --lambda1=0.001  --lambda2=0.1  &
#CUDA_VISIBLE_DEVICES=2 python drive_ourmodel.py --data='Twitter'  --da='False' --tag='-0.0001'  --lr=0.001 --log='dgintra'  --phase=$phase --epochs=30  --max_iter=10 --lambda1=0.0001 --lambda2=0.01  &
#CUDA_VISIBLE_DEVICES=3 python drive_ourmodel.py --data='Twitter'  --da='False' --tag='-0.0001'  --lr=0.001--log='dgintra' --phase=$phase --epochs=30  --max_iter=10 --lambda1=0.0001 --lambda2=0.01  &
#CUDA_VISIBLE_DEVICES=3 python drive_ourmodel.py --data='Twitter'  --da='False' --tag='-0.0001'  --lr=0.001 --log='dgintra'  --phase=$phase --epochs=30  --max_iter=10 --lambda1=0.0001 --lambda2=0.1  &

# response (a)sh
tag='temperature'


# to save as checkpoint
#CUDA_VISIBLE_DEVICES=2 python drive_ourmodel.py --data='Twitter'  --tag=$tag  --da='False' --lr=0.001 --log='orgtwitter'   --phase='test' --epochs=30  --max_iter=10 --lambda1=0 --tsigma="2#4#8#12" --vsigma="2#4#8#12"  --lambda2=0 --temperature=0.5 --threshold=0.5 --ctsize=64 &
CUDA_VISIBLE_DEVICES=2 python drive_ourmodel.py --data='Twitter'  --tag=$tag  --da='False' --lr=0.001 --log='dgtwitter'   --phase='test' --epochs=30  --max_iter=10 --lambda1=1 --tsigma="2#4#8#12" --vsigma="2#4#8#12"  --lambda2=0.9 --temperature=0.5 --threshold=0.5 --ctsize=64 &
#CUDA_VISIBLE_DEVICES=1 python drive_ourmodel.py --data='Twitter'  --tag=$tag  --da='True' --lr=0.001 --log='datwitter'   --phase=$phase --epochs=30  --max_iter=10 --lambda1=1 --tsigma="2#4#8#12" --vsigma="2#4#8#12"  --lambda2=1 --temperature=0.5 --threshold=0.5 --ctsize=64 &
#
#CUDA_VISIBLE_DEVICES=2 python drive_ourmodel.py --data='Pheme' --tag=$tag   --da='True' --lr=0.001 --log='dapheme'  --phase=$phase --epochs=20  --max_iter=10 --lambda1=0.001 --tsigma="2#10#30#40" --vsigma="2#10#30#40"  --lambda2=0.5 --temperature=0.5  --threshold=0.5 --ctsize=64 &
#CUDA_VISIBLE_DEVICES=2 python drive_ourmodel.py --data='Pheme' --tag=$tag   --da='False' --lr=0.001 --log='dgpheme'  --phase='test' --epochs=20  --max_iter=10 --lambda1=0.1 --tsigma="2#10#30#40" --vsigma="2#10#30#40"  --lambda2=0.5 --temperature=0.5  --threshold=0.5 --ctsize=64 &
#CUDA_VISIBLE_DEVICES=3 python drive_ourmodel.py --data='Pheme' --tag=$tag   --da='False' --lr=0.001 --log='orgpheme'  --phase='test' --epochs=20  --max_iter=10 --lambda1=0 --tsigma="2#10#30#40" --vsigma="2#10#30#40"  --lambda2=0 --temperature=0.5  --threshold=0.5 --ctsize=64 &

#CUDA_VISIBLE_DEVICES=2 python drive_ourmodel.py  --data='Pheme' --tag=$tag   --da='True' --lr=0.001 --log='dapheme-0.5-kernel'  --phase='train' --epochs=20  --max_iter=10 --lambda1=0.1 --tsigma="2#10#30#40" --vsigma="2#10#30#40"  --lambda2=0.5 --temperature=0.5  --threshold=0.5 --ctsize=64 &
#CUDA_VISIBLE_DEVICES=2 python drive_ourmodel.py  --data='Pheme' --tag=$tag   --da='True' --lr=0.001 --log='dapheme-0.5-kernel-1-2-4-8'  --phase='train' --epochs=20  --max_iter=10 --lambda1=0.1 --tsigma="1#2#4#8" --vsigma="1#2#4#8" --lambda2=0.5 --temperature=0.5  --threshold=0.5 --ctsize=64 &

# calculate domain difference
#CUDA_VISIBLE_DEVICES=1 python display_diff.py --data='Twitter'  --tag=$tag  --da='False' --lr=0.001 --log='dgtwitter'   --phase='test' --epochs=30  --max_iter=10 --lambda1=1 --tsigma="2#4#8#12" --vsigma="2#4#8#12"  --lambda2=0.9 --temperature=0.5 --threshold=0.5 --ctsize=64 &
#CUDA_VISIBLE_DEVICES=1 python display_diff.py --data='Twitter'  --tag=$tag  --da='True' --lr=0.001 --log='datwitter'   --phase='analysis' --epochs=30  --max_iter=10 --lambda1=1 --tsigma="2#4#8#12" --vsigma="2#4#8#12"  --lambda2=0.9 --temperature=0.5 --threshold=0.5 --ctsize=64 &
#CUDA_VISIBLE_DEVICES=0 python display_diff.py --data='Twitter'  --tag=$tag  --da='False' --lr=0.001 --log='orgtwitter'   --phase='analysis' --epochs=30  --max_iter=10 --lambda1=0 --tsigma="2#4#8#12" --vsigma="2#4#8#12"  --lambda2=0 --temperature=0.5 --threshold=0.5 --ctsize=64 &

#CUDA_VISIBLE_DEVICES=2 python display_diff.py  --data='Pheme' --tag=$tag   --da='True' --lr=0.001 --log='dapheme-0.5'  --phase='test' --epochs=20  --max_iter=10 --lambda1=0.1 --tsigma="2#10#30#40" --vsigma="2#10#30#40"  --lambda2=0.5 --temperature=0.5  --threshold=0.5 --ctsize=64 &
#CUDA_VISIBLE_DEVICES=2 python display_diff.py  --data='Pheme' --tag=$tag   --da='True' --lr=0.001 --log='dapheme-0.5-kernel'  --phase='analysis' --epochs=20  --max_iter=10 --lambda1=0.1 --tsigma="2#4#8#16" --vsigma="2#4#8#16"  --lambda2=0.5 --temperature=0.5  --threshold=0.5 --ctsize=64 &
#CUDA_VISIBLE_DEVICES=2 python display_diff.py  --data='Pheme' --tag=$tag   --da='False' --lr=0.001 --log='dgpheme'  --phase='test' --epochs=20  --max_iter=10 --lambda1=0.1 --tsigma="2#10#30#40" --vsigma="2#10#30#40"  --lambda2=0.5 --temperature=0.5  --threshold=0.5 --ctsize=64 &
#CUDA_VISIBLE_DEVICES=2 python display_diff.py --data='Pheme' --tag=$tag   --da='True' --lr=0.001 --log='dapheme-0.5-kernel'  --phase='analysis' --epochs=20  --max_iter=10 --lambda1=0.1 --tsigma="2#10#30#40" --vsigma="2#10#30#40"  --lambda2=0.5 --temperature=0.5  --threshold=0.5 --ctsize=64 &


#CUDA_VISIBLE_DEVICES=2 python drive_ourmodel.py --data='Twitter'  --tag=$tag  --da='False' --lr=0.001 --log='dgtem1'   --phase=$phase --epochs=30  --max_iter=10 --lambda1=0 --tsigma="2#4#8#12" --vsigma="2#4#8#12"  --lambda2=1 --temperature=1 --threshold=0.5 --ctsize=64 &
#CUDA_VISIBLE_DEVICES=2 python drive_ourmodel.py --data='Twitter'  --tag=$tag  --da='False' --lr=0.001 --log='dgtem10'   --phase=$phase --epochs=30  --max_iter=10 --lambda1=0 --tsigma="2#4#8#12" --vsigma="2#4#8#12"  --lambda2=1 --temperature=10 --threshold=0.5 --ctsize=64
#CUDA_VISIBLE_DEVICES=0 python drive_ourmodel.py --data='Twitter'  --tag=$tag  --da='False' --lr=0.001 --log='dgtem5'   --phase=$phase --epochs=30  --max_iter=10 --lambda1=0 --tsigma="2#4#8#12" --vsigma="2#4#8#12"  --lambda2=1 --temperature=5 --threshold=0.5 --ctsize=64 &

#CUDA_VISIBLE_DEVICES=1 python drive_ourmodel.py --data='Pheme' --tag=$tag   --da='True' --lr=0.001 --log='dgtem0.5'  --phase=$phase --epochs=20  --max_iter=10 --lambda1=0.01 --tsigma="2#10#30#40" --vsigma="2#10#30#40"  --lambda2=1 --temperature=0.01  --threshold=0.5 --ctsize=64 &
#CUDA_VISIBLE_DEVICES=1 python drive_ourmodel.py --data='Pheme' --tag=$tag   --da='True' --lr=0.001 --log='dgtem0.5'  --phase=$phase --epochs=20  --max_iter=10 --lambda1=0.001 --tsigma="2#10#30#40" --vsigma="2#10#30#40"  --lambda2=1 --temperature=0.1  --threshold=0.5 --ctsize=64 &
#CUDA_VISIBLE_DEVICES=2 python drive_ourmodel.py --data='Pheme' --tag=$tag   --da='True' --lr=0.001 --log='orgpheme'  --phase=$phase --epochs=20  --max_iter=10 --lambda1=0.001 --tsigma="2#10#30#40" --vsigma="2#10#30#40"  --lambda2=1 --temperature=0.5  --threshold=0.5 --ctsize=64 &
#CUDA_VISIBLE_DEVICES=2 python drive_ourmodel.py --data='Pheme' --tag=$tag   --da='True' --lr=0.001 --log='orgpheme'  --phase=$phase --epochs=20  --max_iter=10 --lambda1=0.001 --tsigma="2#10#30#40" --vsigma="2#10#30#40"  --lambda2=1 --temperature=0.5  --threshold=0.5 --ctsize=64 &

#CUDA_VISIBLE_DEVICES=2 python drive_ourmodel.py --data='Pheme' --tag=$tag   --da='True' --lr=0.001 --log='dgtem0.5'  --phase=$phase --epochs=20  --max_iter=10 --lambda1=0.001 --tsigma="2#4#8#16" --vsigma="2#4#8#16#"  --lambda2=1 --temperature=0.5  --threshold=0.5 --ctsize=64 &


#CUDA_VISIBLE_DEVICES=2 python drive_ourmodel.py --data='Pheme' --tag=$tag   --da='True' --lr=0.001 --log='dgtem0.5'  --phase=$phase --epochs=20  --max_iter=10 --lambda1=0.005 --tsigma="2#10#30#40" --vsigma="2#10#30#40"  --lambda2=1 --temperature=0.01  --threshold=0.5 --ctsize=64 &



#CUDA_VISIBLE_DEVICES=3 python drive_ourmodel.py --data='Pheme' --tag=$tag   --da='True' --lr=0.001 --log='dgtem0.5'  --phase=$phase --epochs=20  --max_iter=10 --lambda1=0.1 --tsigma="2#10#30#40" --vsigma="2#10#30#40"  --lambda2=1 --temperature=10  --threshold=0.5 --ctsize=64
#CUDA_VISIBLE_DEVICES=1 python drive_ourmodel.py --data='Pheme' --tag=$tag   --da='True' --lr=0.001 --log='dgtem0.5'  --phase=$phase --epochs=20  --max_iter=10 --lambda1=0.1 --tsigma="2#10#30#40" --vsigma="2#10#30#40"  --lambda2=5 --temperature=10  --threshold=0.5 --ctsize=64 &
#CUDA_VISIBLE_DEVICES=3 python drive_ourmodel.py --data='Pheme' --tag=$tag   --da='True' --lr=0.001 --log='dgtem0.5'  --phase=$phase --epochs=20  --max_iter=10 --lambda1=0.1 --tsigma="2#10#30#40" --vsigma="2#10#30#40"  --lambda2=10 --temperature=10  --threshold=0.5 --ctsize=64
#CUDA_VISIBLE_DEVICES=1 python drive_ourmodel.py --data='Pheme' --tag=$tag   --da='True' --lr=0.001 --log='dgtem0.5'  --phase=$phase --epochs=20  --max_iter=10 --lambda1=0.05 --tsigma="2#10#30#40" --vsigma="2#10#30#40"  --lambda2=1 --temperature=10  --threshold=0.5 --ctsize=64 &









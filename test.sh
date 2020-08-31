#Rain100L
python test.py --logdir logs/Rain100L/ --save_path results/Rain100L/ --data_path datasets/test/Rain100L/rainy_images/

#Rain100H
python test.py --logdir logs/Rain100H/ --save_path results/Rain100H/ --data_path datasets/test/Rain100H/rainy_images/

#Rain12
python test.py --logdir logs/Rain100L/ --save_path results/Rain12/ --data_path datasets/test/Rain12/rainy_images/

#RainLight
python test.py --logdir logs/RainLight/ --save_path results/RainLight/ --data_path datasets/test/RainLightTest/rainy_images/

#RainHeavy
python test.py --logdir logs/RainHeavy/ --save_path results/RainHeavy/ --data_path datasets/test/RainHeavyTest/rainy_images/

#RainDDN
python test.py --logdir logs/RainTrainDDN/ --save_path results/DDN/ --data_path datasets/test/DDNTest/rainy_images/

#RainDID
python test.py --logdir logs/RainTrainDID/ --save_path results/DID/ --data_path datasets/test/DIDTest/rainy_images/

#Real_finetune
python test.py --logdir logs/Real/finetune/ --save_path results/Real/finetune/ --data_path datasets/test/Real/rainy_images/

#Real_no_finetune
python test.py --logdir logs/Real/no_finetune/ --save_path results/Real/no_finetune/ --data_path datasets/test/Real/rainy_images/


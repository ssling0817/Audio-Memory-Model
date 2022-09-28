# Audio-Memory-Model
```
python test.py --filename 'audios/clips'
 - output will be file/output.csv
```

```
python train.py
 - need train.csv, user_details.csv to run
 - training files should be in ./audios/clips/
 - model structure
```
	Model: "model"
	__________________________________________________________________________________________________
	 Layer (type)                   Output Shape         Param #     Connected to                     
	==================================================================================================
	 input_1 (InputLayer)           [(None, 34)]         0           []                               
		                                                                                          
	 dense (Dense)                  (None, 512)          17920       ['input_1[0][0]']                
		                                                                                          
	 dense_1 (Dense)                (None, 256)          131328      ['dense[0][0]']                  
		                                                                                          
	 input_2 (InputLayer)           [(None, 1, 110250)]  0           []                               
		                                                                                          
	 dense_2 (Dense)                (None, 128)          32896       ['dense_1[0][0]']                
		                                                                                          
	 lstm (LSTM)                    (None, 1, 128)       56514048    ['input_2[0][0]']                
		                                                                                          
	 dense_3 (Dense)                (None, 64)           8256        ['dense_2[0][0]']                
		                                                                                          
	 lstm_1 (LSTM)                  (None, 1, 64)        49408       ['lstm[0][0]']                   
		                                                                                          
	 dropout (Dropout)              (None, 64)           0           ['dense_3[0][0]']                
		                                                                                          
	 dropout_1 (Dropout)            (None, 1, 64)        0           ['lstm_1[0][0]']                 
		                                                                                          
	 dense_4 (Dense)                (None, 32)           2080        ['dropout[0][0]']                
		                                                                                          
	 lstm_2 (LSTM)                  (None, 32)           12416       ['dropout_1[0][0]']              
		                                                                                          
	 concatenate (Concatenate)      (None, 64)           0           ['dense_4[0][0]',                
		                                                          'lstm_2[0][0]']                 
		                                                                                          
	 dense_5 (Dense)                (None, 64)           4160        ['concatenate[0][0]']            
		                                                                                          
	 dense_6 (Dense)                (None, 16)           1040        ['dense_5[0][0]']                
		                                                                                          
	 dense_7 (Dense)                (None, 1)            17          ['dense_6[0][0]']                
		                                                                                          
	==================================================================================================
	Total params: 56,773,569
	Trainable params: 56,773,569
	Non-trainable params: 0
	__________________________________________________________________________________________________

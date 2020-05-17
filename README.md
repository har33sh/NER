Solution:


Named Entity Recognition(NER) is used to find the Entity type of words. The selection of the right model for the task is an important process, as it can help us in achieving better accuracy. In this case, we will be using Recurrent Neural Network(RNN), in particular, Long Short Term Memory(LSTM), as this considers the sequence order of the data given. 

PS: Some of the code submitted have been reused from the references mentioned below

Steps to run : 
1. Install Virtual Environment

        pip install virtualenv

2. Create/Activate Virtual Environment:
       
        virtualenv venv
        source venv/bin/activate
        
3. Install Requirements      
    
        pip install -r Requirements.txt
    
4. Training : 

        python train.py
        
     Pre-processing - By default the pre-processing is disabled as it takes a lot of time, if you want to enable it please goto train.py and uncomment (line 196)
        
        data = pre_process(dataset)


5. Testing/Deployment: 
            
       python test.py


All the CONFIG parameters are present in the top of the file.

Requirements:
Please look at Requirements.txt


Evaluation:
    
    F1 Score: 96.4%
    Training accuracy: 0.9992
    Testing accuracy: 0.9991

References : 

1. https://www.depends-on-the-definition.com/guide-sequence-tagging-neural-networks-python/
2. https://www.kaggle.com/alouiamine/ner-using-bidirectional-lstm
3. https://www.kaggle.com/navya098/bi-lstm-for-ner
4. https://www.kaggle.com/alouiamine/ner-using-bidirectional-lstm/notebook

# stock prediction with LSTM-DNN
stock price prediction with modified LSTM model

## WHAT IT CAN DO?
It can give the prediction value of one stock's closing price. It can not predict too far away from now and all it can do is to predict tomorrow's prcie. What we need to do is feed the data in the past of one stock.

## EVALUATION
I evaluate the model with MAPE(mean absolute percent error). The best performance of some stocks trained by this model can get the MAPE to 2%-3% which means the accuracy(1-MAPE) can achieve 97%-98%.

## TRAIN YOUR OWN DATA
U first download the csv data of your stock from the internet and then modify the csv route in the code(put the closing pricing in the last column). It takes hours to learn a well-performed model with CPU.

## DOES IT USEFUL?
Although it gets good performance to predict next day's closing pricing, it can not predict the next 2,3,4... days' prcies. So it may be helpful, may be not.

## FUTURE
1) I will put my stock price prediction system on my github with lots of new functions including preprocessing the data, training the models, predicting prices, evaluating the models and recommending high yield stocks. At the same time, it will visualize these results clearly. 
2) Long term prediction will be experimented.
3) Factors of the news will be considerd

## INSTALL
python 3.7

tensorflow 1.4

pandas

numpy


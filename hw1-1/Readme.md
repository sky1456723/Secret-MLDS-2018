# HW1-1
Simulate a Function:
    1. Triangular function:
        trig_func.py :
            Building three different models, all with almost 13,600 parameters.
            It simulates the triangular function.
            Default epoch number is 1,000. 20,000 epochs tried.
    2. Sinc function:
        sinc_func.py :
            Building three different models, all with almost 13,600 parameters.
            It simulates the sinc function.
            Default epoch number is 1,000. 20,000 epochs tried.
    Both programs will draw a graph to show the simulation results.
    Running at different time gets different results. But, mostly deep models perform better. 
        Even sometimes the initialization greatly influences the result.
    With MSE as the loss function, MAE is also shown for comparison. 
    Also, difference between the simulated function and learned result is shown in another graph.

Actual task part :  
    DNN_mnist.py :  
        This script will download mnist data by using keras api in tensorflow, and build two models.  
        One is deep (three hidden layers), having 99710 parameters.  
        One is shallow (two hidden layers), having 99822 parameters.  
        The script is written in tensorflow1.10.0 and python3.6.5.  
        Please use instruction in terminal like: "python DNN_mnist.py"  
        It will plot two charts(loss and acc).  

    CNN.py:  
        To run, type "python CNN.py" in terminal.  
        Python 3.6 is recommended.  
        This script trains 3 CNN models and plots their accuracies and losses after 300 steps.  
        The figures are saved as "Loss.png" and "Accuracy.png"


# Design Below Neural Network Architecture

   take 2 inputs:
 
     an image from the MNIST dataset (say 5), and
     a random number between 0 and 9, (say 7)
   
   and gives two outputs:
 
    the "number" that was represented by the MNIST image (predict 5), and
    the "sum" of this number with the random number and the input image to the network (predict 5 + 7 = 12)

![image](https://user-images.githubusercontent.com/70502759/136892002-fa6fad37-bab3-4f82-8a48-ef43557526b8.png)


 Input Data

  MNIST Image
   


  ![image](https://user-images.githubusercontent.com/70502759/136895005-4cb01984-b509-43cc-935b-7722036b413b.png)
  
  
  Random Number
  
  We have generate the random number dynamically with the help of following function by giving the batchsize 
     
     from random import randint
     randint(0,9)        
     
     [9, 0, 3, 7, 8, 5, 4, 1, 2, 6]
      
     

    
      
We have input image of shape (1,28,28), where 28*28 is height and  weidht of the image and  we have only one channel as it's a gray image 
We have a single random number which we have to add with the MNIST image number and will try to predicate the respective sum . 


Random Number converted to the one hot vector which can be used as an input to the neural network .

       [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]

 


As we can predicate the maximum sum as 18 so including zero we have a total 19 class of prediction for the sum and 10 class of prediction for mnist number and total class is 29. We have prepared the actual labels that can be used for loss calculation after the model predicted handwritten mnist number and sum.


      
    Actual Mnist Number 
      [9, 0, 0, 3, 0, 2, 7, 2, 5, 5]       
    Random Number
      [9, 0, 3, 7, 8, 5, 4, 1, 2, 6]

    One Hot Represenatation 
       [[0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]])


Model Summery
   
FashionCNNI(
  (layer1): Sequential(
    (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (layer2): Sequential(
    (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (fc1): Linear(in_features=2304, out_features=600, bias=True)
  (drop): Dropout2d(p=0.25, inplace=False)
  (fc2): Linear(in_features=600, out_features=120, bias=True)
  (fc3): Linear(in_features=120, out_features=20, bias=True)
  (fc3_a): Linear(in_features=10, out_features=20, bias=True)
  (fc5_a): Linear(in_features=40, out_features=20, bias=True)
  (out): Linear(in_features=20, out_features=10, bias=True)
  (out_a): Linear(in_features=20, out_features=19, bias=True)
)


Combined The Two Inputs

 Initially we have passed through  conv layers   and we have extracted the feature map. We have passed feature map into linear neural layer of output size(input 120, output 20) .  Random number we have passed through one Linear neural layer of output size(input 10,output 20) and concatenated both the outputs(20+20)and passed through linear layer  which resulted in output size (input 40,output 20) one dimension vector . Again this passed through two linear layer and give the two output of one representing the number prediction and other prediction is addition with random number . Hence model forward function return 2 output .



Loss function

 Since this is a classification problem, the choice of loss function may seem obvious – the CrossEntropy loss . 


Prediction at initial Layer : 

![image](https://user-images.githubusercontent.com/70502759/211008290-6c388c9a-d501-4ec1-aaf5-41665461e17a.png)

![image](https://user-images.githubusercontent.com/70502759/211008386-577eb923-ed14-4492-ab4d-f50a37d0bd99.png)


Prediction at final Layer : 

![image](https://user-images.githubusercontent.com/70502759/211008613-2b287a94-ed45-49d5-9930-b6ea9810b4a5.png)




## Tech Stack

**Client:** Python, Pytorch, Numpy


  

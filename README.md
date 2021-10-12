
# We have to design the following  Neural Network Architecture

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
     
      def getRandomNumber(batchsize):
         return random.sample(range(0, 10), batchsize)
      
         
      getRandomNumber(10)   
   
      [9, 0, 3, 7, 8, 5, 4, 1, 2, 6]
      
     

    
      
We have input image of shape 1*28*28 , where 28*28 is height and weight of the image and the number of channel and we have only one as it's a gray channel 
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

 


As we can predicate maximum sum as 18 so including zero we have total 19 class of prediction for the sum and 10 class of prediction  for mnist number and total class is 29
we have prepared the actual labels the can used for loss calculation after the model predicted handwritten mnist number and sum .

      
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
   
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 6, 24, 24]             156
         
            Conv2d-2             [-1, 12, 8, 8]           1,812
            Linear-3                   [-1, 20]             220
            Linear-4                   [-1, 40]             840
            Linear-5                  [-1, 120]          27,960
            Linear-6                   [-1, 60]           7,260
            Linear-7                   [-1, 29]           1,769
================================================================

 Total params: 40,017
 
 Trainable params: 40,017

 Non-trainable params: 0


  Input size (MB): 0.03

  Forward/backward pass size (MB): 0.03

  Params size (MB): 0.15

  Estimated Total Size (MB): 0.22



## Tech Stack

**Client:** Python, Pytorch, Numpy


  

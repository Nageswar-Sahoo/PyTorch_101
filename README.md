
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
   
![image](https://user-images.githubusercontent.com/70502759/211008908-3b311018-f0d9-4b19-875b-a1fa29275c2e.png)


Combined The Two Inputs

 Initially we have passed through  conv layers   and we have extracted the feature map. We have passed feature map into linear neural layer of output size(input 120, output 20) .  Random number we have passed through one Linear neural layer of output size(input 10,output 20) and concatenated both the outputs(20+20)and passed through linear layer  which resulted in output size (input 40,output 20) one dimension vector . Again this passed through two linear layer and give the two output of one representing the number prediction and other prediction is addition with random number . Hence model forward function return 2 output .



Loss function

 Since this is a classification problem, the choice of loss function may seem obvious – the CrossEntropy loss . 


Prediction at initial Layer : 

![image](https://user-images.githubusercontent.com/70502759/211008290-6c388c9a-d501-4ec1-aaf5-41665461e17a.png)

![image](https://user-images.githubusercontent.com/70502759/211008386-577eb923-ed14-4492-ab4d-f50a37d0bd99.png)


Prediction at final Layer : 

![image](https://user-images.githubusercontent.com/70502759/211008613-2b287a94-ed45-49d5-9930-b6ea9810b4a5.png)


After 10 epoch of training :

Iterations vs Loss:

![image](https://user-images.githubusercontent.com/70502759/211010939-610f6944-0a06-40f6-a479-be25acb13c7c.png)


Iterations vs Accuracy Number:

![image](https://user-images.githubusercontent.com/70502759/211010971-fbdc67a4-cd59-429d-978c-fe2698bfe879.png)



Iterations vs Accuracy Number Addition:

![image](https://user-images.githubusercontent.com/70502759/211011025-dc89d193-b590-44d5-b8cf-d3010be95ba3.png)
   

After 100 epoch of training :
Iterations vs Loss:

![image](https://user-images.githubusercontent.com/70502759/211009525-8b626097-7204-45bd-9667-39a9f2fb6c5d.png)

Iterations vs Accuracy Number:

![image](https://user-images.githubusercontent.com/70502759/211009627-951347f0-4d94-415a-8a32-f09a413eaf1b.png)

Iterations vs Accuracy Number Addition:

![image](https://user-images.githubusercontent.com/70502759/211009686-5bcf29f0-1a1b-4e6d-87c6-0cca41f88fdc.png)






## Tech Stack

**Client:** Python, Pytorch, Numpy


  
